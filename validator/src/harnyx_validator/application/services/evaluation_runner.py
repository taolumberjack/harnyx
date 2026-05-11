"""Helper to run miner tasks for a single artifact."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import httpx

from harnyx_commons.application.dto.session import SessionEnvelope, SessionIssued, SessionTokenRequest
from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import (
    EvaluationDetails,
    EvaluationError,
    MinerTask,
    MinerTaskErrorCode,
    is_delivery_disqualifying_validator_pair_error,
)
from harnyx_commons.domain.session import SessionStatus
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.errors import SessionBudgetExhaustedError
from harnyx_commons.llm.provider import LlmRetryExhaustedError
from harnyx_commons.miner_task_failure_policy import (
    SANDBOX_DETAIL_CODE_UNHANDLED_EXCEPTION,
    TERMINAL_TIMEOUT_ERROR_MESSAGE,
    TIMEOUT_REVIEW_MAX_OBSERVATIONS,
    TIMEOUT_TPS_SLOWDOWN_FACTOR,
    ProviderFailureEvidence,
    TimeoutAttributionKind,
    TimeoutObservationEvidence,
    classify_timeout_attribution,
    is_provider_caused_terminal_failure,
    is_script_validation_sandbox_invocation,
    is_timeout_sandbox_invocation,
    provider_batch_failure_evidence,
    provider_batch_failure_message,
    slowest_successful_llm_tps,
    successful_llm_samples,
)
from harnyx_validator.application.dto.evaluation import (
    MinerTaskRunRequest,
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
    TaskRunOutcome,
    TokenUsageSummary,
)
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator, UsageSummarizer
from harnyx_validator.application.invoke_entrypoint import (
    MinerResponseValidationError,
    SandboxInvocationError,
)
from harnyx_validator.application.ports.evaluation_record import EvaluationRecordPort
from harnyx_validator.application.ports.progress import ProgressRecorder
from harnyx_validator.application.ports.subtensor import SubtensorClientPort
from harnyx_validator.domain.evaluation import MinerTaskRun

if TYPE_CHECKING:
    from harnyx_validator.application.scheduler import SchedulerConfig

Clock = Callable[[], datetime]
SubmissionFactory = Callable[[MinerTask, SessionIssued], Awaitable[MinerTaskRunSubmission]]
CompletedArtifactBaseline = Callable[[], float | None]

logger = logging.getLogger("harnyx_validator.scheduler")
measurement_logger = logging.getLogger("harnyx_validator.measurement")
LOCAL_RETRY_ATTEMPTS = 2


def _elapsed_ms(*, issued_at: datetime, completed_at: datetime) -> float:
    return (completed_at - issued_at).total_seconds() * 1000.0


def _monotonic_elapsed_ms(*, started_at: float, completed_at: float) -> float:
    return round((completed_at - started_at) * 1000.0, 3)


def _log_session_finished(
    *,
    batch_id: UUID,
    session_id: UUID,
    artifact_id: UUID,
    task_id: UUID,
    uid: int,
    attempt_count: int,
    session_ms: float,
    terminal_outcome: str,
    error_code: str | None,
) -> None:
    measurement_logger.info(
        "miner-task session finished",
        extra={
            "data": {
                "batch_id": str(batch_id),
                "session_id": str(session_id),
                "artifact_id": str(artifact_id),
                "task_id": str(task_id),
                "uid": uid,
                "attempt_count": attempt_count,
                "session_ms": session_ms,
                "terminal_outcome": terminal_outcome,
                "error_code": error_code,
            }
        },
    )


class AttemptControlKind(StrEnum):
    # Outer control-flow action for one task-attempt loop step.
    SUBMISSION = "submission"
    RETRY = "retry"
    REVIEW_TIMEOUT = "review_timeout"
    TIMEOUT_UNRESOLVED = "timeout_unresolved"
    VALIDATOR_BATCH_FAILURE = "validator_batch_failure"


@dataclass(frozen=True, slots=True)
class TaskAttemptDecision:
    kind: AttemptControlKind
    submission: MinerTaskRunSubmission | None = None
    retry_exc: Exception | None = None
    timeout_exc: SandboxInvocationError | None = None
    timeout_observation: TimeoutObservationEvidence | None = None
    successful_baseline_tps: float | None = None
    validator_failure: ValidatorBatchFailedError | None = None


@dataclass(frozen=True, slots=True)
class ValidatorBatchFailureDetail:
    error_code: str
    error_message: str
    occurred_at: datetime
    artifact_id: UUID | None = None
    task_id: UUID | None = None
    uid: int | None = None
    exception_type: str | None = None
    traceback: str | None = None


class ValidatorBatchFailedError(RuntimeError):
    def __init__(
        self,
        *,
        error_code: str,
        message: str,
        failure_detail: ValidatorBatchFailureDetail,
        completed_submissions: tuple[MinerTaskRunSubmission, ...] | None = None,
        remaining_tasks: tuple[MinerTask, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.failure_detail = failure_detail
        self.completed_submissions = completed_submissions
        self.remaining_tasks = remaining_tasks


class ArtifactExecutionFailedError(RuntimeError):
    def __init__(
        self,
        *,
        error_code: MinerTaskErrorCode,
        message: str,
        failure_detail: ValidatorBatchFailureDetail,
        completed_submissions: tuple[MinerTaskRunSubmission, ...],
        remaining_tasks: tuple[MinerTask, ...],
        artifact_breaker_tripped: bool = False,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.failure_detail = failure_detail
        self.completed_submissions = completed_submissions
        self.remaining_tasks = remaining_tasks
        self.artifact_breaker_tripped = artifact_breaker_tripped


class UnexpectedArtifactExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        cause: Exception,
        completed_submissions: tuple[MinerTaskRunSubmission, ...],
        remaining_tasks: tuple[MinerTask, ...],
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.completed_submissions = completed_submissions
        self.remaining_tasks = remaining_tasks


@dataclass(slots=True)
class _ArtifactDispatchState:
    submissions_by_index: list[MinerTaskRunSubmission | None]
    unresolved_tasks_by_index: dict[int, MinerTask] = field(default_factory=dict)
    timeout_observations_by_pair: dict[tuple[UUID, UUID], tuple[TimeoutObservationEvidence, ...]] = field(
        default_factory=dict
    )
    slowest_successful_tps: float | None = None
    validator_failure: ValidatorBatchFailedError | None = None
    unexpected_failure: Exception | None = None


@dataclass(frozen=True, slots=True)
class ArtifactFailure:
    error_code: MinerTaskErrorCode
    message: str
    failure_detail: ValidatorBatchFailureDetail
    artifact_breaker_tripped: bool = False


@dataclass(frozen=True, slots=True)
class ArtifactEvaluationOutcome:
    submissions: tuple[MinerTaskRunSubmission, ...]
    unresolved_tasks: tuple[MinerTask, ...]
    timeout_observations_by_pair: dict[tuple[UUID, UUID], tuple[TimeoutObservationEvidence, ...]]
    slowest_successful_tps: float | None = None
    artifact_failure: ArtifactFailure | None = None


class EvaluationRunner:
    """Executes miner task runs for artifacts and records submissions."""

    def __init__(
        self,
        *,
        subtensor_client: SubtensorClientPort,
        session_manager: SessionManager,
        evaluation_records: EvaluationRecordPort,
        receipt_log: ReceiptLogPort,
        config: SchedulerConfig,
        clock: Clock,
        progress: ProgressRecorder | None = None,
        usage_summarizer: UsageSummarizer | None = None,
    ) -> None:
        self._subtensor = subtensor_client
        self._sessions = session_manager
        self._evaluation_records = evaluation_records
        self._receipts = receipt_log
        self._config = config
        self._clock = clock
        self._progress = progress
        self._usage = usage_summarizer or UsageSummarizer()
        self._validator_uid: int | None = None

    async def evaluate_artifact(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        orchestrator: TaskRunOrchestrator,
    ) -> ArtifactEvaluationOutcome:
        outcome = ArtifactEvaluationOutcome(
            submissions=(),
            unresolved_tasks=tuple(tasks),
            timeout_observations_by_pair={},
            slowest_successful_tps=None,
        )
        while outcome.unresolved_tasks:
            outcome = await self.evaluate_artifact_with_state(
                batch_id=batch_id,
                artifact=artifact,
                tasks=outcome.unresolved_tasks,
                orchestrator=orchestrator,
                successful_baseline_tps=outcome.slowest_successful_tps,
                timeout_observations_by_pair=outcome.timeout_observations_by_pair,
                earlier_submissions=outcome.submissions,
            )
        return outcome

    async def evaluate_artifact_with_state(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        orchestrator: TaskRunOrchestrator,
        successful_baseline_tps: float | None,
        completed_artifact_baseline: CompletedArtifactBaseline | None = None,
        timeout_observations_by_pair: dict[tuple[UUID, UUID], tuple[TimeoutObservationEvidence, ...]],
        earlier_submissions: tuple[MinerTaskRunSubmission, ...] = (),
    ) -> ArtifactEvaluationOutcome:
        indexed_tasks = tuple(enumerate(tasks))
        if not indexed_tasks:
            return ArtifactEvaluationOutcome(
                submissions=earlier_submissions,
                unresolved_tasks=(),
                timeout_observations_by_pair=dict(timeout_observations_by_pair),
                slowest_successful_tps=successful_baseline_tps,
            )

        dispatch = _ArtifactDispatchState(
            submissions_by_index=[None] * len(indexed_tasks),
            timeout_observations_by_pair=dict(timeout_observations_by_pair),
            slowest_successful_tps=successful_baseline_tps,
        )
        pending_tasks: asyncio.Queue[tuple[int, MinerTask]] = asyncio.Queue()
        for indexed_task in indexed_tasks:
            pending_tasks.put_nowait(indexed_task)

        workers = [
            asyncio.create_task(
                self._run_artifact_worker(
                    batch_id=batch_id,
                    artifact=artifact,
                    orchestrator=orchestrator,
                    pending_tasks=pending_tasks,
                    dispatch=dispatch,
                    completed_artifact_baseline=completed_artifact_baseline,
                )
            )
            for _ in range(
                min(
                    max(1, self._config.artifact_task_parallelism),
                    len(indexed_tasks),
                )
            )
        ]
        await asyncio.gather(*workers)
        completed_submissions = tuple(
            submission for submission in dispatch.submissions_by_index if submission is not None
        )
        all_completed_submissions = (*earlier_submissions, *completed_submissions)
        if dispatch.validator_failure is not None:
            remaining_tasks = tuple(
                task for index, task in indexed_tasks if dispatch.submissions_by_index[index] is None
            )
            raise ValidatorBatchFailedError(
                error_code=dispatch.validator_failure.error_code,
                message=str(dispatch.validator_failure),
                failure_detail=dispatch.validator_failure.failure_detail,
                completed_submissions=all_completed_submissions,
                remaining_tasks=remaining_tasks,
            ) from dispatch.validator_failure
        if dispatch.unexpected_failure is not None:
            remaining_tasks = tuple(
                task for index, task in indexed_tasks if dispatch.submissions_by_index[index] is None
            )
            raise UnexpectedArtifactExecutionError(
                cause=dispatch.unexpected_failure,
                completed_submissions=all_completed_submissions,
                remaining_tasks=remaining_tasks,
            ) from dispatch.unexpected_failure

        unresolved_tasks = tuple(
            task for _, task in sorted(dispatch.unresolved_tasks_by_index.items())
        )
        return ArtifactEvaluationOutcome(
            submissions=all_completed_submissions,
            unresolved_tasks=unresolved_tasks,
            timeout_observations_by_pair=dispatch.timeout_observations_by_pair,
            slowest_successful_tps=dispatch.slowest_successful_tps,
        )

    async def _run_artifact_worker(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        orchestrator: TaskRunOrchestrator,
        pending_tasks: asyncio.Queue[tuple[int, MinerTask]],
        dispatch: _ArtifactDispatchState,
        completed_artifact_baseline: CompletedArtifactBaseline | None = None,
    ) -> None:
        while True:
            if dispatch.validator_failure is not None or dispatch.unexpected_failure is not None:
                return
            try:
                task_index, task = pending_tasks.get_nowait()
            except asyncio.QueueEmpty:
                return

            try:
                pair_key = (artifact.artifact_id, task.task_id)
                decision = await self._evaluate_task_with_retry(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    orchestrator=orchestrator,
                    successful_baseline_tps=dispatch.slowest_successful_tps,
                    completed_artifact_baseline=completed_artifact_baseline,
                    prior_timeout_observations=dispatch.timeout_observations_by_pair.get(pair_key, ()),
                )
                if decision.kind is AttemptControlKind.SUBMISSION:
                    submission = _require_submission(decision)
                    dispatch.submissions_by_index[task_index] = submission
                    dispatch.slowest_successful_tps = _merge_slowest_successful_tps(
                        dispatch.slowest_successful_tps,
                        decision.successful_baseline_tps,
                    )
                    dispatch.timeout_observations_by_pair.pop(pair_key, None)
                    error_code = _submission_error_code_or_none(submission)
                    if (
                        error_code is not None
                        and is_delivery_disqualifying_validator_pair_error(error_code)
                        and dispatch.validator_failure is None
                    ):
                        dispatch.validator_failure = _validator_batch_failed_from_existing_submission(
                            submission=submission,
                            artifact=artifact,
                            task=task,
                            occurred_at=self._clock(),
                        )
                    continue

                if decision.kind is AttemptControlKind.TIMEOUT_UNRESOLVED:
                    observation = _require_timeout_observation(decision)
                    prior_observations = dispatch.timeout_observations_by_pair.get(pair_key, ())
                    dispatch.timeout_observations_by_pair[pair_key] = (*prior_observations, observation)
                    dispatch.unresolved_tasks_by_index[task_index] = task
                    continue

                if decision.kind is AttemptControlKind.VALIDATOR_BATCH_FAILURE:
                    if dispatch.validator_failure is None:
                        dispatch.validator_failure = _require_validator_failure(decision)
                    continue

                raise RuntimeError("unexpected non-terminal decision from task retry loop")
            except ValidatorBatchFailedError as exc:
                if exc.completed_submissions:
                    dispatch.submissions_by_index[task_index] = _require_single_completed_submission(exc)
                if dispatch.validator_failure is None:
                    dispatch.validator_failure = exc
            except Exception as exc:
                if dispatch.unexpected_failure is None:
                    dispatch.unexpected_failure = exc

    async def _run_tasks_with_sessions(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        create_submission: SubmissionFactory,
    ) -> list[MinerTaskRunSubmission]:
        submissions: list[MinerTaskRunSubmission] = []
        for index, task in enumerate(tasks):
            issued = self._issue_session(
                batch_id=batch_id,
                uid=artifact.uid,
                task=task,
            )
            try:
                submissions.append(await create_submission(task, issued))
            except Exception as exc:
                raise UnexpectedArtifactExecutionError(
                    cause=exc,
                    completed_submissions=tuple(submissions),
                    remaining_tasks=tuple(tasks[index:]),
                ) from exc
            finally:
                self._clear_task_session(issued.session.session_id)
                self._sessions.revoke(issued.session.session_id)
        return submissions

    async def _evaluate_task_with_retry(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        orchestrator: TaskRunOrchestrator,
        successful_baseline_tps: float | None,
        completed_artifact_baseline: CompletedArtifactBaseline | None = None,
        prior_timeout_observations: tuple[TimeoutObservationEvidence, ...],
    ) -> TaskAttemptDecision:
        issued = self._issue_session(
            batch_id=batch_id,
            uid=artifact.uid,
            task=task,
        )
        session_started_at = time.monotonic()
        attempt_count = 0
        terminal_outcome = "unexpected"
        error_code: str | None = None
        try:
            for attempt_number in range(1, LOCAL_RETRY_ATTEMPTS + 1):
                attempt_count = attempt_number
                self._sessions.begin_attempt(issued.session.session_id)
                decision = await self._evaluate_task_attempt(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    issued=issued,
                    orchestrator=orchestrator,
                    final_attempt=attempt_number >= LOCAL_RETRY_ATTEMPTS,
                )
                if decision.kind is AttemptControlKind.SUBMISSION:
                    terminal_outcome = AttemptControlKind.SUBMISSION.value
                    error_code = _submission_error_code_or_none(_require_submission(decision))
                    return decision

                if decision.kind is AttemptControlKind.REVIEW_TIMEOUT:
                    current_successful_baseline_tps = _merge_slowest_successful_tps(
                        successful_baseline_tps,
                        None if completed_artifact_baseline is None else completed_artifact_baseline(),
                    )
                    try:
                        timeout_resolution = self._resolve_timeout_attempt(
                            batch_id=batch_id,
                            artifact=artifact,
                            task=task,
                            session_id=issued.session.session_id,
                            exc=_require_timeout_exc(decision),
                            successful_baseline_tps=current_successful_baseline_tps,
                            prior_timeout_observations=prior_timeout_observations,
                        )
                    except ValidatorBatchFailedError as exc:
                        terminal_outcome = AttemptControlKind.VALIDATOR_BATCH_FAILURE.value
                        error_code = str(exc.error_code)
                        raise
                    if timeout_resolution.kind is AttemptControlKind.SUBMISSION:
                        terminal_outcome = AttemptControlKind.SUBMISSION.value
                        error_code = _submission_error_code_or_none(
                            _require_submission(timeout_resolution)
                        )
                        return timeout_resolution
                    if timeout_resolution.kind is AttemptControlKind.TIMEOUT_UNRESOLVED:
                        terminal_outcome = AttemptControlKind.TIMEOUT_UNRESOLVED.value
                        return timeout_resolution
                    if timeout_resolution.kind is AttemptControlKind.VALIDATOR_BATCH_FAILURE:
                        validator_failure = _require_validator_failure(timeout_resolution)
                        terminal_outcome = AttemptControlKind.VALIDATOR_BATCH_FAILURE.value
                        error_code = str(validator_failure.error_code)
                        raise validator_failure
                    raise RuntimeError("timeout review returned unexpected decision")

                if decision.kind is AttemptControlKind.RETRY:
                    self._log_retry_attempt(
                        batch_id=batch_id,
                        artifact=artifact,
                        task=task,
                        attempt_number=attempt_number,
                        exc=_require_retry_exc(decision),
                    )
                    continue

                if decision.kind is AttemptControlKind.VALIDATOR_BATCH_FAILURE:
                    validator_failure = _require_validator_failure(decision)
                    terminal_outcome = AttemptControlKind.VALIDATOR_BATCH_FAILURE.value
                    error_code = str(validator_failure.error_code)
                    raise validator_failure

                raise RuntimeError("task retry loop returned unexpected decision")
        finally:
            _log_session_finished(
                batch_id=batch_id,
                session_id=issued.session.session_id,
                artifact_id=artifact.artifact_id,
                task_id=task.task_id,
                uid=artifact.uid,
                attempt_count=attempt_count,
                session_ms=_monotonic_elapsed_ms(
                    started_at=session_started_at,
                    completed_at=time.monotonic(),
                ),
                terminal_outcome=terminal_outcome,
                error_code=error_code,
            )
            self._clear_task_session(issued.session.session_id)
            self._sessions.revoke(issued.session.session_id)

        raise RuntimeError("task retry loop exited without returning")

    async def _evaluate_task_attempt(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        issued: SessionIssued,
        orchestrator: TaskRunOrchestrator,
        final_attempt: bool,
    ) -> TaskAttemptDecision:
        request = MinerTaskRunRequest(
            batch_id=batch_id,
            session_id=issued.session.session_id,
            token=issued.token,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task=task,
        )
        try:
            outcome = await orchestrator.evaluate(request)
        except SessionBudgetExhaustedError as exc:
            return _submission_decision(
                self._record_exhausted(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=issued.session.session_id,
                    error_message=str(exc),
                )
            )
        except SandboxInvocationError as exc:
            if is_timeout_sandbox_invocation(
                status_code=exc.status_code,
                detail_exception=exc.detail_exception,
            ):
                return _review_timeout_decision(exc)
            provider_failures = self._consume_provider_failures(issued.session.session_id)
            return self._non_timeout_failure_decision(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
                exc=exc,
                provider_failures=provider_failures,
                final_attempt=final_attempt,
            )
        except httpx.TimeoutException as exc:
            if not final_attempt:
                return _retry_decision(exc)
            return _validator_batch_failure_decision(
                ValidatorBatchFailedError(
                    error_code=MinerTaskErrorCode.VALIDATOR_INTERNAL_TIMEOUT,
                    message=str(exc) or type(exc).__name__,
                    failure_detail=ValidatorBatchFailureDetail(
                        error_code=MinerTaskErrorCode.VALIDATOR_INTERNAL_TIMEOUT,
                        error_message=str(exc) or type(exc).__name__,
                        occurred_at=self._clock(),
                        artifact_id=artifact.artifact_id,
                        task_id=task.task_id,
                        uid=artifact.uid,
                        exception_type=_exception_type_name(exc),
                    ),
                )
            )
        except Exception as exc:
            provider_failures = self._consume_provider_failures(issued.session.session_id)
            return self._non_timeout_failure_decision(
                batch_id=batch_id,
                artifact=artifact,
                task=task,
                session_id=issued.session.session_id,
                exc=exc,
                provider_failures=provider_failures,
                final_attempt=final_attempt,
            )

        provider_failures = self._consume_provider_failures(issued.session.session_id)
        provider_failure_decision = self._provider_batch_failure_decision(
            artifact=artifact,
            task=task,
            provider_failures=provider_failures,
            exception_type=None,
        )
        if provider_failure_decision is not None:
            return provider_failure_decision
        return _submission_decision(
            self._record_success(
                batch_id=batch_id,
                session_id=issued.session.session_id,
                outcome=outcome,
            ),
            successful_baseline_tps=slowest_successful_llm_tps(outcome.tool_receipts),
        )

    async def record_failure_for_artifact(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        error_code: MinerTaskErrorCode,
        error_message: str,
    ) -> list[MinerTaskRunSubmission]:
        async def create_submission(task: MinerTask, issued: SessionIssued) -> MinerTaskRunSubmission:
            return self._record_failure(
                batch_id=batch_id,
                session_id=issued.session.session_id,
                uid=artifact.uid,
                artifact_id=artifact.artifact_id,
                task=task,
                error_code=error_code,
                error_message=error_message,
            )

        return await self._run_tasks_with_sessions(
            batch_id=batch_id,
            artifact=artifact,
            tasks=tasks,
            create_submission=create_submission,
        )

    def _record_success(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
        outcome: TaskRunOutcome,
    ) -> MinerTaskRunSubmission:
        breakdown = outcome.run.details.score_breakdown
        if breakdown is None:
            raise RuntimeError("successful task runs require score breakdown details")
        envelope = self._sessions.mark_status(session_id, SessionStatus.COMPLETED)
        submission = MinerTaskRunSubmission(
            batch_id=batch_id,
            validator_uid=self._validator_uid_value(),
            run=outcome.run,
            score=breakdown.total_score,
            execution_log=outcome.tool_receipts,
            usage=outcome.usage,
            session=envelope.session,
        )
        self._record_submission(submission)
        return submission

    def _record_failure(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
        uid: int,
        artifact_id: UUID,
        task: MinerTask,
        error_code: MinerTaskErrorCode,
        error_message: str,
    ) -> MinerTaskRunSubmission:
        return self._record_failed_submission(
            batch_id=batch_id,
            session_id=session_id,
            uid=uid,
            artifact_id=artifact_id,
            task=task,
            error_code=error_code,
            error_message=error_message,
        )

    def _record_failed_submission(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
        uid: int,
        artifact_id: UUID,
        task: MinerTask,
        error_code: MinerTaskErrorCode,
        error_message: str,
        total_tool_usage: ToolUsageSummary | None = None,
        elapsed_ms: float | None = None,
    ) -> MinerTaskRunSubmission:
        envelope = self._sessions.mark_status(session_id, SessionStatus.ERROR)
        return self._record_terminal_failure(
            batch_id=batch_id,
            envelope=envelope,
            uid=uid,
            artifact_id=artifact_id,
            task=task,
            error_code=error_code,
            error_message=error_message,
            total_tool_usage=total_tool_usage,
            elapsed_ms=elapsed_ms,
        )

    def _record_exhausted(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        session_id: UUID,
        error_message: str,
    ) -> MinerTaskRunSubmission:
        envelope = self._sessions.inspect(session_id)
        if envelope.session.status is not SessionStatus.EXHAUSTED:
            raise RuntimeError("exhausted task runs require exhausted session status")
        return self._record_terminal_failure(
            batch_id=batch_id,
            envelope=envelope,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task=task,
            error_code=MinerTaskErrorCode.SESSION_BUDGET_EXHAUSTED,
            error_message=error_message,
        )

    def _record_terminal_failure(
        self,
        *,
        batch_id: UUID,
        envelope: SessionEnvelope,
        uid: int,
        artifact_id: UUID,
        task: MinerTask,
        error_code: MinerTaskErrorCode,
        error_message: str,
        total_tool_usage: ToolUsageSummary | None = None,
        elapsed_ms: float | None = None,
    ) -> MinerTaskRunSubmission:
        session_id = envelope.session.session_id
        completed_at = self._clock()
        usage, summarized_tool_usage = self._summarize_session(envelope)
        execution_log = tuple(self._receipts.for_session(session_id))
        details = EvaluationDetails(
            error=EvaluationError(code=error_code, message=error_message),
            total_tool_usage=total_tool_usage or summarized_tool_usage,
            elapsed_ms=elapsed_ms or _elapsed_ms(issued_at=envelope.session.issued_at, completed_at=completed_at),
        )
        run = MinerTaskRun(
            session_id=session_id,
            uid=uid,
            artifact_id=artifact_id,
            task_id=task.task_id,
            response=None,
            details=details,
            completed_at=completed_at,
        )
        self._receipts.clear_session(session_id)
        submission = MinerTaskRunSubmission(
            batch_id=batch_id,
            validator_uid=self._validator_uid_value(),
            run=run,
            score=0.0,
            execution_log=execution_log,
            usage=usage,
            session=envelope.session,
        )
        self._record_submission(submission)
        return submission

    def _record_task_failure(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        session_id: UUID,
        error_code: MinerTaskErrorCode,
        error_message: str,
        log_message: str,
        exc: Exception,
    ) -> MinerTaskRunSubmission:
        logger.error(
            log_message,
            extra={
                "batch_id": str(batch_id),
                "uid": artifact.uid,
                "artifact_id": str(artifact.artifact_id),
                "task_id": str(task.task_id),
            },
            exc_info=exc,
        )
        return self._record_failure(
            batch_id=batch_id,
            session_id=session_id,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task=task,
            error_code=error_code,
            error_message=error_message,
        )

    def _resolve_timeout_attempt(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        session_id: UUID,
        exc: SandboxInvocationError,
        successful_baseline_tps: float | None,
        prior_timeout_observations: tuple[TimeoutObservationEvidence, ...],
    ) -> TaskAttemptDecision:
        envelope = self._sessions.inspect(session_id)
        observation = self._extract_timeout_observation_evidence(
            session_id=session_id,
            envelope=envelope,
        )
        timeout_attribution = classify_timeout_attribution(
            observation=observation,
            successful_baseline_tps=successful_baseline_tps,
            prior_timeout_observations=prior_timeout_observations,
        )
        if timeout_attribution is None:
            self._receipts.clear_session(session_id)
            return _timeout_unresolved_decision(observation)

        submission = self._record_failed_submission(
            batch_id=batch_id,
            session_id=session_id,
            uid=artifact.uid,
            artifact_id=artifact.artifact_id,
            task=task,
            error_code=(
                MinerTaskErrorCode.TIMEOUT_MINER_OWNED
                if timeout_attribution is TimeoutAttributionKind.MINER_OWNED
                else MinerTaskErrorCode.TIMEOUT_INCONCLUSIVE
            ),
            error_message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
            total_tool_usage=observation.session_summary,
            elapsed_ms=observation.session_elapsed_ms,
        )
        if timeout_attribution is TimeoutAttributionKind.MINER_OWNED:
            logger.error(
                "miner task timed out with miner-owned attribution",
                extra={
                    "batch_id": str(batch_id),
                    "uid": artifact.uid,
                    "artifact_id": str(artifact.artifact_id),
                    "task_id": str(task.task_id),
                },
                exc_info=exc,
            )
            return _submission_decision(submission)

        return _validator_batch_failure_decision(
            ValidatorBatchFailedError(
                error_code=MinerTaskErrorCode.TIMEOUT_INCONCLUSIVE,
                message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
                failure_detail=ValidatorBatchFailureDetail(
                    error_code=MinerTaskErrorCode.TIMEOUT_INCONCLUSIVE,
                    error_message=TERMINAL_TIMEOUT_ERROR_MESSAGE,
                    occurred_at=self._clock(),
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                    uid=artifact.uid,
                    exception_type=_exception_type_name(exc),
                ),
                completed_submissions=(submission,),
                remaining_tasks=(),
            )
        )

    def _non_timeout_failure_decision(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        session_id: UUID,
        exc: Exception,
        provider_failures: tuple[ProviderFailureEvidence, ...],
        final_attempt: bool,
    ) -> TaskAttemptDecision:
        if _is_provider_caused_terminal_failure(exc):
            provider_failure_decision = self._provider_batch_failure_decision(
                artifact=artifact,
                task=task,
                provider_failures=provider_failures,
                exception_type=_exception_type_name(exc),
            )
            if provider_failure_decision is not None:
                return provider_failure_decision

        if isinstance(exc, LlmRetryExhaustedError):
            return _submission_decision(
                self._record_task_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=session_id,
                    error_code=MinerTaskErrorCode.SCORING_LLM_RETRY_EXHAUSTED,
                    error_message=str(exc),
                    log_message="validator scoring provider retries exhausted",
                    exc=exc,
                )
            )

        if isinstance(exc, MinerResponseValidationError):
            return _submission_decision(
                self._record_task_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=session_id,
                    error_code=MinerTaskErrorCode.MINER_RESPONSE_INVALID,
                    error_message=str(exc),
                    log_message="miner returned invalid response payload",
                    exc=exc,
                )
            )

        if isinstance(exc, SandboxInvocationError) and is_script_validation_sandbox_invocation(
            detail_code=exc.detail_code,
        ):
            return _submission_decision(
                self._record_task_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=session_id,
                    error_code=MinerTaskErrorCode.SCRIPT_VALIDATION_FAILED,
                    error_message=exc.detail_error or str(exc),
                    log_message="miner script failed validation during sandbox preload",
                    exc=exc,
                )
            )

        if isinstance(exc, SandboxInvocationError) and exc.detail_code == SANDBOX_DETAIL_CODE_UNHANDLED_EXCEPTION:
            return _submission_decision(
                self._record_task_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=session_id,
                    error_code=MinerTaskErrorCode.MINER_UNHANDLED_EXCEPTION,
                    error_message=exc.detail_error or str(exc),
                    log_message="miner entrypoint raised unhandled exception",
                    exc=exc,
                )
            )

        if isinstance(exc, SandboxInvocationError):
            if not final_attempt:
                return _retry_decision(exc)
            return _submission_decision(
                self._record_task_failure(
                    batch_id=batch_id,
                    artifact=artifact,
                    task=task,
                    session_id=session_id,
                    error_code=MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED,
                    error_message=str(exc),
                    log_message="sandbox invocation failed during miner task run",
                    exc=exc,
                )
            )

        return _validator_batch_failure_decision(
            ValidatorBatchFailedError(
                error_code=MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE,
                message=str(exc),
                failure_detail=ValidatorBatchFailureDetail(
                    error_code=MinerTaskErrorCode.UNEXPECTED_VALIDATOR_FAILURE,
                    error_message=str(exc),
                    occurred_at=self._clock(),
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                    uid=artifact.uid,
                    exception_type=_exception_type_name(exc),
                ),
            )
        )

    def _provider_batch_failure_decision(
        self,
        *,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        provider_failures: tuple[ProviderFailureEvidence, ...],
        exception_type: str | None,
    ) -> TaskAttemptDecision | None:
        provider_batch_evidence = provider_batch_failure_evidence(provider_failures)
        if provider_batch_evidence is None:
            return None
        message = provider_batch_failure_message(provider_batch_evidence)
        return _validator_batch_failure_decision(
            ValidatorBatchFailedError(
                error_code=MinerTaskErrorCode.PROVIDER_BATCH_FAILURE,
                message=message,
                failure_detail=ValidatorBatchFailureDetail(
                    error_code=MinerTaskErrorCode.PROVIDER_BATCH_FAILURE,
                    error_message=message,
                    occurred_at=self._clock(),
                    artifact_id=artifact.artifact_id,
                    task_id=task.task_id,
                    uid=artifact.uid,
                    exception_type=exception_type,
                ),
            )
        )

    def _log_retry_attempt(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        task: MinerTask,
        attempt_number: int,
        exc: Exception,
    ) -> None:
        logger.warning(
            "miner task run attempt failed; retrying once",
            extra={
                "batch_id": str(batch_id),
                "uid": artifact.uid,
                "artifact_id": str(artifact.artifact_id),
                "task_id": str(task.task_id),
                "attempt_number": attempt_number,
            },
            exc_info=exc,
        )

    def _summarize_session(self, envelope: SessionEnvelope) -> tuple[TokenUsageSummary, ToolUsageSummary]:
        receipts = tuple(self._receipts.for_session(envelope.session.session_id))
        return self._usage.summarize(envelope.session, receipts)

    def _extract_timeout_observation_evidence(
        self,
        *,
        session_id: UUID,
        envelope: SessionEnvelope,
    ) -> TimeoutObservationEvidence:
        _, session_summary = self._summarize_session(envelope)
        receipts = tuple(self._receipts.for_session(session_id))
        return TimeoutObservationEvidence(
            successful_llm_samples=successful_llm_samples(receipts),
            session_summary=session_summary,
            session_elapsed_ms=_elapsed_ms(
                issued_at=envelope.session.issued_at,
                completed_at=self._clock(),
            ),
        )

    def _record_submission(self, submission: MinerTaskRunSubmission) -> None:
        self._evaluation_records.record(submission)
        if self._progress is not None:
            self._progress.record(submission)

    def _consume_provider_failures(self, session_id: UUID) -> tuple[ProviderFailureEvidence, ...]:
        if self._progress is None:
            return ()
        return self._progress.consume_provider_failures(session_id)

    def _clear_task_session(self, session_id: UUID) -> None:
        if self._progress is None:
            return
        self._progress.clear_task_session(session_id)

    def _validator_uid_value(self) -> int:
        if self._validator_uid is None:
            info = self._subtensor.validator_info()
            self._validator_uid = int(info.uid)
        return self._validator_uid

    def _issue_session(
        self,
        *,
        batch_id: UUID,
        uid: int,
        task: MinerTask,
    ) -> SessionIssued:
        issued_at = self._clock()
        expires_at = issued_at + self._config.session_ttl
        token = secrets.token_urlsafe(self._config.token_secret_bytes)
        request = SessionTokenRequest(
            session_id=uuid4(),
            uid=uid,
            task_id=task.task_id,
            issued_at=issued_at,
            expires_at=expires_at,
            budget_usd=task.budget_usd,
            token=token,
        )
        issued = self._sessions.issue(request)
        if self._progress is not None:
            self._progress.register_task_session(
                batch_id=batch_id,
                session_id=issued.session.session_id,
            )
        return issued


def _is_provider_caused_terminal_failure(exc: Exception) -> bool:
    if not isinstance(exc, SandboxInvocationError):
        return False
    return is_provider_caused_terminal_failure(
        detail_code=exc.detail_code,
        detail_exception=exc.detail_exception,
        detail_error=exc.detail_error,
    )


def _submission_decision(
    submission: MinerTaskRunSubmission,
    *,
    successful_baseline_tps: float | None = None,
) -> TaskAttemptDecision:
    return TaskAttemptDecision(
        kind=AttemptControlKind.SUBMISSION,
        submission=submission,
        successful_baseline_tps=successful_baseline_tps,
    )


def _retry_decision(exc: Exception) -> TaskAttemptDecision:
    return TaskAttemptDecision(
        kind=AttemptControlKind.RETRY,
        retry_exc=exc,
    )


def _review_timeout_decision(exc: SandboxInvocationError) -> TaskAttemptDecision:
    return TaskAttemptDecision(
        kind=AttemptControlKind.REVIEW_TIMEOUT,
        timeout_exc=exc,
    )


def _timeout_unresolved_decision(observation: TimeoutObservationEvidence) -> TaskAttemptDecision:
    return TaskAttemptDecision(
        kind=AttemptControlKind.TIMEOUT_UNRESOLVED,
        timeout_observation=observation,
    )


def _validator_batch_failure_decision(
    validator_failure: ValidatorBatchFailedError,
) -> TaskAttemptDecision:
    return TaskAttemptDecision(
        kind=AttemptControlKind.VALIDATOR_BATCH_FAILURE,
        validator_failure=validator_failure,
    )


def _require_submission(decision: TaskAttemptDecision) -> MinerTaskRunSubmission:
    if decision.submission is None:
        raise RuntimeError("attempt decision requires submission")
    return decision.submission


def _require_timeout_observation(decision: TaskAttemptDecision) -> TimeoutObservationEvidence:
    if decision.timeout_observation is None:
        raise RuntimeError("attempt decision requires timeout observation")
    return decision.timeout_observation


def _require_timeout_exc(decision: TaskAttemptDecision) -> SandboxInvocationError:
    if decision.timeout_exc is None:
        raise RuntimeError("attempt decision requires timeout exception")
    return decision.timeout_exc


def _require_retry_exc(decision: TaskAttemptDecision) -> Exception:
    if decision.retry_exc is None:
        raise RuntimeError("attempt decision requires retry exception")
    return decision.retry_exc


def _require_validator_failure(decision: TaskAttemptDecision) -> ValidatorBatchFailedError:
    if decision.validator_failure is None:
        raise RuntimeError("attempt decision requires validator batch failure")
    return decision.validator_failure


def _require_single_completed_submission(
    validator_failure: ValidatorBatchFailedError,
) -> MinerTaskRunSubmission:
    completed_submissions = validator_failure.completed_submissions
    if completed_submissions is None or len(completed_submissions) != 1:
        raise RuntimeError("validator batch failure must provide exactly one completed submission")
    return completed_submissions[0]


def _exception_type_name(exc: Exception | None) -> str | None:
    if exc is None:
        return None
    return type(exc).__name__


def _merge_slowest_successful_tps(
    current: float | None,
    candidate: float | None,
) -> float | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return min(current, candidate)


def _submission_error_code(submission: MinerTaskRunSubmission) -> MinerTaskErrorCode:
    error = submission.run.details.error
    if error is None:
        raise RuntimeError("artifact failure submission requires error code")
    return error.code


def _submission_error_message(submission: MinerTaskRunSubmission) -> str:
    error = submission.run.details.error
    if error is None:
        raise RuntimeError("artifact failure submission requires error message")
    return error.message


def _submission_error_code_or_none(submission: MinerTaskRunSubmission) -> MinerTaskErrorCode | None:
    error = submission.run.details.error
    if error is None:
        return None
    return error.code


def _validator_batch_failed_from_existing_submission(
    *,
    submission: MinerTaskRunSubmission,
    artifact: ScriptArtifactSpec,
    task: MinerTask,
    occurred_at: datetime,
) -> ValidatorBatchFailedError:
    error_code = _submission_error_code(submission)
    return ValidatorBatchFailedError(
        error_code=error_code,
        message=_submission_error_message(submission),
        failure_detail=ValidatorBatchFailureDetail(
            error_code=error_code,
            error_message=_submission_error_message(submission),
            occurred_at=occurred_at,
            artifact_id=artifact.artifact_id,
            task_id=task.task_id,
            uid=artifact.uid,
            exception_type=(
                "SandboxInvocationError"
                if error_code == MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED
                else None
            ),
        ),
        completed_submissions=(submission,),
        remaining_tasks=(),
    )


__all__ = [
    "ArtifactExecutionFailedError",
    "ArtifactEvaluationOutcome",
    "ArtifactFailure",
    "EvaluationRunner",
    "LOCAL_RETRY_ATTEMPTS",
    "TERMINAL_TIMEOUT_ERROR_MESSAGE",
    "TaskAttemptDecision",
    "TIMEOUT_REVIEW_MAX_OBSERVATIONS",
    "TIMEOUT_TPS_SLOWDOWN_FACTOR",
    "TimeoutObservationEvidence",
    "ValidatorBatchFailureDetail",
    "ValidatorBatchFailedError",
]
