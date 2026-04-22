"""Batch scheduler orchestrating miner task runs across artifacts."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import TypeVar
from uuid import UUID

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import (
    MinerTask,
    MinerTaskErrorCode,
    is_delivery_disqualifying_validator_pair_error,
)
from harnyx_commons.sandbox.client import SandboxClient
from harnyx_commons.sandbox.manager import SandboxDeployment, SandboxManager
from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_validator.application.dto.evaluation import (
    MinerTaskBatchRunResult,
    MinerTaskRunSubmission,
    ScriptArtifactSpec,
)
from harnyx_validator.application.evaluate_task_run import TaskRunOrchestrator
from harnyx_validator.application.ports.evaluation_record import EvaluationRecordPort
from harnyx_validator.application.ports.progress import ProgressRecorder
from harnyx_validator.application.ports.subtensor import SubtensorClientPort
from harnyx_validator.application.services.evaluation_runner import (
    LOCAL_RETRY_ATTEMPTS,
    ArtifactEvaluationOutcome,
    ArtifactExecutionFailedError,
    EvaluationRunner,
    TimeoutObservationEvidence,
    UnexpectedArtifactExecutionError,
    ValidatorBatchFailedError,
    ValidatorBatchFailureDetail,
)
from harnyx_validator.runtime.agent_artifact import ArtifactPreparationError

SandboxOptionsFactory = Callable[[ScriptArtifactSpec], SandboxOptions]
TaskRunOrchestratorFactory = Callable[[SandboxClient], TaskRunOrchestrator]
Clock = Callable[[], datetime]
_T = TypeVar("_T")

logger = logging.getLogger("harnyx_validator.scheduler")
measurement_logger = logging.getLogger("harnyx_validator.measurement")


@dataclass(frozen=True, slots=True)
class TimeoutRetryState:
    prior_observations: tuple[TimeoutObservationEvidence, ...] = ()


@dataclass(frozen=True)
class SchedulerConfig:
    """Static configuration used for session issuance."""

    token_secret_bytes: int
    session_ttl: timedelta
    artifact_parallelism: int = 2
    artifact_task_parallelism: int = 5


@dataclass(frozen=True, slots=True)
class _CompletedArtifactResult:
    artifact_id: UUID
    submissions: tuple[MinerTaskRunSubmission, ...]
    slowest_successful_tps: float | None
    timeout_retry_state_by_pair: dict[tuple[UUID, UUID], TimeoutRetryState]
    validator_batch_failure: ValidatorBatchFailedError | None = None


@dataclass(slots=True)
class _BatchArtifactDispatchState:
    submissions_by_artifact_index: list[tuple[MinerTaskRunSubmission, ...] | None]
    slowest_successful_tps: float | None = None
    timeout_retry_state_by_pair: dict[tuple[UUID, UUID], TimeoutRetryState] = field(default_factory=dict)
    stop_dequeuing: bool = False
    published_batch_failure: ValidatorBatchFailedError | None = None
    validator_batch_failures_by_artifact_index: dict[int, ValidatorBatchFailedError] = field(default_factory=dict)
    merge_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _monotonic_elapsed_ms(*, started_at: float, completed_at: float) -> float:
    return round((completed_at - started_at) * 1000.0, 3)


def _count_submission_outcomes(
    submissions: Sequence[MinerTaskRunSubmission],
) -> tuple[int, int]:
    success_count = 0
    failure_count = 0
    for submission in submissions:
        if submission.run.details.error is None:
            success_count += 1
            continue
        failure_count += 1
    return success_count, failure_count


def _merge_slowest_successful_tps(
    current: float | None,
    candidate: float | None,
) -> float | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return min(current, candidate)


def _has_primary_artifact_outcome(
    *,
    outcome: str,
    primary_failure_raised: bool,
) -> bool:
    _ = outcome
    return primary_failure_raised


def _log_batch_execution_started(
    *,
    batch_id: UUID,
    artifact_count: int,
    task_count: int,
    artifact_parallelism: int,
    artifact_task_parallelism: int,
    recorded_pair_count: int,
) -> None:
    measurement_logger.info(
        "miner-task batch execution started",
        extra={
            "data": {
                "batch_id": str(batch_id),
                "artifact_count": artifact_count,
                "task_count": task_count,
                "artifact_parallelism": artifact_parallelism,
                "artifact_task_parallelism": artifact_task_parallelism,
                "recorded_pair_count": recorded_pair_count,
            }
        },
    )


def _log_artifact_execution_finished(
    *,
    batch_id: UUID,
    artifact: ScriptArtifactSpec,
    artifact_index: int,
    artifact_count: int,
    planned_task_count: int,
    success_count: int,
    failure_count: int,
    unresolved_count: int,
    setup_ms: float,
    evaluation_ms: float,
    teardown_ms: float,
    total_ms: float,
    outcome: str,
    error_code: str | None,
) -> None:
    measurement_logger.info(
        "miner-task artifact execution finished",
        extra={
            "data": {
                "batch_id": str(batch_id),
                "artifact_id": str(artifact.artifact_id),
                "uid": artifact.uid,
                "artifact_index": artifact_index,
                "artifact_count": artifact_count,
                "planned_task_count": planned_task_count,
                "success_count": success_count,
                "failure_count": failure_count,
                "unresolved_count": unresolved_count,
                "setup_ms": setup_ms,
                "evaluation_ms": evaluation_ms,
                "teardown_ms": teardown_ms,
                "total_ms": total_ms,
                "outcome": outcome,
                "error_code": error_code,
            }
        },
    )


class EvaluationScheduler:
    """Coordinates issuing sessions and running tasks across artifacts."""

    def __init__(
        self,
        *,
        tasks: Sequence[MinerTask],
        subtensor_client: SubtensorClientPort,
        sandbox_manager: SandboxManager,
        session_manager: SessionManager,
        evaluation_records: EvaluationRecordPort,
        receipt_log: ReceiptLogPort,
        blocking_executor: Executor,
        orchestrator_factory: TaskRunOrchestratorFactory,
        sandbox_options_factory: SandboxOptionsFactory,
        clock: Clock,
        config: SchedulerConfig,
        progress: ProgressRecorder | None = None,
    ) -> None:
        self._tasks = tuple(tasks)
        self._sandboxes = sandbox_manager
        self._make_orchestrator = orchestrator_factory
        self._sandbox_options = sandbox_options_factory
        self._progress = progress
        self._clock = clock
        self._blocking_executor = blocking_executor
        self._config = config
        self._runner = EvaluationRunner(
            subtensor_client=subtensor_client,
            session_manager=session_manager,
            evaluation_records=evaluation_records,
            receipt_log=receipt_log,
            config=config,
            clock=clock,
            progress=progress,
        )

    async def run(
        self,
        *,
        batch_id: UUID,
        requested_artifacts: Sequence[ScriptArtifactSpec],
    ) -> MinerTaskBatchRunResult:
        tasks = self._tasks
        if not tasks:
            raise ValueError("scheduler requires at least one task")

        artifacts = tuple(requested_artifacts)
        if not artifacts:
            raise ValueError("scheduler requires at least one artifact")

        return await self._run_artifacts(
            batch_id=batch_id,
            tasks=tasks,
            artifacts=artifacts,
            blocking_executor=self._blocking_executor,
        )

    async def _run_artifacts(
        self,
        *,
        batch_id: UUID,
        tasks: tuple[MinerTask, ...],
        artifacts: tuple[ScriptArtifactSpec, ...],
        blocking_executor: Executor,
    ) -> MinerTaskBatchRunResult:
        recorded_pairs = self._progress.recorded_pairs(batch_id) if self._progress is not None else frozenset()
        artifact_parallelism = min(max(1, self._config.artifact_parallelism), len(artifacts))
        _log_batch_execution_started(
            batch_id=batch_id,
            artifact_count=len(artifacts),
            task_count=len(tasks),
            artifact_parallelism=artifact_parallelism,
            artifact_task_parallelism=self._config.artifact_task_parallelism,
            recorded_pair_count=len(recorded_pairs),
        )
        dispatch = _BatchArtifactDispatchState(
            submissions_by_artifact_index=[None] * len(artifacts),
        )
        pending_artifacts: asyncio.Queue[tuple[int, ScriptArtifactSpec]] = asyncio.Queue()
        for artifact_index, artifact in enumerate(artifacts, start=1):
            pending_artifacts.put_nowait((artifact_index, artifact))

        workers = [
            asyncio.create_task(
                self._run_artifact_worker(
                    batch_id=batch_id,
                    tasks=tasks,
                    artifact_count=len(artifacts),
                    recorded_pairs=recorded_pairs,
                    pending_artifacts=pending_artifacts,
                    dispatch=dispatch,
                    blocking_executor=blocking_executor,
                )
            )
            for _ in range(artifact_parallelism)
        ]
        try:
            await asyncio.gather(*workers)
        except BaseException:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise

        if dispatch.published_batch_failure is not None:
            raise self._build_final_batch_failure(
                batch_id=batch_id,
                dispatch=dispatch,
            )

        return MinerTaskBatchRunResult(
            batch_id=batch_id,
            tasks=tasks,
            runs=self._flatten_submissions_in_requested_artifact_order(dispatch.submissions_by_artifact_index),
        )

    async def _run_artifact_worker(
        self,
        *,
        batch_id: UUID,
        tasks: tuple[MinerTask, ...],
        artifact_count: int,
        recorded_pairs: frozenset[tuple[UUID, UUID]],
        pending_artifacts: asyncio.Queue[tuple[int, ScriptArtifactSpec]],
        dispatch: _BatchArtifactDispatchState,
        blocking_executor: Executor,
    ) -> None:
        while True:
            async with dispatch.merge_lock:
                if dispatch.stop_dequeuing or dispatch.published_batch_failure is not None:
                    return
                try:
                    artifact_index, artifact = pending_artifacts.get_nowait()
                except asyncio.QueueEmpty:
                    return
                timeout_retry_state_snapshot = {
                    pair_key: TimeoutRetryState(prior_observations=state.prior_observations)
                    for pair_key, state in dispatch.timeout_retry_state_by_pair.items()
                }

            artifact_result = await self._run_single_artifact(
                batch_id=batch_id,
                artifact_index=artifact_index,
                artifact_count=artifact_count,
                artifact=artifact,
                tasks=tasks,
                recorded_pairs=recorded_pairs,
                blocking_executor=blocking_executor,
                completed_artifact_baseline=lambda: dispatch.slowest_successful_tps,
                timeout_retry_state_snapshot=timeout_retry_state_snapshot,
                stop_dequeuing=lambda: self._stop_artifact_dequeue(dispatch),
            )

            async with dispatch.merge_lock:
                dispatch.submissions_by_artifact_index[artifact_index - 1] = artifact_result.submissions
                dispatch.slowest_successful_tps = _merge_slowest_successful_tps(
                    dispatch.slowest_successful_tps,
                    artifact_result.slowest_successful_tps,
                )
                dispatch.timeout_retry_state_by_pair.update(artifact_result.timeout_retry_state_by_pair)
                if artifact_result.validator_batch_failure is not None:
                    dispatch.validator_batch_failures_by_artifact_index[artifact_index] = (
                        artifact_result.validator_batch_failure
                    )
                    if dispatch.published_batch_failure is None:
                        dispatch.published_batch_failure = artifact_result.validator_batch_failure

    async def _evaluate_artifact_with_timeout_state(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: tuple[MinerTask, ...],
        orchestrator: TaskRunOrchestrator,
        successful_baseline_tps: float | None,
        completed_artifact_baseline: Callable[[], float | None] | None = None,
        timeout_retry_state_by_pair: dict[tuple[UUID, UUID], TimeoutRetryState],
    ) -> ArtifactEvaluationOutcome:
        artifact_result = ArtifactEvaluationOutcome(
            submissions=(),
            unresolved_tasks=tasks,
            timeout_observations_by_pair={
                pair_key: state.prior_observations
                for pair_key, state in timeout_retry_state_by_pair.items()
            },
            slowest_successful_tps=successful_baseline_tps,
        )
        current_timeout_states = dict(timeout_retry_state_by_pair)
        while artifact_result.unresolved_tasks:
            artifact_result = await self._runner.evaluate_artifact_with_state(
                batch_id=batch_id,
                artifact=artifact,
                tasks=artifact_result.unresolved_tasks,
                orchestrator=orchestrator,
                successful_baseline_tps=artifact_result.slowest_successful_tps,
                completed_artifact_baseline=completed_artifact_baseline,
                timeout_observations_by_pair={
                    pair_key: state.prior_observations
                    for pair_key, state in current_timeout_states.items()
                },
                earlier_submissions=artifact_result.submissions,
            )
            current_timeout_states = {
                pair_key: TimeoutRetryState(prior_observations=observations)
                for pair_key, observations in artifact_result.timeout_observations_by_pair.items()
            }
        return ArtifactEvaluationOutcome(
            submissions=artifact_result.submissions,
            unresolved_tasks=(),
            timeout_observations_by_pair={
                pair_key: state.prior_observations
                for pair_key, state in current_timeout_states.items()
            },
            slowest_successful_tps=artifact_result.slowest_successful_tps,
        )

    async def _run_single_artifact(
        self,
        *,
        batch_id: UUID,
        artifact_index: int,
        artifact_count: int,
        artifact: ScriptArtifactSpec,
        tasks: tuple[MinerTask, ...],
        recorded_pairs: frozenset[tuple[UUID, UUID]],
        blocking_executor: Executor,
        completed_artifact_baseline: Callable[[], float | None],
        timeout_retry_state_snapshot: dict[tuple[UUID, UUID], TimeoutRetryState],
        stop_dequeuing: Callable[[], None] | None = None,
    ) -> _CompletedArtifactResult:
        remaining_tasks = tuple(
            task for task in tasks if (artifact.artifact_id, task.task_id) not in recorded_pairs
        )
        if not remaining_tasks:
            return _CompletedArtifactResult(
                artifact_id=artifact.artifact_id,
                submissions=(),
                slowest_successful_tps=None,
                timeout_retry_state_by_pair=timeout_retry_state_snapshot,
            )

        artifact_started_at = time.monotonic()
        setup_ms = 0.0
        evaluation_ms = 0.0
        teardown_ms = 0.0
        artifact_submissions: tuple[MinerTaskRunSubmission, ...] = ()
        unresolved_count = len(remaining_tasks)
        outcome = "completed"
        error_code: str | None = None
        backfill_primary_outcome: tuple[str, str] | None = None
        try:
            setup_started_at = time.monotonic()
            deployment = await self._start_artifact_with_retry(
                batch_id=batch_id,
                artifact=artifact,
                tasks=remaining_tasks,
                blocking_executor=blocking_executor,
            )
            setup_ms = _monotonic_elapsed_ms(
                started_at=setup_started_at,
                completed_at=time.monotonic(),
            )
        except ArtifactExecutionFailedError as exc:
            setup_ms = _monotonic_elapsed_ms(
                started_at=setup_started_at,
                completed_at=time.monotonic(),
            )
            if is_delivery_disqualifying_validator_pair_error(exc.error_code):
                _log_artifact_execution_finished(
                    batch_id=batch_id,
                    artifact=artifact,
                    artifact_index=artifact_index,
                    artifact_count=artifact_count,
                    planned_task_count=len(remaining_tasks),
                    success_count=0,
                    failure_count=0,
                    unresolved_count=len(remaining_tasks),
                    setup_ms=setup_ms,
                    evaluation_ms=0.0,
                    teardown_ms=0.0,
                    total_ms=_monotonic_elapsed_ms(
                        started_at=artifact_started_at,
                        completed_at=time.monotonic(),
                    ),
                    outcome="validator_batch_failure",
                    error_code=str(exc.error_code),
                )
                if stop_dequeuing is not None:
                    stop_dequeuing()
                return _CompletedArtifactResult(
                    artifact_id=artifact.artifact_id,
                    submissions=(),
                    slowest_successful_tps=completed_artifact_baseline(),
                    timeout_retry_state_by_pair=timeout_retry_state_snapshot,
                    validator_batch_failure=self._conclusive_batch_failure_from_artifact_error(
                        artifact=artifact,
                        tasks=tasks,
                        completed_submissions=(),
                        failure=exc,
                        recorded_pairs=recorded_pairs,
                    ),
                )
            try:
                artifact_submissions = tuple(
                    await self._record_artifact_failure(
                        batch_id=batch_id,
                        artifact=artifact,
                        failure=exc,
                    )
                )
                unresolved_count = 0
            except UnexpectedArtifactExecutionError as backfill_exc:
                artifact_submissions = backfill_exc.completed_submissions
                unresolved_count = len(backfill_exc.remaining_tasks)
                success_count, failure_count = _count_submission_outcomes(artifact_submissions)
                _log_artifact_execution_finished(
                    batch_id=batch_id,
                    artifact=artifact,
                    artifact_index=artifact_index,
                    artifact_count=artifact_count,
                    planned_task_count=len(remaining_tasks),
                    success_count=success_count,
                    failure_count=failure_count,
                    unresolved_count=unresolved_count,
                    setup_ms=setup_ms,
                    evaluation_ms=0.0,
                    teardown_ms=0.0,
                    total_ms=_monotonic_elapsed_ms(
                        started_at=artifact_started_at,
                        completed_at=time.monotonic(),
                    ),
                    outcome="setup_failed",
                    error_code=str(exc.error_code),
                )
                raise backfill_exc.cause from backfill_exc
            success_count, failure_count = _count_submission_outcomes(artifact_submissions)
            _log_artifact_execution_finished(
                batch_id=batch_id,
                artifact=artifact,
                artifact_index=artifact_index,
                artifact_count=artifact_count,
                planned_task_count=len(remaining_tasks),
                success_count=success_count,
                failure_count=failure_count,
                unresolved_count=unresolved_count,
                setup_ms=setup_ms,
                evaluation_ms=0.0,
                teardown_ms=0.0,
                total_ms=_monotonic_elapsed_ms(
                    started_at=artifact_started_at,
                    completed_at=time.monotonic(),
                ),
                outcome="setup_failed",
                error_code=str(exc.error_code),
            )
            return _CompletedArtifactResult(
                artifact_id=artifact.artifact_id,
                submissions=artifact_submissions,
                slowest_successful_tps=None,
                timeout_retry_state_by_pair=timeout_retry_state_snapshot,
            )

        artifact_result: ArtifactEvaluationOutcome | None = None
        primary_failure_raised = False
        evaluation_started_at: float | None = None
        try:
            orchestrator = self._make_orchestrator(deployment.client)
            evaluation_started_at = time.monotonic()
            artifact_result = await self._evaluate_artifact_with_timeout_state(
                batch_id=batch_id,
                artifact=artifact,
                tasks=remaining_tasks,
                orchestrator=orchestrator,
                successful_baseline_tps=completed_artifact_baseline(),
                completed_artifact_baseline=completed_artifact_baseline,
                timeout_retry_state_by_pair=timeout_retry_state_snapshot,
            )
            evaluation_ms = _monotonic_elapsed_ms(
                started_at=evaluation_started_at,
                completed_at=time.monotonic(),
            )
            artifact_submissions = tuple(artifact_result.submissions)
            unresolved_count = len(artifact_result.unresolved_tasks)
        except ValidatorBatchFailedError as exc:
            primary_failure_raised = True
            if stop_dequeuing is not None:
                stop_dequeuing()
            if evaluation_started_at is not None:
                evaluation_ms = _monotonic_elapsed_ms(
                    started_at=evaluation_started_at,
                    completed_at=time.monotonic(),
                )
            if exc.completed_submissions is not None:
                artifact_submissions = exc.completed_submissions
            if exc.remaining_tasks is not None:
                unresolved_count = len(exc.remaining_tasks)
            outcome = "validator_batch_failure"
            error_code = str(exc.error_code)
            return _CompletedArtifactResult(
                artifact_id=artifact.artifact_id,
                submissions=artifact_submissions,
                slowest_successful_tps=completed_artifact_baseline(),
                timeout_retry_state_by_pair=timeout_retry_state_snapshot,
                validator_batch_failure=ValidatorBatchFailedError(
                    error_code=exc.error_code,
                    message=str(exc),
                    failure_detail=exc.failure_detail,
                    completed_submissions=artifact_submissions,
                    remaining_tasks=exc.remaining_tasks,
                ),
            )
        except UnexpectedArtifactExecutionError as exc:
            primary_failure_raised = True
            if evaluation_started_at is not None:
                evaluation_ms = _monotonic_elapsed_ms(
                    started_at=evaluation_started_at,
                    completed_at=time.monotonic(),
                )
            artifact_submissions = exc.completed_submissions
            unresolved_count = len(exc.remaining_tasks)
            outcome = "unexpected_failure"
            raise exc.cause from exc
        except Exception:
            primary_failure_raised = True
            if evaluation_started_at is not None:
                evaluation_ms = _monotonic_elapsed_ms(
                    started_at=evaluation_started_at,
                    completed_at=time.monotonic(),
                )
            if backfill_primary_outcome is None:
                outcome = "unexpected_failure"
            else:
                outcome, error_code = backfill_primary_outcome
            raise
        finally:
            teardown_started_at = time.monotonic()
            teardown_exc: Exception | None = None
            try:
                await _run_blocking_call(blocking_executor, self._sandboxes.stop, deployment)
            except Exception as exc:
                teardown_exc = exc
                teardown_ms = _monotonic_elapsed_ms(
                    started_at=teardown_started_at,
                    completed_at=time.monotonic(),
                )
                if not _has_primary_artifact_outcome(
                    outcome=outcome,
                    primary_failure_raised=primary_failure_raised,
                ):
                    outcome = "teardown_failed"
                    error_code = str(MinerTaskErrorCode.SANDBOX_FAILED)
                else:
                    logger.warning(
                        "artifact teardown failed after primary failure",
                        extra={
                            "data": {
                                "batch_id": str(batch_id),
                                "uid": artifact.uid,
                                "artifact_id": str(artifact.artifact_id),
                                "primary_outcome": outcome,
                                "primary_error_code": error_code,
                            }
                        },
                    )
            else:
                teardown_ms = _monotonic_elapsed_ms(
                    started_at=teardown_started_at,
                    completed_at=time.monotonic(),
                )
            success_count, failure_count = _count_submission_outcomes(artifact_submissions)
            _log_artifact_execution_finished(
                batch_id=batch_id,
                artifact=artifact,
                artifact_index=artifact_index,
                artifact_count=artifact_count,
                planned_task_count=len(remaining_tasks),
                success_count=success_count,
                failure_count=failure_count,
                unresolved_count=unresolved_count,
                setup_ms=setup_ms,
                evaluation_ms=evaluation_ms,
                teardown_ms=teardown_ms,
                total_ms=_monotonic_elapsed_ms(
                    started_at=artifact_started_at,
                    completed_at=time.monotonic(),
                ),
                outcome=outcome,
                error_code=error_code,
            )
            if teardown_exc is not None and not _has_primary_artifact_outcome(
                outcome=outcome,
                primary_failure_raised=primary_failure_raised,
            ):
                raise teardown_exc

        if artifact_result is None:
            raise RuntimeError("artifact evaluation completed without result")

        return _CompletedArtifactResult(
            artifact_id=artifact.artifact_id,
            submissions=artifact_submissions,
            slowest_successful_tps=artifact_result.slowest_successful_tps,
            timeout_retry_state_by_pair={
                pair_key: TimeoutRetryState(prior_observations=observations)
                for pair_key, observations in artifact_result.timeout_observations_by_pair.items()
            },
        )

    def _stop_artifact_dequeue(self, dispatch: _BatchArtifactDispatchState) -> None:
        dispatch.stop_dequeuing = True

    async def _start_artifact_with_retry(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        blocking_executor: Executor,
    ) -> SandboxDeployment:
        try:
            options = await _run_blocking_call(blocking_executor, self._sandbox_options, artifact)
        except ArtifactPreparationError as exc:
            logger.error(
                "failed to prepare sandbox options",
                extra={"batch_id": str(batch_id), "uid": artifact.uid, "artifact_id": str(artifact.artifact_id)},
                exc_info=exc,
            )
            raise self._artifact_execution_failure(
                artifact=artifact,
                tasks=tasks,
                error_code=MinerTaskErrorCode(exc.error_code),
                error_message=str(exc),
                exception_type=exc.exception_type,
            ) from exc
        except Exception as exc:
            logger.error(
                "failed to prepare sandbox options",
                extra={"batch_id": str(batch_id), "uid": artifact.uid, "artifact_id": str(artifact.artifact_id)},
                exc_info=exc,
            )
            raise self._artifact_execution_failure(
                artifact=artifact,
                tasks=tasks,
                error_code=MinerTaskErrorCode.ARTIFACT_SETUP_FAILED,
                error_message=str(exc),
                exception_type=type(exc).__name__,
            ) from exc

        last_error_message = ""
        for attempt_number in range(1, LOCAL_RETRY_ATTEMPTS + 1):
            try:
                return await _run_blocking_call(blocking_executor, self._sandboxes.start, options)
            except Exception as exc:
                last_error_message = str(exc)
                if attempt_number < LOCAL_RETRY_ATTEMPTS:
                    self._log_artifact_retry(
                        batch_id=batch_id,
                        artifact=artifact,
                        attempt_number=attempt_number,
                        stage="sandbox start",
                        exc=exc,
                    )
                    continue
                logger.error(
                    "failed to start sandbox",
                    extra={"batch_id": str(batch_id), "uid": artifact.uid, "artifact_id": str(artifact.artifact_id)},
                    exc_info=exc,
                )
                break

        raise self._artifact_execution_failure(
            artifact=artifact,
            tasks=tasks,
            error_code=MinerTaskErrorCode.SANDBOX_START_FAILED,
            error_message=last_error_message or "artifact setup failed",
            exception_type=None,
        )

    def _log_artifact_retry(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        attempt_number: int,
        stage: str,
        exc: Exception,
    ) -> None:
        logger.warning(
            "artifact setup attempt failed; retrying once",
            extra={
                "batch_id": str(batch_id),
                "uid": artifact.uid,
                "artifact_id": str(artifact.artifact_id),
                "attempt_number": attempt_number,
                "stage": stage,
            },
            exc_info=exc,
        )

    async def _record_artifact_failure(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        failure: ArtifactExecutionFailedError,
    ) -> list[MinerTaskRunSubmission]:
        submissions = list(failure.completed_submissions)
        submissions.extend(
            await self._runner.record_failure_for_artifact(
                batch_id=batch_id,
                artifact=artifact,
                tasks=failure.remaining_tasks,
                error_code=failure.error_code,
                error_message=str(failure),
            )
        )
        return submissions

    def _artifact_execution_failure(
        self,
        *,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        error_code: MinerTaskErrorCode,
        error_message: str,
        exception_type: str | None,
    ) -> ArtifactExecutionFailedError:
        return ArtifactExecutionFailedError(
            error_code=error_code,
            message=error_message,
            failure_detail=ValidatorBatchFailureDetail(
                error_code=error_code,
                error_message=error_message,
                occurred_at=self._clock().astimezone(UTC),
                artifact_id=artifact.artifact_id,
                uid=artifact.uid,
                exception_type=exception_type,
            ),
            completed_submissions=(),
            remaining_tasks=tuple(tasks),
        )

    def _remaining_tasks_for_failed_artifact(
        self,
        *,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        recorded_pairs: frozenset[tuple[UUID, UUID]],
    ) -> tuple[MinerTask, ...]:
        return tuple(
            task
            for task in tasks
            if (artifact.artifact_id, task.task_id) not in recorded_pairs
        )

    def _conclusive_batch_failure_from_artifact_error(
        self,
        *,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
        completed_submissions: tuple[MinerTaskRunSubmission, ...],
        failure: ArtifactExecutionFailedError,
        recorded_pairs: frozenset[tuple[UUID, UUID]],
    ) -> ValidatorBatchFailedError:
        return ValidatorBatchFailedError(
            error_code=failure.error_code,
            message=str(failure),
            failure_detail=failure.failure_detail,
            completed_submissions=completed_submissions,
            remaining_tasks=self._remaining_tasks_for_failed_artifact(
                artifact=artifact,
                tasks=tasks,
                recorded_pairs=recorded_pairs,
            ),
        )

    def _build_final_batch_failure(
        self,
        *,
        batch_id: UUID,
        dispatch: _BatchArtifactDispatchState,
    ) -> ValidatorBatchFailedError:
        canonical_validator_failure_index = (
            min(dispatch.validator_batch_failures_by_artifact_index)
            if dispatch.validator_batch_failures_by_artifact_index
            else None
        )
        failure = dispatch.published_batch_failure
        if failure is None:
            raise RuntimeError("published batch failure missing when finalizing batch failure")
        if canonical_validator_failure_index is not None:
            failure = dispatch.validator_batch_failures_by_artifact_index[canonical_validator_failure_index]
        return ValidatorBatchFailedError(
            error_code=failure.error_code,
            message=str(failure),
            failure_detail=failure.failure_detail,
            completed_submissions=self._flatten_submissions_in_requested_artifact_order(
                dispatch.submissions_by_artifact_index
            ),
            remaining_tasks=failure.remaining_tasks,
        )

    def _flatten_submissions_in_requested_artifact_order(
        self,
        submissions_by_artifact_index: list[tuple[MinerTaskRunSubmission, ...] | None],
    ) -> tuple[MinerTaskRunSubmission, ...]:
        return tuple(
            submission
            for artifact_submissions in submissions_by_artifact_index
            if artifact_submissions is not None
            for submission in artifact_submissions
        )


async def _run_blocking_call(
    executor: Executor,
    func: Callable[..., _T],
    /,
    *args: object,
) -> _T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(func, *args))


__all__ = ["EvaluationScheduler", "SchedulerConfig"]
