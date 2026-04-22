"""Batch scheduler orchestrating miner task runs across artifacts."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
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
    artifact_task_parallelism: int = 5


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
        submissions = []
        recorded_pairs = self._progress.recorded_pairs(batch_id) if self._progress is not None else frozenset()
        successful_baseline_tps: float | None = None
        timeout_retry_state_by_pair: dict[tuple[UUID, UUID], TimeoutRetryState] = {}
        _log_batch_execution_started(
            batch_id=batch_id,
            artifact_count=len(artifacts),
            task_count=len(tasks),
            artifact_task_parallelism=self._config.artifact_task_parallelism,
            recorded_pair_count=len(recorded_pairs),
        )
        for artifact_index, artifact in enumerate(artifacts, start=1):
            remaining_tasks = tuple(
                task
                for task in tasks
                if (artifact.artifact_id, task.task_id) not in recorded_pairs
            )
            if not remaining_tasks:
                continue

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
                        artifact_count=len(artifacts),
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
                    raise self._conclusive_batch_failure_from_artifact_error(
                        artifact=artifact,
                        tasks=tasks,
                        completed_submissions=tuple(submissions),
                        failure=exc,
                        recorded_pairs=recorded_pairs,
                    ) from exc
                try:
                    failed_submissions = tuple(
                        await self._record_artifact_failure(
                            batch_id=batch_id,
                            artifact=artifact,
                            failure=exc,
                        )
                    )
                    setup_unresolved_count = 0
                except UnexpectedArtifactExecutionError as backfill_exc:
                    failed_submissions = backfill_exc.completed_submissions
                    setup_unresolved_count = len(backfill_exc.remaining_tasks)
                    submissions.extend(failed_submissions)
                    success_count, failure_count = _count_submission_outcomes(failed_submissions)
                    _log_artifact_execution_finished(
                        batch_id=batch_id,
                        artifact=artifact,
                        artifact_index=artifact_index,
                        artifact_count=len(artifacts),
                        planned_task_count=len(remaining_tasks),
                        success_count=success_count,
                        failure_count=failure_count,
                        unresolved_count=setup_unresolved_count,
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
                submissions.extend(failed_submissions)
                success_count, failure_count = _count_submission_outcomes(failed_submissions)
                _log_artifact_execution_finished(
                    batch_id=batch_id,
                    artifact=artifact,
                    artifact_index=artifact_index,
                    artifact_count=len(artifacts),
                    planned_task_count=len(remaining_tasks),
                    success_count=success_count,
                    failure_count=failure_count,
                    unresolved_count=setup_unresolved_count,
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
                continue

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
                    successful_baseline_tps=successful_baseline_tps,
                    timeout_retry_state_by_pair=timeout_retry_state_by_pair,
                )
                evaluation_ms = _monotonic_elapsed_ms(
                    started_at=evaluation_started_at,
                    completed_at=time.monotonic(),
                )
                artifact_submissions = tuple(artifact_result.submissions)
                submissions.extend(artifact_submissions)
                successful_baseline_tps = artifact_result.slowest_successful_tps
                timeout_retry_state_by_pair = {
                    pair_key: TimeoutRetryState(prior_observations=observations)
                    for pair_key, observations in artifact_result.timeout_observations_by_pair.items()
                }
                unresolved_count = len(artifact_result.unresolved_tasks)
            except ValidatorBatchFailedError as exc:
                primary_failure_raised = True
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
                raise
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
                    artifact_count=len(artifacts),
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

        return MinerTaskBatchRunResult(
            batch_id=batch_id,
            tasks=tasks,
            runs=tuple(submissions),
        )

    async def _evaluate_artifact_with_timeout_state(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: tuple[MinerTask, ...],
        orchestrator: TaskRunOrchestrator,
        successful_baseline_tps: float | None,
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


async def _run_blocking_call(
    executor: Executor,
    func: Callable[..., _T],
    /,
    *args: object,
) -> _T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(func, *args))


__all__ = ["EvaluationScheduler", "SchedulerConfig"]
