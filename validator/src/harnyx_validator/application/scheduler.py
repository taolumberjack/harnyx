"""Batch scheduler orchestrating miner task runs across artifacts."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from harnyx_commons.application.ports.receipt_log import ReceiptLogPort
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import MinerTask
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
    ArtifactExecutionFailedError,
    EvaluationRunner,
    ValidatorBatchFailedError,
    ValidatorBatchFailureDetail,
)
from harnyx_validator.runtime.agent_artifact import ArtifactPreparationError

SandboxOptionsFactory = Callable[[ScriptArtifactSpec], SandboxOptions]
TaskRunOrchestratorFactory = Callable[[SandboxClient], TaskRunOrchestrator]
Clock = Callable[[], datetime]

logger = logging.getLogger("harnyx_validator.scheduler")
BATCH_ARTIFACT_BREAKER_THRESHOLD = 3
_ARTIFACT_PREPARATION_BREAKER_ERROR_CODES = frozenset(
    ("artifact_fetch_failed", "artifact_staging_failed", "artifact_setup_failed")
)


@dataclass(frozen=True)
class SchedulerConfig:
    """Static configuration used for session issuance."""

    token_secret_bytes: int
    session_ttl: timedelta


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

        submissions = []
        recorded_pairs = self._progress.recorded_pairs(batch_id) if self._progress is not None else frozenset()
        artifacts_with_breaker: set[UUID] = set()
        for artifact in artifacts:
            remaining_tasks = tuple(
                task
                for task in tasks
                if (artifact.artifact_id, task.task_id) not in recorded_pairs
            )
            if not remaining_tasks:
                continue

            logger.debug(
                "starting miner task run for artifact",
                extra={"uid": artifact.uid, "artifact_id": str(artifact.artifact_id)},
            )
            try:
                deployment = await self._start_artifact_with_retry(
                    batch_id=batch_id,
                    artifact=artifact,
                    tasks=remaining_tasks,
                )
            except ArtifactExecutionFailedError as exc:
                submissions.extend(
                    await self._record_artifact_failure(
                        batch_id=batch_id,
                        artifact=artifact,
                        failure=exc,
                    )
                )
                if exc.artifact_breaker_tripped:
                    self._raise_if_batch_breaker_tripped(
                        batch_id=batch_id,
                        artifact=artifact,
                        failure_detail=exc.failure_detail,
                        artifacts_with_breaker=artifacts_with_breaker,
                    )
                continue

            batch_breaker_failure: ValidatorBatchFailedError | None = None
            try:
                orchestrator = self._make_orchestrator(deployment.client)
                try:
                    submissions.extend(
                        await self._runner.evaluate_artifact(
                            batch_id=batch_id,
                            artifact=artifact,
                            tasks=remaining_tasks,
                            orchestrator=orchestrator,
                        ),
                    )
                except ArtifactExecutionFailedError as exc:
                    submissions.extend(
                        await self._record_artifact_failure(
                            batch_id=batch_id,
                            artifact=artifact,
                            failure=exc,
                        )
                    )
                    if exc.artifact_breaker_tripped:
                        batch_breaker_failure = self._batch_breaker_failure(
                            batch_id=batch_id,
                            artifact=artifact,
                            failure_detail=exc.failure_detail,
                            artifacts_with_breaker=artifacts_with_breaker,
                        )
            finally:
                await asyncio.to_thread(self._sandboxes.stop, deployment)
            if batch_breaker_failure is not None:
                raise batch_breaker_failure

            logger.debug(
                "finished miner task run for artifact",
                extra={"uid": artifact.uid, "artifact_id": str(artifact.artifact_id)},
            )

        return MinerTaskBatchRunResult(
            batch_id=batch_id,
            tasks=tasks,
            runs=tuple(submissions),
        )

    async def _start_artifact_with_retry(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        tasks: Sequence[MinerTask],
    ) -> SandboxDeployment:
        try:
            options = await asyncio.to_thread(self._sandbox_options, artifact)
        except ArtifactPreparationError as exc:
            logger.error(
                "failed to prepare sandbox options",
                extra={"batch_id": str(batch_id), "uid": artifact.uid, "artifact_id": str(artifact.artifact_id)},
                exc_info=exc,
            )
            raise self._artifact_execution_failure(
                artifact=artifact,
                tasks=tasks,
                error_code=exc.error_code,
                error_message=str(exc),
                exception_type=exc.exception_type,
                artifact_breaker_tripped=exc.error_code in _ARTIFACT_PREPARATION_BREAKER_ERROR_CODES,
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
                error_code="artifact_setup_failed",
                error_message=str(exc),
                exception_type=type(exc).__name__,
                artifact_breaker_tripped=True,
            ) from exc

        last_error_message = ""
        for attempt_number in range(1, LOCAL_RETRY_ATTEMPTS + 1):
            try:
                return await asyncio.to_thread(self._sandboxes.start, options)
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
            error_code="sandbox_start_failed",
            error_message=last_error_message or "artifact setup failed",
            exception_type=None,
            artifact_breaker_tripped=True,
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
        error_code: str,
        error_message: str,
        exception_type: str | None,
        artifact_breaker_tripped: bool = False,
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
            artifact_breaker_tripped=artifact_breaker_tripped,
        )

    def _raise_if_batch_breaker_tripped(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        failure_detail: ValidatorBatchFailureDetail,
        artifacts_with_breaker: set[UUID],
    ) -> None:
        failure = self._batch_breaker_failure(
            batch_id=batch_id,
            artifact=artifact,
            failure_detail=failure_detail,
            artifacts_with_breaker=artifacts_with_breaker,
        )
        if failure is not None:
            raise failure

    def _batch_breaker_failure(
        self,
        *,
        batch_id: UUID,
        artifact: ScriptArtifactSpec,
        failure_detail: ValidatorBatchFailureDetail,
        artifacts_with_breaker: set[UUID],
    ) -> ValidatorBatchFailedError | None:
        artifacts_with_breaker.add(artifact.artifact_id)
        logger.warning(
            "artifact breaker tripped",
            extra={
                "batch_id": str(batch_id),
                "uid": artifact.uid,
                "artifact_id": str(artifact.artifact_id),
                "artifacts_with_breaker": len(artifacts_with_breaker),
            },
        )
        if len(artifacts_with_breaker) < BATCH_ARTIFACT_BREAKER_THRESHOLD:
            return None
        return ValidatorBatchFailedError(
            error_code="artifact_breaker_tripped",
            message="validator artifact breaker tripped across 3 artifacts",
            failure_detail=ValidatorBatchFailureDetail(
                error_code="artifact_breaker_tripped",
                error_message="validator artifact breaker tripped across 3 artifacts",
                occurred_at=self._clock().astimezone(UTC),
                artifact_id=failure_detail.artifact_id,
                uid=failure_detail.uid,
                exception_type=failure_detail.exception_type,
            ),
        )


__all__ = ["EvaluationScheduler", "SchedulerConfig"]
