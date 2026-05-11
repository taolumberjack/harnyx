"""In-memory tracker for per-batch miner-task progress."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeAlias, TypedDict
from uuid import UUID

from harnyx_commons.domain.miner_task import MinerTask
from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, MinerTaskRunSubmission
from harnyx_validator.application.ports.progress import ProviderFailureEvidence

ProviderEvidenceSnapshot: TypeAlias = ProviderFailureEvidence


class RunProgressSnapshot(TypedDict):
    batch_id: UUID
    total: int
    completed: int
    remaining: int
    tasks: tuple[MinerTask, ...]
    miner_task_runs: tuple[MinerTaskRunSubmission, ...]
    provider_evidence: tuple[ProviderEvidenceSnapshot, ...]


@dataclass(slots=True)
class _SessionRunContext:
    batch_id: UUID


@dataclass(slots=True)
class _ProviderEvidenceCounter:
    total_calls: int = 0
    failed_calls: int = 0
    failure_reason: str | None = None


@dataclass(slots=True)
class InMemoryRunProgress:
    batches_by_id: dict[UUID, MinerTaskBatchSpec] = field(default_factory=dict)
    expected_by_batch: dict[UUID, int] = field(default_factory=dict)
    tasks_by_batch: dict[UUID, tuple[MinerTask, ...]] = field(default_factory=dict)
    task_positions_by_batch: dict[UUID, dict[UUID, int]] = field(default_factory=dict)
    artifact_positions_by_batch: dict[UUID, dict[UUID, int]] = field(default_factory=dict)
    results_by_batch: dict[
        UUID,
        dict[tuple[UUID, UUID], MinerTaskRunSubmission],
    ] = field(default_factory=dict)
    session_context_by_id: dict[UUID, _SessionRunContext] = field(default_factory=dict)
    provider_counters_by_batch: dict[
        UUID,
        dict[tuple[str, str], _ProviderEvidenceCounter],
    ] = field(default_factory=dict)
    failed_provider_keys_by_session: dict[UUID, set[tuple[str, str]]] = field(default_factory=dict)

    def register(self, batch: MinerTaskBatchSpec) -> None:
        existing = self.batches_by_id.get(batch.batch_id)
        if existing is not None:
            if existing != batch:
                raise RuntimeError("batch_id already exists with different contents")
            return

        self.batches_by_id[batch.batch_id] = batch
        self.expected_by_batch[batch.batch_id] = len(batch.tasks) * len(batch.artifacts)
        self.tasks_by_batch[batch.batch_id] = batch.tasks
        self.task_positions_by_batch[batch.batch_id] = {
            task.task_id: index for index, task in enumerate(batch.tasks)
        }
        self.artifact_positions_by_batch[batch.batch_id] = {
            artifact.artifact_id: index for index, artifact in enumerate(batch.artifacts)
        }

    def record(self, result: MinerTaskRunSubmission) -> None:
        bucket = self.results_by_batch.setdefault(result.batch_id, {})
        self._record_submission(bucket, result)

    def restore_completed_runs(
        self,
        batch: MinerTaskBatchSpec,
        submissions: Sequence[MinerTaskRunSubmission],
        provider_evidence: Sequence[ProviderEvidenceSnapshot] = (),
    ) -> None:
        self.register(batch)
        staged_results = dict(self.results_by_batch.get(batch.batch_id, {}))
        for submission in submissions:
            if submission.batch_id != batch.batch_id:
                raise RuntimeError("restored submission batch_id mismatch")
            self._record_submission(staged_results, submission)
        self.results_by_batch[batch.batch_id] = staged_results
        self.provider_counters_by_batch[batch.batch_id] = self._merged_provider_counters(
            batch.batch_id,
            provider_evidence,
        )

    def _merged_provider_counters(
        self,
        batch_id: UUID,
        provider_evidence: Sequence[ProviderEvidenceSnapshot],
    ) -> dict[tuple[str, str], _ProviderEvidenceCounter]:
        existing = self.provider_counters_by_batch.get(batch_id, {})
        merged = dict(existing)
        for entry in provider_evidence:
            key = _provider_model_key(provider=entry["provider"], model=entry["model"])
            restored = _ProviderEvidenceCounter(
                total_calls=entry["total_calls"],
                failed_calls=entry["failed_calls"],
                failure_reason=entry.get("failure_reason"),
            )
            current = merged.get(key)
            if current is None:
                merged[key] = restored
                continue
            merged[key] = _ProviderEvidenceCounter(
                total_calls=max(current.total_calls, restored.total_calls),
                failed_calls=max(current.failed_calls, restored.failed_calls),
                failure_reason=current.failure_reason or restored.failure_reason,
            )
        return merged

    def recorded_pairs(self, batch_id: UUID) -> frozenset[tuple[UUID, UUID]]:
        bucket = self.results_by_batch.get(batch_id, {})
        return frozenset(bucket)

    def register_task_session(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
    ) -> None:
        self.session_context_by_id[session_id] = _SessionRunContext(batch_id=batch_id)

    def record_provider_call(
        self,
        *,
        session_id: UUID,
        provider: str,
        model: str,
    ) -> None:
        key = _provider_model_key(provider=provider, model=model)
        context = self.session_context_by_id.get(session_id)
        if context is None:
            return
        counter = self.provider_counters_by_batch.setdefault(context.batch_id, {}).setdefault(
            key,
            _ProviderEvidenceCounter(),
        )
        counter.total_calls += 1

    def record_provider_failure(
        self,
        *,
        session_id: UUID,
        provider: str,
        model: str,
        reason: str,
    ) -> None:
        key = _provider_model_key(provider=provider, model=model)
        context = self.session_context_by_id.get(session_id)
        if context is None:
            return
        counter = self.provider_counters_by_batch.setdefault(context.batch_id, {}).setdefault(
            key,
            _ProviderEvidenceCounter(),
        )
        counter.failed_calls += 1
        failure_reason = reason.strip()
        if failure_reason:
            counter.failure_reason = failure_reason
        keys = self.failed_provider_keys_by_session.setdefault(session_id, set())
        keys.add(key)

    def consume_provider_failures(self, session_id: UUID) -> tuple[ProviderEvidenceSnapshot, ...]:
        keys = self.failed_provider_keys_by_session.pop(session_id, None)
        if not keys:
            return ()
        context = self.session_context_by_id.get(session_id)
        if context is None:
            return ()
        snapshots: list[ProviderEvidenceSnapshot] = []
        for key in sorted(keys):
            snapshot = self._provider_evidence_snapshot(batch_id=context.batch_id, key=key)
            if snapshot is None:
                continue
            snapshots.append(snapshot)
        return tuple(snapshots)

    def clear_task_session(self, session_id: UUID) -> None:
        self.session_context_by_id.pop(session_id, None)
        self.failed_provider_keys_by_session.pop(session_id, None)

    def provider_evidence(self, batch_id: UUID) -> tuple[ProviderEvidenceSnapshot, ...]:
        provider_counters = self.provider_counters_by_batch.get(batch_id, {})
        snapshots: list[ProviderEvidenceSnapshot] = []
        for provider, model in sorted(provider_counters):
            snapshot = self._provider_evidence_snapshot(batch_id=batch_id, key=(provider, model))
            if snapshot is None:
                continue
            snapshots.append(snapshot)
        return tuple(snapshots)

    def snapshot(self, batch_id: UUID) -> RunProgressSnapshot:
        results = tuple(
            sorted(
                self.results_by_batch.get(batch_id, {}).values(),
                key=lambda result: self._result_sort_key(batch_id, result),
            )
        )
        total = int(self.expected_by_batch.get(batch_id, 0))
        completed = len(results)
        remaining = max(0, total - completed)
        return {
            "batch_id": batch_id,
            "total": total,
            "completed": completed,
            "remaining": remaining,
            "tasks": self.tasks_by_batch.get(batch_id, ()),
            "miner_task_runs": results,
            "provider_evidence": self.provider_evidence(batch_id),
        }

    def _result_sort_key(
        self,
        batch_id: UUID,
        result: MinerTaskRunSubmission,
    ) -> tuple[int, int, str, str]:
        artifact_positions = self.artifact_positions_by_batch.get(batch_id, {})
        task_positions = self.task_positions_by_batch.get(batch_id, {})
        fallback_artifact = len(artifact_positions)
        fallback_task = len(task_positions)
        return (
            artifact_positions.get(result.run.artifact_id, fallback_artifact),
            task_positions.get(result.run.task_id, fallback_task),
            str(result.run.artifact_id),
            str(result.run.task_id),
        )

    def _provider_evidence_snapshot(
        self,
        *,
        batch_id: UUID,
        key: tuple[str, str],
    ) -> ProviderEvidenceSnapshot | None:
        provider_counters = self.provider_counters_by_batch.get(batch_id, {})
        counter = provider_counters.get(key)
        if counter is None:
            return None
        provider, model = key
        snapshot: ProviderEvidenceSnapshot = {
            "provider": provider,
            "model": model,
            "total_calls": counter.total_calls,
            "failed_calls": counter.failed_calls,
        }
        if counter.failure_reason is not None:
            snapshot["failure_reason"] = counter.failure_reason
        return snapshot

    @staticmethod
    def _record_submission(
        bucket: dict[tuple[UUID, UUID], MinerTaskRunSubmission],
        result: MinerTaskRunSubmission,
    ) -> None:
        pair = (result.run.artifact_id, result.run.task_id)
        existing = bucket.get(pair)
        if existing is not None:
            if existing != result:
                raise RuntimeError(
                    "batch already recorded a different result for artifact/task pair"
                )
            return
        bucket[pair] = result


def _provider_model_key(*, provider: str, model: str) -> tuple[str, str]:
    normalized_provider = provider.strip()
    normalized_model = model.strip()
    if not normalized_provider:
        raise RuntimeError("provider key must not be empty")
    if not normalized_model:
        raise RuntimeError("model key must not be empty")
    return normalized_provider, normalized_model


__all__ = ["InMemoryRunProgress", "ProviderEvidenceSnapshot", "RunProgressSnapshot"]
