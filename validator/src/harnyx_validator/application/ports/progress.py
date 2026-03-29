"""Port for recording miner-task batch progress snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypedDict
from uuid import UUID

from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, MinerTaskRunSubmission


class ProviderFailureEvidence(TypedDict):
    provider: str
    model: str
    total_calls: int
    failed_calls: int


class ProgressRecorder(Protocol):
    def register(self, batch: MinerTaskBatchSpec) -> None:
        ...

    def record(self, result: MinerTaskRunSubmission) -> None:
        ...

    def restore_completed_runs(
        self,
        batch: MinerTaskBatchSpec,
        submissions: Sequence[MinerTaskRunSubmission],
        provider_evidence: Sequence[ProviderFailureEvidence] = (),
    ) -> None:
        ...

    def recorded_pairs(self, batch_id: UUID) -> frozenset[tuple[UUID, UUID]]:
        ...

    def register_task_session(
        self,
        *,
        batch_id: UUID,
        session_id: UUID,
    ) -> None:
        ...

    def record_provider_call(
        self,
        *,
        session_id: UUID,
        provider: str,
        model: str,
    ) -> None:
        ...

    def record_provider_failure(
        self,
        *,
        session_id: UUID,
        provider: str,
        model: str,
    ) -> None:
        ...

    def consume_provider_failures(self, session_id: UUID) -> tuple[ProviderFailureEvidence, ...]:
        ...

    def clear_task_session(self, session_id: UUID) -> None:
        ...


__all__ = ["ProgressRecorder", "ProviderFailureEvidence"]
