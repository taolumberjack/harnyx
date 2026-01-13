"""Use case for accepting platform-supplied batches into the inbox."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from caster_validator.application.dto.evaluation import MinerTaskBatchSpec
from caster_validator.application.status import StatusProvider
from caster_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox


class ProgressTracker(Protocol):
    def register(self, batch_id: UUID, *, candidate_count: int, claims_count: int) -> None:
        ...


@dataclass(slots=True)
class AcceptEvaluationBatch:
    """Store the provided batch for later execution."""

    inbox: InMemoryBatchInbox
    status: StatusProvider | None = None
    progress: ProgressTracker | None = None

    def execute(self, batch: MinerTaskBatchSpec) -> None:
        self.inbox.put(batch)
        if self.status is not None:
            self.status.state.queued_batches = len(self.inbox)
        if self.progress is not None:
            claims_count = len(batch.claims)
            self.progress.register(batch.batch_id, candidate_count=len(batch.candidates), claims_count=claims_count)

__all__ = ["AcceptEvaluationBatch"]
