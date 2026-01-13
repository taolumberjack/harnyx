"""In-memory tracker for per-batch evaluation progress."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict
from uuid import UUID

from caster_validator.application.dto.evaluation import MinerTaskResult


class RunProgressSnapshot(TypedDict):
    batch_id: UUID
    total: int
    completed: int
    remaining: int
    miner_task_results: tuple[MinerTaskResult, ...]


@dataclass(slots=True)
class InMemoryRunProgress:
    expected_by_batch: dict[UUID, int] = field(default_factory=dict)
    results_by_batch: dict[UUID, list[MinerTaskResult]] = field(default_factory=dict)

    def register(self, batch_id: UUID, *, candidate_count: int, claims_count: int) -> None:
        total = candidate_count * claims_count
        self.expected_by_batch[batch_id] = total

    def record(self, result: MinerTaskResult) -> None:
        bucket = self.results_by_batch.setdefault(result.batch_id, [])
        bucket.append(result)

    def snapshot(self, batch_id: UUID) -> RunProgressSnapshot:
        results = tuple(self.results_by_batch.get(batch_id, ()))
        total = int(self.expected_by_batch.get(batch_id, 0))
        completed = len(results)
        remaining = max(0, total - completed)
        return {
            "batch_id": batch_id,
            "total": total,
            "completed": completed,
            "remaining": remaining,
            "miner_task_results": results if total > 0 and completed >= total else (),
        }


__all__ = ["InMemoryRunProgress", "RunProgressSnapshot"]
