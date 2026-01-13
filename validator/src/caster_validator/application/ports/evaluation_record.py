"""Port describing durable evaluation record persistence."""

from __future__ import annotations

from typing import Protocol

from caster_validator.application.dto.evaluation import MinerTaskResult


class EvaluationRecordPort(Protocol):
    """Persists miner-task results to an external store."""

    def record(self, result: MinerTaskResult) -> None:
        """Persist the supplied miner-task result payload."""


__all__ = ["EvaluationRecordPort"]
