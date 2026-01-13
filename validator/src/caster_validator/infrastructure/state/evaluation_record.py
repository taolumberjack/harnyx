"""In-memory evaluation record store."""

from __future__ import annotations

from threading import Lock

from caster_validator.application.dto.evaluation import MinerTaskResult
from caster_validator.application.ports.evaluation_record import EvaluationRecordPort


class InMemoryEvaluationRecordStore(EvaluationRecordPort):
    """Stores miner-task results in memory."""

    def __init__(self) -> None:
        self._records: list[MinerTaskResult] = []
        self._lock = Lock()

    def record(self, result: MinerTaskResult) -> None:
        with self._lock:
            self._records.append(result)

    def records(self) -> tuple[MinerTaskResult, ...]:
        with self._lock:
            return tuple(self._records)


__all__ = ["InMemoryEvaluationRecordStore"]
