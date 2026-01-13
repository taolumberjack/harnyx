"""DTOs related to weight updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from caster_validator.application.dto.evaluation import ScoredEvaluation, TokenUsageSummary


@dataclass(frozen=True)
class WeightUpdateRequest:
    """Payload describing a batch of scored miner criterion evaluations."""

    run_id: UUID
    evaluations: Sequence[ScoredEvaluation]


@dataclass(frozen=True)
class WeightUpdateResult:
    """Normalized weight vector derived from scored miner criterion evaluations."""

    run_id: UUID
    weights: Mapping[int, float]
    usage_by_uid: Mapping[int, TokenUsageSummary]
    ordering: tuple[int, ...]
