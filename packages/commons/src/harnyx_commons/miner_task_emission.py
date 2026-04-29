"""Miner-task champion emission policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from uuid import UUID

OWNER_UID = 0
MAX_MINER_EMISSION_FRACTION = 0.10
EMISSION_BENCHMARK_STATES = frozenset({"completed", "partial_success"})


@dataclass(frozen=True, slots=True)
class BenchmarkScoredChampionSelection:
    champion_uid: int
    weights: dict[int, float]
    benchmark_score: float
    champion_artifact_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class ChampionWeightEvent:
    batch_id: UUID
    computed_at: datetime
    sequence_id: int
    champion_uid: int | None
    weights: dict[int, float]
    champion_artifact_id: UUID | None


@dataclass(frozen=True, slots=True)
class BenchmarkEmissionCandidate:
    source_batch_id: UUID
    run_id: UUID
    created_at: datetime
    state: str
    champion_uid: int
    champion_artifact_id: UUID
    mean_total_score: float | None


def compose_champion_weights(champion_uid: int | None) -> dict[int, float]:
    if champion_uid is None:
        return {}
    return {champion_uid: 1.0}


def benchmark_state_is_emission_eligible(state: str) -> bool:
    return state in EMISSION_BENCHMARK_STATES


def select_benchmark_scored_champion(
    *,
    weight_events: tuple[ChampionWeightEvent, ...],
    benchmark_candidates: tuple[BenchmarkEmissionCandidate, ...],
) -> BenchmarkScoredChampionSelection | None:
    ordered_events = sorted(
        weight_events,
        key=lambda event: (event.computed_at, event.sequence_id),
        reverse=True,
    )
    for event in ordered_events:
        if event.champion_uid is None or not event.weights:
            return None
        candidate = _latest_matching_benchmark(event=event, candidates=benchmark_candidates)
        if candidate is None:
            continue
        if candidate.mean_total_score is None:
            raise ValueError("benchmark-scored champion selection missing benchmark score")
        return BenchmarkScoredChampionSelection(
            champion_uid=event.champion_uid,
            weights=event.weights,
            benchmark_score=candidate.mean_total_score,
            champion_artifact_id=event.champion_artifact_id,
        )
    return None


def benchmark_score_to_miner_fraction(score: float) -> float:
    if not isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError("benchmark score must be between 0.0 and 1.0")
    return score * MAX_MINER_EMISSION_FRACTION


def apply_benchmark_scaled_emission(selection: BenchmarkScoredChampionSelection) -> dict[int, float]:
    miner_fraction = benchmark_score_to_miner_fraction(selection.benchmark_score)
    base = {uid: weight for uid, weight in selection.weights.items() if uid != OWNER_UID}
    if not base:
        raise ValueError("benchmarked weights are empty")
    total = float(sum(base.values()))
    if total <= 0.0:
        raise ValueError("benchmarked weights must have positive miner total")

    scaled: dict[int, float] = {
        uid: float(weight) / total * miner_fraction for uid, weight in base.items()
    }
    scaled[OWNER_UID] = 1.0 - miner_fraction
    return scaled


def owner_fallback_weights() -> dict[int, float]:
    return {OWNER_UID: 1.0}


def _latest_matching_benchmark(
    *,
    event: ChampionWeightEvent,
    candidates: tuple[BenchmarkEmissionCandidate, ...],
) -> BenchmarkEmissionCandidate | None:
    matching = tuple(candidate for candidate in candidates if _matches_event(candidate, event))
    if not matching:
        return None
    return max(matching, key=lambda candidate: (candidate.created_at, str(candidate.run_id)))


def _matches_event(candidate: BenchmarkEmissionCandidate, event: ChampionWeightEvent) -> bool:
    if candidate.source_batch_id != event.batch_id:
        return False
    if candidate.mean_total_score is None:
        return False
    if not benchmark_state_is_emission_eligible(candidate.state):
        return False
    if candidate.champion_uid != event.champion_uid:
        return False
    return event.champion_artifact_id is None or candidate.champion_artifact_id == event.champion_artifact_id


__all__ = [
    "BenchmarkEmissionCandidate",
    "BenchmarkScoredChampionSelection",
    "ChampionWeightEvent",
    "EMISSION_BENCHMARK_STATES",
    "MAX_MINER_EMISSION_FRACTION",
    "OWNER_UID",
    "apply_benchmark_scaled_emission",
    "benchmark_score_to_miner_fraction",
    "benchmark_state_is_emission_eligible",
    "compose_champion_weights",
    "owner_fallback_weights",
    "select_benchmark_scored_champion",
]
