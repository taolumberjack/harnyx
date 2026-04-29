from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from harnyx_commons.miner_task_emission import (
    BenchmarkEmissionCandidate,
    BenchmarkScoredChampionSelection,
    ChampionWeightEvent,
    apply_benchmark_scaled_emission,
    owner_fallback_weights,
    select_benchmark_scored_champion,
)


def test_apply_benchmark_scaled_emission_caps_miner_share_at_ten_percent() -> None:
    weights = apply_benchmark_scaled_emission(
        BenchmarkScoredChampionSelection(
            champion_uid=7,
            weights={7: 0.6, 8: 0.4},
            benchmark_score=0.5,
        )
    )

    assert weights == {
        0: pytest.approx(0.95),
        7: pytest.approx(0.03),
        8: pytest.approx(0.02),
    }


def test_apply_benchmark_scaled_emission_rejects_invalid_scores() -> None:
    selection = BenchmarkScoredChampionSelection(champion_uid=7, weights={7: 1.0}, benchmark_score=1.1)

    with pytest.raises(ValueError, match="benchmark score must be between 0.0 and 1.0"):
        apply_benchmark_scaled_emission(selection)


def test_owner_fallback_weights_assigns_all_emission_to_owner() -> None:
    assert owner_fallback_weights() == {0: 1.0}


def test_select_benchmark_scored_champion_skips_newer_unscored_event() -> None:
    now = datetime(2026, 4, 28, tzinfo=UTC)
    old_batch_id = uuid4()
    new_batch_id = uuid4()
    old_artifact_id = uuid4()
    new_artifact_id = uuid4()

    selection = select_benchmark_scored_champion(
        weight_events=(
            ChampionWeightEvent(
                batch_id=old_batch_id,
                computed_at=now,
                sequence_id=1,
                champion_uid=7,
                weights={7: 1.0},
                champion_artifact_id=old_artifact_id,
            ),
            ChampionWeightEvent(
                batch_id=new_batch_id,
                computed_at=now + timedelta(minutes=1),
                sequence_id=2,
                champion_uid=8,
                weights={8: 1.0},
                champion_artifact_id=new_artifact_id,
            ),
        ),
        benchmark_candidates=(
            BenchmarkEmissionCandidate(
                source_batch_id=old_batch_id,
                run_id=uuid4(),
                created_at=now,
                state="completed",
                champion_uid=7,
                champion_artifact_id=old_artifact_id,
                mean_total_score=0.25,
            ),
        ),
    )

    assert selection == BenchmarkScoredChampionSelection(
        champion_uid=7,
        weights={7: 1.0},
        benchmark_score=0.25,
        champion_artifact_id=old_artifact_id,
    )


def test_select_benchmark_scored_champion_stops_when_latest_event_clears_champion() -> None:
    now = datetime(2026, 4, 28, tzinfo=UTC)
    batch_id = uuid4()
    artifact_id = uuid4()

    selection = select_benchmark_scored_champion(
        weight_events=(
            ChampionWeightEvent(
                batch_id=batch_id,
                computed_at=now,
                sequence_id=1,
                champion_uid=7,
                weights={7: 1.0},
                champion_artifact_id=artifact_id,
            ),
            ChampionWeightEvent(
                batch_id=uuid4(),
                computed_at=now + timedelta(minutes=1),
                sequence_id=2,
                champion_uid=None,
                weights={},
                champion_artifact_id=None,
            ),
        ),
        benchmark_candidates=(
            BenchmarkEmissionCandidate(
                source_batch_id=batch_id,
                run_id=uuid4(),
                created_at=now,
                state="completed",
                champion_uid=7,
                champion_artifact_id=artifact_id,
                mean_total_score=1.0,
            ),
        ),
    )

    assert selection is None
