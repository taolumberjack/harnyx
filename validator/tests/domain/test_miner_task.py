from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from caster_commons.domain.miner_task import EvaluationDetails, EvaluationError, Response, ScoreBreakdown
from caster_validator.domain.evaluation import MinerTaskRun

_NOW = datetime.now(UTC)


def test_successful_run_requires_response() -> None:
    with pytest.raises(ValidationError, match="response"):
        MinerTaskRun(
            session_id=uuid4(),
            uid=1,
            artifact_id=uuid4(),
            task_id=uuid4(),
            details=EvaluationDetails(
                score_breakdown=ScoreBreakdown(
                    comparison_score=1.0,
                    similarity_score=0.5,
                    total_score=0.75,
                    scoring_version="v1",
                )
            ),
            completed_at=_NOW,
        )


def test_failed_run_rejects_response() -> None:
    with pytest.raises(ValidationError, match="must not include a response"):
        MinerTaskRun(
            session_id=uuid4(),
            uid=1,
            artifact_id=uuid4(),
            task_id=uuid4(),
            response=Response(text="This should not be present."),
            details=EvaluationDetails(
                error=EvaluationError(code="sandbox_failed", message="sandbox failed"),
            ),
            completed_at=_NOW,
        )
