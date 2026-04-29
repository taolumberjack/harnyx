from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from harnyx_commons.domain.miner_task import EvaluationError, MinerTaskErrorCode
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.miner_task_failure_policy import (
    ProviderFailureEvidence,
    SuccessfulLlmSample,
    TimeoutAttributionKind,
    TimeoutObservationEvidence,
    classify_timeout_attribution,
    delivery_exclusion_from_completed_pair_results,
    is_provider_caused_terminal_failure,
    is_script_validation_sandbox_invocation,
    is_timeout_sandbox_invocation,
    provider_batch_failure_evidence,
)


@dataclass(frozen=True, slots=True)
class _Run:
    completed_at: datetime | None
    artifact_id: UUID
    task_id: UUID


@dataclass(frozen=True, slots=True)
class _Validator:
    uid: int


@dataclass(frozen=True, slots=True)
class _Specifics:
    error: EvaluationError | None


@dataclass(frozen=True, slots=True)
class _Submission:
    run: _Run
    validator: _Validator
    specifics: _Specifics


def test_provider_batch_failure_requires_minimum_calls_and_failure_rate() -> None:
    below_calls: ProviderFailureEvidence = {
        "provider": "chutes",
        "model": "model",
        "total_calls": 9,
        "failed_calls": 9,
    }
    threshold_met: ProviderFailureEvidence = {
        "provider": "chutes",
        "model": "model",
        "total_calls": 20,
        "failed_calls": 20,
    }

    assert provider_batch_failure_evidence((below_calls, threshold_met)) == threshold_met


def test_delivery_exclusion_selects_first_validator_owned_completed_pair_failure() -> None:
    observed_at = datetime(2026, 4, 29, 4, 0, tzinfo=UTC)
    completed_at = datetime(2026, 4, 29, 4, 1, tzinfo=UTC)
    validator_owned_artifact_id = uuid4()
    validator_owned_task_id = uuid4()
    validator_uid = 42

    decision = delivery_exclusion_from_completed_pair_results(
        (
            _submission(
                code=MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED,
                completed_at=completed_at,
                artifact_id=validator_owned_artifact_id,
                task_id=validator_owned_task_id,
                validator_uid=validator_uid,
            ),
            _submission(
                code=MinerTaskErrorCode.TIMEOUT_MINER_OWNED,
                completed_at=completed_at,
                validator_uid=validator_uid,
            ),
        ),
        observed_at=observed_at,
    )

    assert decision is not None
    assert decision.error.code is MinerTaskErrorCode.SANDBOX_INVOCATION_FAILED
    assert decision.artifact_id == validator_owned_artifact_id
    assert decision.task_id == validator_owned_task_id
    assert decision.uid == validator_uid
    assert decision.occurred_at == completed_at


def test_delivery_exclusion_uses_observed_at_when_completed_at_is_missing() -> None:
    observed_at = datetime(2026, 4, 29, 4, 0, tzinfo=UTC)

    decision = delivery_exclusion_from_completed_pair_results(
        (
            _submission(
                code=MinerTaskErrorCode.TIMEOUT_INCONCLUSIVE,
                completed_at=None,
            ),
        ),
        observed_at=observed_at,
    )

    assert decision is not None
    assert decision.occurred_at == observed_at


def test_delivery_exclusion_ignores_miner_owned_pair_failures() -> None:
    observed_at = datetime(2026, 4, 29, 4, 0, tzinfo=UTC)

    decision = delivery_exclusion_from_completed_pair_results(
        (
            _submission(code=MinerTaskErrorCode.TIMEOUT_MINER_OWNED),
            _submission(code=MinerTaskErrorCode.SCRIPT_VALIDATION_FAILED),
        ),
        observed_at=observed_at,
    )

    assert decision is None


def test_timeout_attribution_marks_fast_llm_sample_as_miner_owned() -> None:
    observation = TimeoutObservationEvidence(
        successful_llm_samples=(SuccessfulLlmSample(elapsed_ms=1000.0, total_tokens=100, llm_tps=100.0),),
        session_summary=ToolUsageSummary(),
        session_elapsed_ms=60000.0,
    )

    assert (
        classify_timeout_attribution(
            observation=observation,
            successful_baseline_tps=100.0,
            prior_timeout_observations=(),
        )
        is TimeoutAttributionKind.MINER_OWNED
    )


def test_timeout_attribution_waits_until_observations_are_exhausted_without_baseline() -> None:
    observation = TimeoutObservationEvidence(
        successful_llm_samples=(),
        session_summary=ToolUsageSummary(),
        session_elapsed_ms=60000.0,
    )

    assert (
        classify_timeout_attribution(
            observation=observation,
            successful_baseline_tps=None,
            prior_timeout_observations=(observation,),
        )
        is None
    )
    assert (
        classify_timeout_attribution(
            observation=observation,
            successful_baseline_tps=None,
            prior_timeout_observations=(observation, observation),
        )
        is TimeoutAttributionKind.MINER_OWNED
    )


def test_sandbox_failure_shape_classifiers_expose_validator_attribution_policy() -> None:
    assert is_timeout_sandbox_invocation(status_code=504, detail_exception="TimeoutError")
    assert is_script_validation_sandbox_invocation(detail_code="MissingEntrypoint")
    assert is_provider_caused_terminal_failure(
        detail_code="UnhandledException",
        detail_exception="ToolInvocationError",
        detail_error="tool invocation failed with 400: tool execution failed",
    )


def _submission(
    *,
    code: MinerTaskErrorCode,
    completed_at: datetime | None = None,
    artifact_id: UUID | None = None,
    task_id: UUID | None = None,
    validator_uid: int = 1,
) -> _Submission:
    return _Submission(
        run=_Run(
            completed_at=completed_at,
            artifact_id=artifact_id or uuid4(),
            task_id=task_id or uuid4(),
        ),
        validator=_Validator(uid=validator_uid),
        specifics=_Specifics(error=EvaluationError(code=code, message=f"{code.value} happened")),
    )
