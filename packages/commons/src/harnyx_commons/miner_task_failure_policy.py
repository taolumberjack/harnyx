"""Miner-task failure attribution policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import NotRequired, Protocol, TypedDict
from uuid import UUID

from harnyx_commons.domain.miner_task import (
    EvaluationError,
    is_delivery_disqualifying_validator_pair_error,
)
from harnyx_commons.domain.tool_call import ToolCall
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.json_types import JsonValue
from harnyx_commons.llm.tool_models import ToolModelName, parse_tool_model

PROVIDER_BATCH_MIN_TOTAL_CALLS = 10
PROVIDER_BATCH_MIN_FAILURE_RATE = 0.95
TIMEOUT_REVIEW_MAX_OBSERVATIONS = 3
TIMEOUT_TPS_SLOWDOWN_FACTOR = 2.0
TERMINAL_TIMEOUT_ERROR_MESSAGE = "terminal timeout"

SANDBOX_TIMEOUT_EXCEPTIONS = frozenset({"TimeoutError", "TimeoutException"})
SANDBOX_DETAIL_CODE_UNHANDLED_EXCEPTION = "UnhandledException"
SANDBOX_DETAIL_CODE_MISSING_ENTRYPOINT = "MissingEntrypoint"
SANDBOX_DETAIL_CODE_PRELOAD_FAILED = "PreloadFailed"


class ProviderFailureEvidence(TypedDict):
    provider: str
    model: str
    total_calls: int
    failed_calls: int
    failure_reason: NotRequired[str]


class TimeoutAttributionKind(StrEnum):
    MINER_OWNED = "miner_owned"
    NOT_MINER_OWNED = "not_miner_owned"


@dataclass(frozen=True, slots=True)
class SuccessfulLlmSample:
    model: ToolModelName
    elapsed_ms: float
    total_tokens: int
    llm_tps: float


@dataclass(frozen=True, slots=True)
class ValidatorModelLlmBaseline:
    slowest_tps_by_model: Mapping[ToolModelName, float]

    @classmethod
    def empty(cls) -> ValidatorModelLlmBaseline:
        return cls(slowest_tps_by_model={})

    @classmethod
    def from_samples(cls, samples: Sequence[SuccessfulLlmSample]) -> ValidatorModelLlmBaseline:
        slowest_by_model: dict[ToolModelName, float] = {}
        for sample in samples:
            current = slowest_by_model.get(sample.model)
            if current is None or sample.llm_tps < current:
                slowest_by_model[sample.model] = sample.llm_tps
        return cls(slowest_tps_by_model=slowest_by_model)

    def threshold_for(self, model: ToolModelName) -> float | None:
        baseline_tps = self.slowest_tps_by_model.get(model)
        if baseline_tps is None:
            return None
        return baseline_tps / TIMEOUT_TPS_SLOWDOWN_FACTOR

    def merge(self, other: ValidatorModelLlmBaseline) -> ValidatorModelLlmBaseline:
        merged = dict(self.slowest_tps_by_model)
        for model, tps in other.slowest_tps_by_model.items():
            current = merged.get(model)
            if current is None or tps < current:
                merged[model] = tps
        return ValidatorModelLlmBaseline(slowest_tps_by_model=merged)


@dataclass(frozen=True, slots=True)
class TimeoutObservationEvidence:
    successful_llm_samples: tuple[SuccessfulLlmSample, ...]
    session_summary: ToolUsageSummary
    session_elapsed_ms: float


class DeliveryRunInput(Protocol):
    @property
    def completed_at(self) -> datetime | None: ...

    @property
    def artifact_id(self) -> UUID: ...

    @property
    def task_id(self) -> UUID: ...


class DeliveryValidatorInput(Protocol):
    @property
    def uid(self) -> int: ...


class DeliverySpecificsInput(Protocol):
    @property
    def error(self) -> EvaluationError | None: ...


class DeliverySubmissionInput(Protocol):
    @property
    def run(self) -> DeliveryRunInput: ...

    @property
    def validator(self) -> DeliveryValidatorInput: ...

    @property
    def specifics(self) -> DeliverySpecificsInput: ...


@dataclass(frozen=True, slots=True)
class ValidatorDeliveryExclusion:
    error: EvaluationError
    occurred_at: datetime
    artifact_id: UUID
    task_id: UUID
    uid: int


def delivery_exclusion_from_completed_pair_results(
    submissions: Sequence[DeliverySubmissionInput],
    *,
    observed_at: datetime,
) -> ValidatorDeliveryExclusion | None:
    for submission in submissions:
        error = submission.specifics.error
        if error is None:
            continue
        if not is_delivery_disqualifying_validator_pair_error(error.code):
            continue
        return ValidatorDeliveryExclusion(
            error=error,
            occurred_at=submission.run.completed_at or observed_at,
            artifact_id=submission.run.artifact_id,
            task_id=submission.run.task_id,
            uid=submission.validator.uid,
        )
    return None


def provider_batch_failure_evidence(
    provider_failures: tuple[ProviderFailureEvidence, ...],
) -> ProviderFailureEvidence | None:
    for evidence in provider_failures:
        if evidence["total_calls"] < PROVIDER_BATCH_MIN_TOTAL_CALLS:
            continue
        if evidence["failed_calls"] / evidence["total_calls"] <= PROVIDER_BATCH_MIN_FAILURE_RATE:
            continue
        return evidence
    return None


def provider_batch_failure_message(evidence: ProviderFailureEvidence) -> str:
    reason = evidence.get("failure_reason")
    if reason:
        return (
            "provider failure threshold reached "
            f"(provider={evidence['provider']} model={evidence['model']} "
            f"failed_calls={evidence['failed_calls']} total_calls={evidence['total_calls']} "
            f"reason={reason})"
        )
    return (
        "provider failure threshold reached "
        f"(provider={evidence['provider']} model={evidence['model']} "
        f"failed_calls={evidence['failed_calls']} total_calls={evidence['total_calls']})"
    )


def is_provider_caused_terminal_failure(
    *,
    detail_code: str | None,
    detail_exception: str | None,
    detail_error: str | None,
) -> bool:
    if detail_code != SANDBOX_DETAIL_CODE_UNHANDLED_EXCEPTION:
        return False
    if detail_exception != "ToolInvocationError":
        return False
    return detail_error == "tool invocation failed with 400: tool execution failed"


def is_timeout_sandbox_invocation(
    *,
    status_code: int | None,
    detail_exception: str | None,
) -> bool:
    return status_code == 504 and detail_exception in SANDBOX_TIMEOUT_EXCEPTIONS


def is_script_validation_sandbox_invocation(*, detail_code: str | None) -> bool:
    return detail_code in {
        SANDBOX_DETAIL_CODE_MISSING_ENTRYPOINT,
        SANDBOX_DETAIL_CODE_PRELOAD_FAILED,
    }


def successful_llm_samples(receipts: Sequence[ToolCall]) -> tuple[SuccessfulLlmSample, ...]:
    samples: list[SuccessfulLlmSample] = []
    for receipt in receipts:
        if not receipt.is_successful() or receipt.tool != "llm_chat":
            continue
        model = _receipt_llm_model(receipt)
        if model is None:
            continue
        execution = receipt.details.execution
        if execution is None or execution.elapsed_ms is None or execution.elapsed_ms <= 0:
            continue
        total_tokens = _receipt_total_tokens(receipt)
        if total_tokens is None or total_tokens <= 0:
            continue
        samples.append(
            SuccessfulLlmSample(
                model=model,
                elapsed_ms=execution.elapsed_ms,
                total_tokens=total_tokens,
                llm_tps=total_tokens / (execution.elapsed_ms / 1000.0),
            )
        )
    return tuple(samples)


def classify_timeout_attribution(
    *,
    observation: TimeoutObservationEvidence,
    validator_model_llm_baseline: ValidatorModelLlmBaseline,
    prior_timeout_observations: tuple[TimeoutObservationEvidence, ...],
) -> TimeoutAttributionKind | None:
    comparable_samples = tuple(
        sample
        for timeout_observation in (*prior_timeout_observations, observation)
        for sample in timeout_observation.successful_llm_samples
        if validator_model_llm_baseline.threshold_for(sample.model) is not None
    )
    exhausted = len(prior_timeout_observations) + 1 >= TIMEOUT_REVIEW_MAX_OBSERVATIONS
    if any(_is_slow_llm_sample(sample, validator_model_llm_baseline) for sample in comparable_samples):
        return TimeoutAttributionKind.NOT_MINER_OWNED if exhausted else None
    if any(_is_fast_llm_sample(sample, validator_model_llm_baseline) for sample in comparable_samples):
        return TimeoutAttributionKind.MINER_OWNED
    return TimeoutAttributionKind.MINER_OWNED if exhausted else None


def validator_model_llm_baseline(receipts: Sequence[ToolCall]) -> ValidatorModelLlmBaseline:
    return ValidatorModelLlmBaseline.from_samples(successful_llm_samples(receipts))


def _is_slow_llm_sample(
    sample: SuccessfulLlmSample,
    baseline: ValidatorModelLlmBaseline,
) -> bool:
    threshold_tps = baseline.threshold_for(sample.model)
    if threshold_tps is None:
        return False
    return sample.llm_tps < threshold_tps


def _is_fast_llm_sample(
    sample: SuccessfulLlmSample,
    baseline: ValidatorModelLlmBaseline,
) -> bool:
    threshold_tps = baseline.threshold_for(sample.model)
    if threshold_tps is None:
        return False
    return sample.llm_tps >= threshold_tps


def _receipt_total_tokens(receipt: ToolCall) -> int | None:
    response_payload = receipt.details.response_payload
    if not isinstance(response_payload, dict):
        return None
    usage = response_payload.get("usage")
    if not isinstance(usage, dict):
        return None
    total_tokens = usage.get("total_tokens")
    if not isinstance(total_tokens, int):
        return None
    return total_tokens


def _receipt_llm_model(receipt: ToolCall) -> ToolModelName | None:
    request_payload = receipt.details.request_payload
    raw_model = _raw_model_from_request_payload(request_payload)
    if raw_model is None:
        return None
    return parse_tool_model(raw_model)


def _raw_model_from_request_payload(payload: JsonValue | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    direct_model = payload.get("model")
    if isinstance(direct_model, str):
        return direct_model

    kwargs = payload.get("kwargs")
    if isinstance(kwargs, dict):
        kwargs_model = kwargs.get("model")
        if isinstance(kwargs_model, str):
            return kwargs_model

    args = payload.get("args")
    if isinstance(args, list) and args:
        first_arg = args[0]
        if isinstance(first_arg, dict):
            arg_model = first_arg.get("model")
            if isinstance(arg_model, str):
                return arg_model
    return None


__all__ = [
    "PROVIDER_BATCH_MIN_FAILURE_RATE",
    "PROVIDER_BATCH_MIN_TOTAL_CALLS",
    "SANDBOX_DETAIL_CODE_MISSING_ENTRYPOINT",
    "SANDBOX_DETAIL_CODE_PRELOAD_FAILED",
    "SANDBOX_DETAIL_CODE_UNHANDLED_EXCEPTION",
    "SANDBOX_TIMEOUT_EXCEPTIONS",
    "TERMINAL_TIMEOUT_ERROR_MESSAGE",
    "TIMEOUT_REVIEW_MAX_OBSERVATIONS",
    "TIMEOUT_TPS_SLOWDOWN_FACTOR",
    "DeliveryRunInput",
    "DeliverySpecificsInput",
    "DeliverySubmissionInput",
    "DeliveryValidatorInput",
    "ProviderFailureEvidence",
    "SuccessfulLlmSample",
    "TimeoutAttributionKind",
    "TimeoutObservationEvidence",
    "ValidatorModelLlmBaseline",
    "ValidatorDeliveryExclusion",
    "classify_timeout_attribution",
    "delivery_exclusion_from_completed_pair_results",
    "is_provider_caused_terminal_failure",
    "is_script_validation_sandbox_invocation",
    "is_timeout_sandbox_invocation",
    "provider_batch_failure_evidence",
    "provider_batch_failure_message",
    "successful_llm_samples",
    "validator_model_llm_baseline",
]
