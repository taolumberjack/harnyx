"""Miner-task failure attribution policies."""

from __future__ import annotations

from collections.abc import Sequence
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
    elapsed_ms: float
    total_tokens: int
    llm_tps: float


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
        execution = receipt.details.execution
        if execution is None or execution.elapsed_ms is None or execution.elapsed_ms <= 0:
            continue
        total_tokens = _receipt_total_tokens(receipt)
        if total_tokens is None or total_tokens <= 0:
            continue
        samples.append(
            SuccessfulLlmSample(
                elapsed_ms=execution.elapsed_ms,
                total_tokens=total_tokens,
                llm_tps=total_tokens / (execution.elapsed_ms / 1000.0),
            )
        )
    return tuple(samples)


def classify_timeout_attribution(
    *,
    observation: TimeoutObservationEvidence,
    successful_baseline_tps: float | None,
    prior_timeout_observations: tuple[TimeoutObservationEvidence, ...],
) -> TimeoutAttributionKind | None:
    comparable_samples = observation.successful_llm_samples
    exhausted = len(prior_timeout_observations) + 1 >= TIMEOUT_REVIEW_MAX_OBSERVATIONS
    threshold_tps = (
        None
        if successful_baseline_tps is None
        else successful_baseline_tps / TIMEOUT_TPS_SLOWDOWN_FACTOR
    )
    if threshold_tps is None:
        return TimeoutAttributionKind.MINER_OWNED if exhausted else None
    if any(sample.llm_tps >= threshold_tps for sample in comparable_samples):
        return TimeoutAttributionKind.MINER_OWNED
    if not exhausted:
        return None
    if comparable_samples and all(sample.llm_tps < threshold_tps for sample in comparable_samples):
        return TimeoutAttributionKind.NOT_MINER_OWNED
    return TimeoutAttributionKind.MINER_OWNED


def slowest_successful_llm_tps(receipts: Sequence[ToolCall]) -> float | None:
    samples = successful_llm_samples(receipts)
    if not samples:
        return None
    return min(sample.llm_tps for sample in samples)


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
    "ValidatorDeliveryExclusion",
    "classify_timeout_attribution",
    "delivery_exclusion_from_completed_pair_results",
    "is_provider_caused_terminal_failure",
    "is_script_validation_sandbox_invocation",
    "is_timeout_sandbox_invocation",
    "provider_batch_failure_evidence",
    "provider_batch_failure_message",
    "slowest_successful_llm_tps",
    "successful_llm_samples",
]
