from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Protocol, TypeVar
from uuid import UUID

from pydantic import BaseModel

from harnyx_commons.llm.json_utils import pydantic_postprocessor
from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.provider_types import LlmProviderName
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

_CORRECTNESS_SYSTEM_PROMPT = (
    "You are a strict benchmark judge.\n\n"
    "You must decide whether `generated_answer` is materially correct when compared "
    "against `reference_answer`, which is the canonical ground truth for this benchmark item.\n\n"
    "Rules:\n"
    "- Treat `reference_answer` as canonical.\n"
    "- Judge only from `question`, `reference_answer`, and `generated_answer`.\n"
    "- Do not independently solve, research, or reconstruct the answer.\n"
    "- Mark `is_correct` true only when `generated_answer` gives the same material answer as "
    "`reference_answer`.\n"
    "- If `generated_answer` is contradictory, ambiguous, hedged, materially incomplete, or "
    "answers a different thing, mark `is_correct` false.\n"
    "- Extra wording is acceptable only if the core answer is still clearly correct.\n"
    "- Return a short reason tied to the comparison with `reference_answer`.\n\n"
    "Return JSON only with exactly two keys: `is_correct` and `reason`."
)
BENCHMARK_SAMPLE_SIZE = 20
SUPPORTED_BENCHMARK_SCORING_VERSION = "correctness-v1"


class BenchmarkSampleItem(Protocol):
    item_index: int


class BenchmarkRunMetricsInput(Protocol):
    queued_item_count: int
    running_item_count: int
    completed_item_count: int
    failed_item_count: int
    correct_item_count: int
    mean_total_score: float | None


_SampleItemT = TypeVar("_SampleItemT", bound=BenchmarkSampleItem)


class _CorrectnessDecision(BaseModel):
    is_correct: bool
    reason: str


@dataclass(frozen=True, slots=True)
class BenchmarkCorrectnessScore:
    is_correct: bool
    reason: str


class BenchmarkRunState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class BenchmarkItemState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BenchmarkItemOutcome:
    state: BenchmarkItemState
    is_correct: bool | None


@dataclass(frozen=True, slots=True)
class BenchmarkRunMetrics:
    queued_item_count: int
    running_item_count: int
    completed_item_count: int
    failed_item_count: int
    correct_item_count: int
    mean_total_score: float | None

    def derive_state(self) -> BenchmarkRunState:
        return derive_benchmark_run_state(self)


@dataclass(frozen=True, slots=True)
class BenchmarkCorrectnessScoringConfig:
    provider: LlmProviderName
    model: str
    temperature: float | None = None
    max_output_tokens: int | None = 512
    reasoning_effort: str | None = None
    timeout_seconds: float = 60.0


class BenchmarkCorrectnessScoringService:
    def __init__(
        self,
        llm_provider: LlmProviderPort,
        config: BenchmarkCorrectnessScoringConfig,
    ) -> None:
        self._llm = llm_provider
        self._config = config

    async def score(
        self,
        *,
        question: str,
        reference_answer: str,
        generated_answer: str,
    ) -> BenchmarkCorrectnessScore:
        request = LlmRequest(
            provider=self._config.provider,
            model=self._config.model,
            messages=(
                LlmMessage(
                    role="system",
                    content=(LlmMessageContentPart.input_text(_CORRECTNESS_SYSTEM_PROMPT),),
                ),
                LlmMessage(
                    role="user",
                    content=(
                        LlmMessageContentPart.input_text(
                            _render_user_prompt(
                                question=question,
                                reference_answer=reference_answer,
                                generated_answer=generated_answer,
                            )
                        ),
                    ),
                ),
            ),
            output_mode="structured",
            output_schema=_CorrectnessDecision,
            postprocessor=pydantic_postprocessor(_CorrectnessDecision),
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_output_tokens,
            reasoning_effort=self._config.reasoning_effort,
            timeout_seconds=self._config.timeout_seconds,
            use_case="benchmark_correctness_judge",
        )
        response = await self._llm.invoke(request)
        parsed = response.postprocessed
        if parsed is None:
            raise RuntimeError("benchmark correctness judge did not return structured output")
        decision = _CorrectnessDecision.model_validate(parsed)
        return BenchmarkCorrectnessScore(
            is_correct=decision.is_correct,
            reason=decision.reason.strip(),
        )


def _render_user_prompt(
    *,
    question: str,
    reference_answer: str,
    generated_answer: str,
) -> str:
    return (
        f"question:\n{question}\n\n"
        f"reference_answer:\n{reference_answer}\n\n"
        f"generated_answer:\n{generated_answer}"
    )


def aggregate_benchmark_metrics(items: tuple[BenchmarkItemOutcome, ...]) -> BenchmarkRunMetrics:
    queued_item_count = sum(1 for item in items if item.state is BenchmarkItemState.QUEUED)
    running_item_count = sum(1 for item in items if item.state is BenchmarkItemState.RUNNING)
    completed = tuple(item for item in items if item.state is BenchmarkItemState.COMPLETED)
    failed = tuple(item for item in items if item.state is BenchmarkItemState.FAILED)
    failed_item_count = len(failed)
    completed_item_count = len(completed)
    terminal_item_count = completed_item_count + failed_item_count
    if terminal_item_count == 0:
        return BenchmarkRunMetrics(
            queued_item_count=queued_item_count,
            running_item_count=running_item_count,
            completed_item_count=completed_item_count,
            failed_item_count=failed_item_count,
            correct_item_count=0,
            mean_total_score=None,
        )
    correct_item_count = sum(1 for item in completed if item.is_correct is True)
    return BenchmarkRunMetrics(
        queued_item_count=queued_item_count,
        running_item_count=running_item_count,
        completed_item_count=completed_item_count,
        failed_item_count=failed_item_count,
        correct_item_count=correct_item_count,
        mean_total_score=correct_item_count / terminal_item_count,
    )


def derive_benchmark_run_state(metrics: BenchmarkRunMetricsInput) -> BenchmarkRunState:
    if metrics.running_item_count > 0:
        return BenchmarkRunState.RUNNING
    if metrics.queued_item_count > 0:
        return BenchmarkRunState.QUEUED
    if metrics.completed_item_count > 0 and metrics.failed_item_count > 0:
        return BenchmarkRunState.PARTIAL_SUCCESS
    if metrics.completed_item_count > 0:
        return BenchmarkRunState.COMPLETED
    return BenchmarkRunState.FAILED


def project_benchmark_run_state(
    *,
    metrics: BenchmarkRunMetricsInput,
    backing_batch_is_terminal: bool,
) -> BenchmarkRunState:
    if backing_batch_is_terminal:
        if metrics.running_item_count > 0 or metrics.queued_item_count > 0:
            if metrics.completed_item_count > 0:
                return BenchmarkRunState.PARTIAL_SUCCESS
            return BenchmarkRunState.FAILED
        return derive_benchmark_run_state(metrics)
    if metrics.running_item_count > 0:
        return BenchmarkRunState.RUNNING
    if metrics.queued_item_count > 0:
        return BenchmarkRunState.QUEUED
    return BenchmarkRunState.RUNNING


def benchmark_backing_batch_terminalizes_unfinished_items(*, backing_batch_is_terminal: bool) -> bool:
    return backing_batch_is_terminal


def is_supported_benchmark_scoring_version(scoring_version: str) -> bool:
    return scoring_version == SUPPORTED_BENCHMARK_SCORING_VERSION


def unsupported_benchmark_scoring_version_error(scoring_version: str) -> RuntimeError:
    return RuntimeError(
        f"unsupported benchmark scoring_version {scoring_version!r}; "
        f"expected {SUPPORTED_BENCHMARK_SCORING_VERSION!r}"
    )


def sample_benchmark_items(
    *,
    items: tuple[_SampleItemT, ...],
    run_id: UUID,
    dataset_version: str,
    scoring_version: str,
    sample_size: int = BENCHMARK_SAMPLE_SIZE,
) -> tuple[_SampleItemT, ...]:
    if len(items) <= sample_size:
        return items
    sampled_items = sorted(
        items,
        key=lambda item: (
            sha256(
                (
                    f"{dataset_version}:"
                    f"{scoring_version}:"
                    f"{run_id}:"
                    f"{item.item_index}"
                ).encode()
            ).digest(),
            item.item_index,
        ),
    )[:sample_size]
    return tuple(sorted(sampled_items, key=lambda item: item.item_index))


__all__ = [
    "BENCHMARK_SAMPLE_SIZE",
    "BenchmarkItemOutcome",
    "BenchmarkItemState",
    "BenchmarkCorrectnessScore",
    "BenchmarkCorrectnessScoringConfig",
    "BenchmarkCorrectnessScoringService",
    "BenchmarkRunMetrics",
    "BenchmarkRunMetricsInput",
    "BenchmarkRunState",
    "SUPPORTED_BENCHMARK_SCORING_VERSION",
    "aggregate_benchmark_metrics",
    "benchmark_backing_batch_terminalizes_unfinished_items",
    "derive_benchmark_run_state",
    "is_supported_benchmark_scoring_version",
    "project_benchmark_run_state",
    "sample_benchmark_items",
    "unsupported_benchmark_scoring_version_error",
]
