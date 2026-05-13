from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Generic, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, field_validator, model_validator

from harnyx_miner_sdk._internal.tool_invoker import _current_tool_invoker
from harnyx_miner_sdk.llm import LlmResponse, LlmThinkingConfig
from harnyx_miner_sdk.tools.http_models import (
    ToolBudgetDTO,
    ToolExecuteResponseDTO,
    ToolResultDTO,
    ToolUsageDTO,
)
from harnyx_miner_sdk.tools.search_models import (
    FetchPageRequest,
    FetchPageResponse,
    SearchAiSearchRequest,
    SearchAiSearchResponse,
    SearchWebSearchRequest,
    SearchWebSearchResponse,
)
from harnyx_miner_sdk.tools.types import ToolInvocationTimeout

TResponse = TypeVar("TResponse")


@dataclass(frozen=True)
class ToolCallResponse(Generic[TResponse]):
    """Typed envelope returned by all hosted tool calls."""

    receipt_id: str
    response: TResponse
    results: tuple[ToolResultDTO, ...]
    result_policy: str
    cost_usd: float | None
    usage: ToolUsageDTO | None
    budget: ToolBudgetDTO


@dataclass(frozen=True)
class LlmChatResult(ToolCallResponse[LlmResponse]):
    """Typed payload returned by the llm_chat tool."""

    @property
    def llm(self) -> LlmResponse:
        return self.response


class TestToolResponse(BaseModel):
    status: str = ""
    echo: str = ""

    @field_validator("status", "echo", mode="before")
    @classmethod
    def _coerce_text(cls, value: object) -> str:
        return "" if value is None else str(value)


class _ToolingInfoInvocationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout: ToolInvocationTimeout | None = None


class _TestToolInvocationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    timeout: ToolInvocationTimeout | None = None


class _LlmChatThinkingPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: StrictBool
    budget: StrictInt | None = None
    effort: Literal["low", "medium", "high"] | None = None

    @field_validator("budget")
    @classmethod
    def _validate_budget(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("thinking.budget must be positive")
        return value

    @model_validator(mode="after")
    def _validate_single_tuning_knob(self) -> _LlmChatThinkingPayload:
        if self.budget is not None and self.effort is not None:
            raise ValueError("thinking.budget and thinking.effort are mutually exclusive")
        return self


class _LlmChatInvocationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[dict[str, Any]]
    timeout: ToolInvocationTimeout | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    max_tokens: int | None = None
    response_format: Any | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Literal["auto", "required"] | None = None
    include: list[str] | None = None
    thinking: _LlmChatThinkingPayload | None = None

def _parse_execute_response(raw_response: object) -> ToolExecuteResponseDTO:
    return ToolExecuteResponseDTO.model_validate(raw_response)


def _require_response_mapping(response_payload: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(response_payload, Mapping):
        raise RuntimeError(label)
    return cast(Mapping[str, Any], response_payload)


async def test_tool(
    message: str,
    *,
    timeout: float | None = None,
) -> ToolCallResponse[TestToolResponse]:
    """Invoke the validator-hosted test tool."""

    kwargs: dict[str, Any] = {}
    if timeout is not None:
        payload = _TestToolInvocationPayload.model_validate({"message": message, "timeout": timeout})
        message = payload.message
        kwargs["timeout"] = payload.timeout
    raw_response = await _current_tool_invoker().invoke("test_tool", args=(message,), kwargs=kwargs)
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="test_tool response payload must be a mapping")
    response = TestToolResponse.model_validate(response_payload)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


async def tooling_info(
    *,
    timeout: float | None = None,
) -> ToolCallResponse[dict[str, Any]]:
    """Fetch tool pricing and current session budget metadata."""

    payload = _ToolingInfoInvocationPayload.model_validate({"timeout": timeout}).model_dump(
        exclude_none=True,
        mode="json",
    )
    raw_response = await _current_tool_invoker().invoke("tooling_info", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="tooling_info response payload must be a mapping")
    response = dict(response_payload)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


async def search_web(
    search_queries: str | Sequence[str],
    /,
    *,
    timeout: float | None = None,
    **kwargs: Any,
) -> ToolCallResponse[SearchWebSearchResponse]:
    """Execute the validator-hosted search tool and return its response payload."""

    raw_payload = {"search_queries": search_queries, **kwargs}
    if timeout is not None:
        raw_payload["timeout"] = timeout
    payload = SearchWebSearchRequest.model_validate(raw_payload).model_dump(exclude_none=True, mode="json")
    raw_response = await _current_tool_invoker().invoke("search_web", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="search_web response payload must be a mapping")
    response = SearchWebSearchResponse.model_validate(response_payload)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


async def search_ai(
    prompt: str,
    /,
    *,
    timeout: float | None = None,
    **kwargs: Any,
) -> ToolCallResponse[SearchAiSearchResponse]:
    """Execute the validator-hosted AI search tool and return its response payload."""

    raw_payload = {"prompt": prompt, **kwargs}
    if timeout is not None:
        raw_payload["timeout"] = timeout
    payload = SearchAiSearchRequest.model_validate(raw_payload).model_dump(exclude_none=True, mode="json")
    raw_response = await _current_tool_invoker().invoke("search_ai", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="search_ai response payload must be a mapping")
    response = SearchAiSearchResponse.model_validate(response_payload)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


async def fetch_page(
    url: str,
    /,
    *,
    timeout: float | None = None,
    **kwargs: Any,
) -> ToolCallResponse[FetchPageResponse]:
    """Execute the validator-hosted page fetch tool and return its response payload."""

    raw_payload = {"url": url, **kwargs}
    if timeout is not None:
        raw_payload["timeout"] = timeout
    payload = FetchPageRequest.model_validate(raw_payload).model_dump(exclude_none=True, mode="json")
    raw_response = await _current_tool_invoker().invoke("fetch_page", args=(), kwargs=payload)
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="fetch_page response payload must be a mapping")
    response = FetchPageResponse.model_validate(response_payload)
    return ToolCallResponse(
        receipt_id=dto.receipt_id,
        response=response,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


async def llm_chat(
    *,
    messages: Sequence[Mapping[str, Any]],
    model: str,
    thinking: Mapping[str, Any] | LlmThinkingConfig | None = None,
    timeout: float | None = None,
    **params: Any,
) -> LlmChatResult:
    """Invoke the validator-hosted LLM chat tool and return its response payload."""

    payload_raw = {"model": model, "messages": [dict(message) for message in messages]}
    if thinking is not None:
        payload_raw["thinking"] = asdict(thinking) if isinstance(thinking, LlmThinkingConfig) else thinking
    if timeout is not None:
        payload_raw["timeout"] = timeout
    if "provider" in params:
        params = {k: v for k, v in params.items() if k != "provider"}
    if params:
        payload_raw.update(params)
    payload = _LlmChatInvocationPayload.model_validate(payload_raw).model_dump(
        exclude_none=True,
        mode="json",
    )
    raw_response = await _current_tool_invoker().invoke(
        "llm_chat",
        args=(),
        kwargs=payload,
    )
    dto = _parse_execute_response(raw_response)
    response_payload = _require_response_mapping(dto.response, label="llm_chat response missing 'response' payload")
    llm = LlmResponse.from_payload(response_payload)
    return LlmChatResult(
        receipt_id=dto.receipt_id,
        response=llm,
        results=dto.results,
        result_policy=dto.result_policy,
        cost_usd=dto.cost_usd,
        usage=dto.usage,
        budget=dto.budget,
    )


__all__ = [
    "fetch_page",
    "llm_chat",
    "search_web",
    "search_ai",
    "test_tool",
    "tooling_info",
    "ToolCallResponse",
    "LlmChatResult",
    "TestToolResponse",
]
