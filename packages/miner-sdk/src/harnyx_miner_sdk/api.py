from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, field_validator

from harnyx_miner_sdk._internal.tool_invoker import _current_tool_invoker
from harnyx_miner_sdk.llm import LlmResponse
from harnyx_miner_sdk.tools.http_models import (
    ToolBudgetDTO,
    ToolExecuteResponseDTO,
    ToolResultDTO,
    ToolUsageDTO,
)
from harnyx_miner_sdk.tools.search_models import (
    FetchPageResponse,
    SearchAiSearchResponse,
    SearchWebSearchResponse,
)

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


class _SearchWebInvocationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_queries: tuple[str, ...]
    num: int | None = None

    @field_validator("search_queries", mode="before")
    @classmethod
    def _normalize_search_queries(cls, value: object) -> object:
        if isinstance(value, str):
            return (value,)
        return value

def _parse_execute_response(raw_response: object) -> ToolExecuteResponseDTO:
    return ToolExecuteResponseDTO.model_validate(raw_response)


def _require_response_mapping(response_payload: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(response_payload, Mapping):
        raise RuntimeError(label)
    return cast(Mapping[str, Any], response_payload)


async def test_tool(message: str) -> ToolCallResponse[TestToolResponse]:
    """Invoke the validator-hosted test tool."""

    raw_response = await _current_tool_invoker().invoke("test_tool", args=(message,), kwargs={})
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


async def tooling_info() -> ToolCallResponse[dict[str, Any]]:
    """Fetch tool pricing and current session budget metadata."""

    raw_response = await _current_tool_invoker().invoke("tooling_info", args=(), kwargs={})
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
    **kwargs: Any,
) -> ToolCallResponse[SearchWebSearchResponse]:
    """Execute the validator-hosted search tool and return its response payload."""

    payload = _SearchWebInvocationPayload.model_validate(
        {"search_queries": search_queries, **kwargs}
    ).model_dump(exclude_none=True, mode="json")
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


async def search_ai(prompt: str, /, **kwargs: Any) -> ToolCallResponse[SearchAiSearchResponse]:
    """Execute the validator-hosted AI search tool and return its response payload."""

    payload = {"prompt": prompt}
    payload.update(kwargs)
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


async def fetch_page(url: str, /, **kwargs: Any) -> ToolCallResponse[FetchPageResponse]:
    """Execute the validator-hosted page fetch tool and return its response payload."""

    payload = {"url": url}
    payload.update(kwargs)
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
    **params: Any,
) -> LlmChatResult:
    """Invoke the validator-hosted LLM chat tool and return its response payload."""

    payload = {"model": model, "messages": [dict(message) for message in messages]}
    if "provider" in params:
        params = {k: v for k, v in params.items() if k != "provider"}
    if params:
        payload.update(params)
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
