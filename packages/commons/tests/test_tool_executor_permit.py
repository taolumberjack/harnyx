from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest

from harnyx_commons.domain.tool_call import ToolCall, ToolCallDetails, ToolCallOutcome, ToolResultPolicy
from harnyx_commons.errors import ConcurrencyLimitError
from harnyx_commons.tools.dto import ToolBudgetSnapshot, ToolInvocationRequest, ToolInvocationResult
from harnyx_commons.tools.executor import execute_tool_with_concurrency_permit
from harnyx_commons.tools.token_semaphore import (
    DEFAULT_TOOL_CONCURRENCY_LIMITS,
    ToolConcurrencyLimiter,
    ToolConcurrencyLimits,
)
from harnyx_commons.tools.types import ToolName

pytestmark = pytest.mark.anyio("asyncio")
TEST_TOKEN = "token"  # noqa: S105
DEFAULT_LLM_MODEL = "deepseek-ai/DeepSeek-V3.2-TEE"
OTHER_LLM_MODEL = "zai-org/GLM-5-TEE"


class RecordingExecutor:
    def __init__(self, result: ToolInvocationResult) -> None:
        self.result = result
        self.invocations: list[ToolInvocationRequest] = []

    async def execute(self, invocation: ToolInvocationRequest) -> ToolInvocationResult:
        self.invocations.append(invocation)
        return self.result


class FailingExecutor:
    async def execute(self, _: ToolInvocationRequest) -> ToolInvocationResult:
        raise RuntimeError("expected failure")


def _invocation(token: str, tool: ToolName = "search_web") -> ToolInvocationRequest:
    kwargs = {"model": DEFAULT_LLM_MODEL} if tool == "llm_chat" else {"query": "demo"}
    return ToolInvocationRequest(
        session_id=uuid4(),
        token=token,
        tool=tool,
        args=(),
        kwargs=kwargs,
    )


def _llm_invocation(token: str, model: str, *, positional_payload: bool = False) -> ToolInvocationRequest:
    payload = {"model": model, "messages": [{"role": "user", "content": "demo"}]}
    if positional_payload:
        return ToolInvocationRequest(
            session_id=uuid4(),
            token=token,
            tool="llm_chat",
            args=(payload,),
            kwargs={},
        )
    return ToolInvocationRequest(
        session_id=uuid4(),
        token=token,
        tool="llm_chat",
        args=(),
        kwargs=payload,
    )


def _result(session_id, tool: ToolName = "search_web") -> ToolInvocationResult:
    return ToolInvocationResult(
        receipt=ToolCall(
            receipt_id="receipt-1",
            session_id=session_id,
            uid=7,
            tool=tool,
            issued_at=datetime(2026, 5, 7, tzinfo=UTC),
            outcome=ToolCallOutcome.OK,
            details=ToolCallDetails(
                request_hash="request-hash",
                response_hash="response-hash",
                response_payload={"data": []},
                result_policy=ToolResultPolicy.REFERENCEABLE,
            ),
        ),
        response_payload={"data": []},
        budget=ToolBudgetSnapshot(
            session_budget_usd=1.0,
            session_hard_limit_usd=1.0,
            session_used_budget_usd=0.0,
            session_remaining_budget_usd=1.0,
        ),
    )


async def test_execute_tool_with_concurrency_permit_waits_for_released_same_model_llm_permit() -> None:
    invocation = _invocation(TEST_TOKEN, "llm_chat")
    expected = _result(invocation.session_id, invocation.tool)
    executor = RecordingExecutor(expected)
    limiter = ToolConcurrencyLimiter(ToolConcurrencyLimits(llm_max_parallel_calls=1, search_max_parallel_calls=5))

    limiter.acquire(invocation)
    waiter = asyncio.create_task(execute_tool_with_concurrency_permit(executor, limiter, invocation))
    await asyncio.sleep(0.05)

    assert not waiter.done()
    assert executor.invocations == []

    limiter.release(invocation)
    result = await asyncio.wait_for(waiter, timeout=1.0)

    assert result == expected
    assert executor.invocations == [invocation]
    assert limiter.in_flight(invocation) == 0


async def test_execute_tool_with_concurrency_permit_releases_same_model_llm_permit_after_executor_failure() -> None:
    invocation = _invocation(TEST_TOKEN, "llm_chat")
    limiter = ToolConcurrencyLimiter(ToolConcurrencyLimits(llm_max_parallel_calls=2, search_max_parallel_calls=1))

    with pytest.raises(RuntimeError, match="expected failure"):
        await execute_tool_with_concurrency_permit(FailingExecutor(), limiter, invocation)

    assert limiter.in_flight(invocation) == 0


async def test_llm_and_search_lanes_do_not_block_each_other() -> None:
    llm_invocation = _invocation(TEST_TOKEN, "llm_chat")
    search_invocation = _invocation(TEST_TOKEN, "search_web")
    expected = _result(search_invocation.session_id, search_invocation.tool)
    executor = RecordingExecutor(expected)
    limiter = ToolConcurrencyLimiter(ToolConcurrencyLimits(llm_max_parallel_calls=1, search_max_parallel_calls=1))

    limiter.acquire(llm_invocation)
    try:
        result = await execute_tool_with_concurrency_permit(executor, limiter, search_invocation)
    finally:
        limiter.release(llm_invocation)

    assert result == expected
    assert executor.invocations == [search_invocation]
    assert limiter.in_flight(search_invocation) == 0


def test_search_lane_allows_five_calls_and_blocks_sixth() -> None:
    limiter = ToolConcurrencyLimiter(DEFAULT_TOOL_CONCURRENCY_LIMITS)
    held = [
        _invocation(TEST_TOKEN, "search_web"),
        _invocation(TEST_TOKEN, "search_ai"),
        _invocation(TEST_TOKEN, "fetch_page"),
        _invocation(TEST_TOKEN, "tooling_info"),
        _invocation(TEST_TOKEN, "test_tool"),
    ]
    for invocation in held:
        limiter.acquire(invocation)
    try:
        with pytest.raises(ConcurrencyLimitError):
            limiter.acquire(_invocation(TEST_TOKEN, "search_web"))
    finally:
        for invocation in held:
            limiter.release(invocation)


def test_llm_lane_allows_two_same_model_calls_and_blocks_third() -> None:
    limiter = ToolConcurrencyLimiter(DEFAULT_TOOL_CONCURRENCY_LIMITS)
    held = [_invocation(TEST_TOKEN, "llm_chat"), _invocation(TEST_TOKEN, "llm_chat")]
    for invocation in held:
        limiter.acquire(invocation)
    try:
        with pytest.raises(ConcurrencyLimitError):
            limiter.acquire(_invocation(TEST_TOKEN, "llm_chat"))
    finally:
        for invocation in held:
            limiter.release(invocation)


def test_llm_lane_is_independent_per_model() -> None:
    limiter = ToolConcurrencyLimiter(DEFAULT_TOOL_CONCURRENCY_LIMITS)
    held = [
        _llm_invocation(TEST_TOKEN, DEFAULT_LLM_MODEL),
        _llm_invocation(TEST_TOKEN, DEFAULT_LLM_MODEL),
        _llm_invocation(TEST_TOKEN, OTHER_LLM_MODEL),
        _llm_invocation(TEST_TOKEN, OTHER_LLM_MODEL),
    ]
    for invocation in held:
        limiter.acquire(invocation)
    try:
        with pytest.raises(ConcurrencyLimitError):
            limiter.acquire(_llm_invocation(TEST_TOKEN, DEFAULT_LLM_MODEL))
        with pytest.raises(ConcurrencyLimitError):
            limiter.acquire(_llm_invocation(TEST_TOKEN, OTHER_LLM_MODEL))
    finally:
        for invocation in held:
            limiter.release(invocation)


def test_llm_lane_reads_model_from_first_positional_payload() -> None:
    limiter = ToolConcurrencyLimiter(DEFAULT_TOOL_CONCURRENCY_LIMITS)
    held = [
        _llm_invocation(TEST_TOKEN, DEFAULT_LLM_MODEL, positional_payload=True),
        _llm_invocation(TEST_TOKEN, DEFAULT_LLM_MODEL, positional_payload=True),
    ]
    other_model = _llm_invocation(TEST_TOKEN, OTHER_LLM_MODEL, positional_payload=True)
    for invocation in held:
        limiter.acquire(invocation)
    try:
        limiter.acquire(other_model)
        with pytest.raises(ConcurrencyLimitError):
            limiter.acquire(_llm_invocation(TEST_TOKEN, DEFAULT_LLM_MODEL, positional_payload=True))
    finally:
        limiter.release(other_model)
        for invocation in held:
            limiter.release(invocation)


def test_llm_lane_requires_string_model() -> None:
    limiter = ToolConcurrencyLimiter(DEFAULT_TOOL_CONCURRENCY_LIMITS)
    missing_model = ToolInvocationRequest(
        session_id=uuid4(),
        token=TEST_TOKEN,
        tool="llm_chat",
        args=(),
        kwargs={},
    )
    non_string_model = ToolInvocationRequest(
        session_id=uuid4(),
        token=TEST_TOKEN,
        tool="llm_chat",
        args=(),
        kwargs={"model": 123},
    )

    with pytest.raises(ValueError, match="model must be provided"):
        limiter.acquire(missing_model)
    with pytest.raises(ValueError, match="model must be a string"):
        limiter.acquire(non_string_model)


def test_limiter_rejects_tool_without_lane() -> None:
    limiter = ToolConcurrencyLimiter(DEFAULT_TOOL_CONCURRENCY_LIMITS)
    invocation = _invocation(TEST_TOKEN, cast(ToolName, "unknown_tool"))

    with pytest.raises(ValueError, match="has no concurrency lane"):
        limiter.acquire(invocation)


async def test_default_limits_wait_on_third_same_model_llm_chat_until_release() -> None:
    limiter = ToolConcurrencyLimiter(DEFAULT_TOOL_CONCURRENCY_LIMITS)
    held = [_invocation(TEST_TOKEN, "llm_chat"), _invocation(TEST_TOKEN, "llm_chat")]
    waiter_invocation = _invocation(TEST_TOKEN, "llm_chat")
    expected = _result(waiter_invocation.session_id, waiter_invocation.tool)
    executor = RecordingExecutor(expected)

    for invocation in held:
        limiter.acquire(invocation)
    try:
        waiter = asyncio.create_task(execute_tool_with_concurrency_permit(executor, limiter, waiter_invocation))
        await asyncio.sleep(0.05)
        assert not waiter.done()
        assert executor.invocations == []

        limiter.release(held.pop())
        result = await asyncio.wait_for(waiter, timeout=1.0)

        assert result == expected
        assert executor.invocations == [waiter_invocation]
    finally:
        for invocation in held:
            limiter.release(invocation)

    assert limiter.in_flight(waiter_invocation) == 0


async def test_default_limits_do_not_wait_on_different_llm_chat_model() -> None:
    limiter = ToolConcurrencyLimiter(DEFAULT_TOOL_CONCURRENCY_LIMITS)
    held = [
        _llm_invocation(TEST_TOKEN, DEFAULT_LLM_MODEL),
        _llm_invocation(TEST_TOKEN, DEFAULT_LLM_MODEL),
    ]
    waiter_invocation = _llm_invocation(TEST_TOKEN, OTHER_LLM_MODEL)
    expected = _result(waiter_invocation.session_id, waiter_invocation.tool)
    executor = RecordingExecutor(expected)

    for invocation in held:
        limiter.acquire(invocation)
    try:
        result = await execute_tool_with_concurrency_permit(executor, limiter, waiter_invocation)
        assert result == expected
        assert executor.invocations == [waiter_invocation]
    finally:
        for invocation in held:
            limiter.release(invocation)

    assert limiter.in_flight(waiter_invocation) == 0


async def test_default_limits_wait_on_sixth_non_llm_until_release() -> None:
    limiter = ToolConcurrencyLimiter(DEFAULT_TOOL_CONCURRENCY_LIMITS)
    held = [
        _invocation(TEST_TOKEN, "search_web"),
        _invocation(TEST_TOKEN, "search_ai"),
        _invocation(TEST_TOKEN, "fetch_page"),
        _invocation(TEST_TOKEN, "tooling_info"),
        _invocation(TEST_TOKEN, "test_tool"),
    ]
    waiter_invocation = _invocation(TEST_TOKEN, "search_web")
    expected = _result(waiter_invocation.session_id, waiter_invocation.tool)
    executor = RecordingExecutor(expected)

    for invocation in held:
        limiter.acquire(invocation)
    try:
        waiter = asyncio.create_task(execute_tool_with_concurrency_permit(executor, limiter, waiter_invocation))
        await asyncio.sleep(0.05)
        assert not waiter.done()
        assert executor.invocations == []

        limiter.release(held.pop())
        result = await asyncio.wait_for(waiter, timeout=1.0)

        assert result == expected
        assert executor.invocations == [waiter_invocation]
    finally:
        for invocation in held:
            limiter.release(invocation)

    assert limiter.in_flight(waiter_invocation) == 0
