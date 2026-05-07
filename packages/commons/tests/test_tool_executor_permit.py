from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from harnyx_commons.domain.tool_call import ToolCall, ToolCallDetails, ToolCallOutcome, ToolResultPolicy
from harnyx_commons.tools.dto import ToolBudgetSnapshot, ToolInvocationRequest, ToolInvocationResult
from harnyx_commons.tools.executor import execute_tool_with_token_permit
from harnyx_commons.tools.token_semaphore import TokenSemaphore

pytestmark = pytest.mark.anyio("asyncio")
TEST_TOKEN = "token"  # noqa: S105


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


def _invocation(token: str) -> ToolInvocationRequest:
    return ToolInvocationRequest(
        session_id=uuid4(),
        token=token,
        tool="search_web",
        args=(),
        kwargs={"query": "demo"},
    )


def _result(session_id) -> ToolInvocationResult:
    return ToolInvocationResult(
        receipt=ToolCall(
            receipt_id="receipt-1",
            session_id=session_id,
            uid=7,
            tool="search_web",
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


async def test_execute_tool_with_token_permit_waits_for_released_permit() -> None:
    invocation = _invocation(TEST_TOKEN)
    expected = _result(invocation.session_id)
    executor = RecordingExecutor(expected)
    semaphore = TokenSemaphore(max_parallel_calls=1)

    semaphore.acquire(invocation.token)
    waiter = asyncio.create_task(execute_tool_with_token_permit(executor, semaphore, invocation))
    await asyncio.sleep(0.05)

    assert not waiter.done()
    assert executor.invocations == []

    semaphore.release(invocation.token)
    result = await asyncio.wait_for(waiter, timeout=1.0)

    assert result == expected
    assert executor.invocations == [invocation]
    assert semaphore.in_flight(invocation.token) == 0


async def test_execute_tool_with_token_permit_releases_after_executor_failure() -> None:
    invocation = _invocation(TEST_TOKEN)
    semaphore = TokenSemaphore(max_parallel_calls=1)

    with pytest.raises(RuntimeError, match="expected failure"):
        await execute_tool_with_token_permit(FailingExecutor(), semaphore, invocation)

    assert semaphore.in_flight(invocation.token) == 0
