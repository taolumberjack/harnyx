"""Local in-process tool host used by miner development utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from harnyx_commons.application.dto.session import SessionTokenRequest
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.clients import CHUTES, DESEARCH, PARALLEL
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.domain.miner_task import DEFAULT_MINER_TASK_BUDGET_USD
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.llm.providers.chutes import ChutesLlmProvider
from harnyx_commons.tools.desearch import DeSearchClient
from harnyx_commons.tools.dto import ToolInvocationRequest
from harnyx_commons.tools.executor import ToolExecutor
from harnyx_commons.tools.http_serialization import serialize_tool_execute_response
from harnyx_commons.tools.parallel import ParallelClient
from harnyx_commons.tools.ports import WebSearchProviderPort
from harnyx_commons.tools.runtime_invoker import ALLOWED_TOOL_MODELS, build_miner_sandbox_tool_invoker
from harnyx_commons.tools.types import parse_tool_name
from harnyx_commons.tools.usage_tracker import UsageTracker


@dataclass(frozen=True, slots=True)
class LocalToolHost:
    """In-process tool host that matches the miner SDK tool invoker protocol."""

    session_id: UUID
    token: str
    _tool_executor: ToolExecutor
    _search_client: WebSearchProviderPort
    _llm_provider: ChutesLlmProvider

    async def invoke(
        self,
        method: str,
        *,
        args: Sequence[Any] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        tool = parse_tool_name(method)
        result = await self._tool_executor.execute(
            ToolInvocationRequest(
                session_id=self.session_id,
                token=self.token,
                tool=tool,
                args=tuple(args or ()),
                kwargs=dict(kwargs or {}),
            )
        )
        return serialize_tool_execute_response(result).model_dump(mode="json")

    async def aclose(self) -> None:
        await self._search_client.aclose()
        await self._llm_provider.aclose()


def create_local_tool_host(*, uid: int = 1, session_ttl_minutes: int = 30) -> LocalToolHost:
    llm_settings = LlmSettings()
    if llm_settings.search_provider == "parallel":
        if not llm_settings.parallel_api_key_value:
            raise RuntimeError("PARALLEL_API_KEY must be set to run local tool host")
    elif not llm_settings.desearch_api_key_value:
        raise RuntimeError("DESEARCH_API_KEY must be set to run local tool host")
    if not llm_settings.chutes_api_key_value:
        raise RuntimeError("CHUTES_API_KEY must be set to run local tool host")

    usage_tracker = UsageTracker()

    sessions = InMemorySessionRegistry()
    tokens = InMemoryTokenRegistry()
    receipts = InMemoryReceiptLog()
    session_manager = SessionManager(sessions, tokens)

    session_id = uuid4()
    token = uuid4().hex
    now = datetime.now(UTC)
    session_manager.issue(
        SessionTokenRequest(
            session_id=session_id,
            uid=uid,
            task_id=uuid4(),
            issued_at=now,
            expires_at=now + timedelta(minutes=session_ttl_minutes),
            budget_usd=DEFAULT_MINER_TASK_BUDGET_USD,
            token=token,
        )
    )

    search_client = _build_local_search_client(llm_settings)
    llm_provider = ChutesLlmProvider(
        base_url=CHUTES.base_url,
        api_key=llm_settings.chutes_api_key_value,
        timeout=CHUTES.timeout_seconds,
        max_concurrent=llm_settings.chutes_max_concurrent,
    )
    tool_invoker = build_miner_sandbox_tool_invoker(
        receipts,
        web_search_client=search_client,
        llm_provider=llm_provider,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )
    tool_executor = ToolExecutor(
        sessions,
        receipts,
        usage_tracker,
        tool_invoker,
        token_registry=tokens,
        clock=lambda: datetime.now(UTC),
    )

    return LocalToolHost(
        session_id=session_id,
        token=token,
        _tool_executor=tool_executor,
        _search_client=search_client,
        _llm_provider=llm_provider,
    )


def _build_local_search_client(settings: LlmSettings) -> WebSearchProviderPort:
    if settings.search_provider == "parallel":
        return ParallelClient(
            base_url=settings.parallel_base_url,
            api_key=settings.parallel_api_key_value,
            timeout=PARALLEL.timeout_seconds,
            max_concurrent=settings.parallel_max_concurrent,
        )
    return DeSearchClient(
        base_url=DESEARCH.base_url,
        api_key=settings.desearch_api_key_value,
        timeout=DESEARCH.timeout_seconds,
        max_concurrent=settings.desearch_max_concurrent,
    )


__all__ = ["LocalToolHost", "create_local_tool_host"]
