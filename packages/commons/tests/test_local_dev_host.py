from __future__ import annotations

import pytest

from harnyx_commons.tools import local_dev_host

pytestmark = pytest.mark.anyio("asyncio")


class _StubSettings:
    search_provider = "desearch"
    desearch_api_key_value = "desearch-key"
    parallel_api_key_value = "parallel-key"
    parallel_base_url = "https://parallel.local.test"
    chutes_api_key_value = "chutes-key"
    desearch_max_concurrent = 1
    parallel_max_concurrent = 1
    chutes_max_concurrent = 1


class _StubSearchClient:
    def __init__(self, **_: object) -> None:
        pass

    async def aclose(self) -> None:
        return None


class _StubLlmProvider:
    def __init__(self, **_: object) -> None:
        pass

    async def aclose(self) -> None:
        return None


async def test_create_local_tool_host_tooling_info_matches_miner_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_dev_host, "LlmSettings", lambda: _StubSettings())
    monkeypatch.setattr(local_dev_host, "DeSearchClient", _StubSearchClient)
    monkeypatch.setattr(local_dev_host, "ChutesLlmProvider", _StubLlmProvider)

    host = local_dev_host.create_local_tool_host()
    try:
        payload = await host.invoke("tooling_info")
    finally:
        await host.aclose()

    assert "search_repo" not in payload["response"]["tool_names"]
    assert "get_repo_file" not in payload["response"]["tool_names"]
    assert "search_items" not in payload["response"]["tool_names"]
    assert "search_repo" not in payload["response"]["pricing"]
    assert "get_repo_file" not in payload["response"]["pricing"]
    assert "search_items" not in payload["response"]["pricing"]


def test_build_local_search_client_uses_configured_parallel_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _ParallelStub:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(local_dev_host, "ParallelClient", _ParallelStub)

    settings = _StubSettings()
    settings.search_provider = "parallel"
    settings.parallel_base_url = "https://proxy.parallel.test"
    settings.parallel_api_key_value = "parallel-key"
    settings.parallel_max_concurrent = 7

    local_dev_host._build_local_search_client(settings)

    assert captured["base_url"] == "https://proxy.parallel.test"
    assert captured["api_key"] == "parallel-key"
    assert captured["max_concurrent"] == 7
