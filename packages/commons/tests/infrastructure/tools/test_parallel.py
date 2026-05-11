from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from harnyx_commons.errors import ToolProviderError
from harnyx_commons.tools.parallel import ParallelClient
from harnyx_commons.tools.search_models import FetchPageRequest, SearchAiSearchRequest, SearchWebSearchRequest

pytestmark = pytest.mark.anyio("asyncio")


async def test_parallel_client_search_web_posts_keyword_list() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "search_id": "search-1",
                "results": [
                    {
                        "url": "https://example.com/a",
                        "title": "Alpha",
                        "excerpts": ["alpha snippet"],
                        "publish_date": "2026-03-24T00:00:00Z",
                    }
                ],
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = ParallelClient(base_url="https://api.parallel.ai", api_key="parallel-key", client=client)

    response = await adapter.search_web(SearchWebSearchRequest(search_queries=("alpha", "beta"), num=3))

    assert response.data[0].link == "https://example.com/a"
    assert response.data[0].snippet == "alpha snippet"
    assert response.attempts == 1
    assert response.retry_reasons == ()
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.parallel.ai/v1beta/search"
    assert captured["headers"]["x-api-key"] == "parallel-key"
    assert captured["json"] == {
        "search_queries": ["alpha", "beta"],
        "max_results": 3,
    }


async def test_parallel_client_search_ai_uses_objective() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "search_id": "search-2",
                "results": [
                    {
                        "url": "https://example.com/b",
                        "title": "Beta",
                        "excerpts": ["beta summary"],
                    }
                ],
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = ParallelClient(base_url="https://api.parallel.ai", api_key="parallel-key", client=client)

    response = await adapter.search_ai(SearchAiSearchRequest(prompt="find beta", count=10))

    assert response.data[0].url == "https://example.com/b"
    assert response.data[0].note == "beta summary"
    assert captured["json"] == {
        "objective": "find beta",
        "max_results": 10,
    }


async def test_parallel_client_fetch_page_uses_extract() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "extract_id": "extract-1",
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "Example",
                        "full_content": "full page text",
                    }
                ],
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = ParallelClient(base_url="https://api.parallel.ai", api_key="parallel-key", client=client)

    response = await adapter.fetch_page(FetchPageRequest(url="https://example.com"))

    assert response.data[0].url == "https://example.com"
    assert response.data[0].content == "full page text"
    assert response.attempts == 1
    assert response.retry_reasons == ()
    assert captured["json"] == {
        "urls": ["https://example.com"],
        "full_content": True,
        "excerpts": False,
    }


async def test_parallel_client_raises_on_error_status() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "failure"})

    client = httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = ParallelClient(base_url="https://api.parallel.ai", api_key="parallel-key", client=client)

    with pytest.raises(ToolProviderError):
        await adapter.fetch_page(FetchPageRequest(url="https://example.com"))


async def test_parallel_client_fetch_page_raises_on_empty_extract_results() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"extract_id": "extract-1", "results": []})

    client = httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = ParallelClient(base_url="https://api.parallel.ai", api_key="parallel-key", client=client)

    with pytest.raises(ToolProviderError):
        await adapter.fetch_page(FetchPageRequest(url="https://example.com"))
