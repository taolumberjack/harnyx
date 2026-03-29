from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from harnyx_commons.tools.desearch import DeSearchAiDateFilter, DeSearchClient
from harnyx_commons.tools.search_models import (
    FetchPageRequest,
    SearchAiSearchRequest,
    SearchWebSearchRequest,
    SearchXSearchRequest,
)

pytestmark = pytest.mark.anyio("asyncio")


def _capture_request() -> tuple[dict[str, Any], httpx.MockTransport]:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["params"] = request.url.params
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    return captured, transport


async def test_desearch_client_posts_payload() -> None:
    captured, transport = _capture_request()
    client = httpx.AsyncClient(base_url="https://api.desearch.ai", transport=transport)

    adapter = DeSearchClient(
        base_url="https://api.desearch.ai",
        api_key="test-key",
        client=client,
    )

    request = SearchWebSearchRequest(search_queries=("harnyx", "subnet"), num=5)
    result = await adapter.search_links_web(request)

    assert result.data == []
    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.desearch.ai/web?query=%28harnyx%29+OR+%28subnet%29&num=5"
    assert captured["headers"]["authorization"] == "test-key"


async def test_desearch_client_preserves_single_search_term() -> None:
    captured, transport = _capture_request()
    client = httpx.AsyncClient(base_url="https://api.desearch.ai", transport=transport)

    adapter = DeSearchClient(
        base_url="https://api.desearch.ai",
        api_key="test-key",
        client=client,
    )

    request = SearchWebSearchRequest(search_queries=("United States",), num=5)
    result = await adapter.search_links_web(request)

    assert result.data == []
    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.desearch.ai/web?query=United+States&num=5"
    assert captured["headers"]["authorization"] == "test-key"


async def test_desearch_client_raises_on_error_status() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "failure"})

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="test-key", client=client)

    with pytest.raises(RuntimeError):
        await adapter.search_links_web(SearchWebSearchRequest(search_queries=("harnyx", "subnet")))


async def test_desearch_client_twitter_search() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["params"] = request.url.params
        return httpx.Response(200, json=[{"text": "hello", "user": {"username": "foo"}}])

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    response = await adapter.search_links_twitter(SearchXSearchRequest(query="#harnyx", count=3))

    assert response.data[0].text == "hello"
    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.desearch.ai/twitter?query=%23harnyx&count=3"


async def test_desearch_client_ai_search_twitter_posts_posts_payload() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "key"

        if request.url.path == "/desearch/ai/search":
            assert request.method == "POST"
            payload = json.loads(request.content)
            assert payload["prompt"] == "harnyx subnet"
            assert payload["tools"] == ["twitter"]
            assert payload["result_type"] == "LINKS_WITH_FINAL_SUMMARY"
            assert payload["system_message"] == ""
            assert payload["streaming"] is False
            assert payload["count"] == 200
            assert payload["date_filter"] == "PAST_24_HOURS"
            assert "start_date" not in payload
            assert "end_date" not in payload
            assert "model" not in payload

            return httpx.Response(
                200,
                json={
                    "tweets": [
                        {
                            "id": "123",
                            "url": "https://x.com/foo/status/123",
                            "text": "hi",
                            "user": {"username": "foo"},
                        }
                    ],
                    "completion": "hello",
                },
            )

        if request.url.path == "/twitter/post":
            raise AssertionError("ai_search_twitter_posts should not call /twitter/post when tweets are present")

        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    response = await adapter.ai_search_twitter_posts(
        prompt="harnyx subnet",
        count=300,
        date_filter=DeSearchAiDateFilter.PAST_24_HOURS,
    )

    assert response.tweets and len(response.tweets) == 1
    assert response.tweets[0].id == "123"
    assert response.completion == "hello"


async def test_desearch_client_search_ai_preserves_retry_metadata() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/desearch/ai/search":
            raise AssertionError(f"unexpected request: {request.method} {request.url}")
        payload = json.loads(request.content)
        assert payload["prompt"] == "harnyx subnet"
        assert payload["tools"] == ["web", "hackernews", "reddit", "wikipedia", "youtube", "arxiv"]
        assert payload["result_type"] == "LINKS_WITH_FINAL_SUMMARY"
        assert payload["system_message"] == ""
        assert payload["count"] == 3
        return httpx.Response(
            200,
            json={
                "search": [
                    {
                        "link": "https://example.com",
                        "title": "Example",
                        "snippet": "Snippet",
                    }
                ]
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    response = await adapter.search_ai(SearchAiSearchRequest(prompt="harnyx subnet", count=3))

    assert [item.model_dump(exclude_none=True) for item in response.data] == [
        {
            "url": "https://example.com",
            "title": "Example",
            "note": "Snippet",
        }
    ]
    assert response.attempts == 1
    assert response.retry_reasons == ()


async def test_desearch_client_search_ai_accepts_summary_and_results_shape() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/desearch/ai/search":
            raise AssertionError(f"unexpected request: {request.method} {request.url}")
        return httpx.Response(
            200,
            json={
                "summary": "Hamlet is a tragedy.",
                "results": [
                    {
                        "url": "https://example.com/hamlet",
                        "title": "Hamlet",
                        "snippet": "Plot summary",
                    }
                ],
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    response = await adapter.search_ai(SearchAiSearchRequest(prompt="hamlet", count=3))

    assert [item.model_dump(exclude_none=True) for item in response.data] == [
        {
            "url": "https://example.com/hamlet",
            "title": "Hamlet",
            "note": "Plot summary",
        }
    ]


async def test_desearch_client_search_ai_accepts_sdk_search_results_shape() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/desearch/ai/search":
            raise AssertionError(f"unexpected request: {request.method} {request.url}")
        return httpx.Response(
            200,
            json={
                "youtube_search_results": {
                    "organic_results": [
                        {
                            "link": "https://example.com/video",
                            "title": "Hamlet video",
                            "summary_description": "Video summary",
                        }
                    ]
                }
            },
        )

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    response = await adapter.search_ai(SearchAiSearchRequest(prompt="hamlet", count=3))

    assert [item.model_dump(exclude_none=True) for item in response.data] == [
        {
            "url": "https://example.com/video",
            "title": "Hamlet video",
            "note": "Video summary",
        }
    ]


async def test_desearch_client_search_ai_summary_only_shape_returns_empty_results() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/desearch/ai/search":
            raise AssertionError(f"unexpected request: {request.method} {request.url}")
        return httpx.Response(
            200,
            json={"summary": "Hamlet is a tragedy by Shakespeare."},
        )

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    response = await adapter.search_ai(SearchAiSearchRequest(prompt="hamlet", count=3))

    assert response.data == []
    assert response.attempts == 1
    assert response.retry_reasons == ()


async def test_desearch_client_fetch_page_text() -> None:
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        return httpx.Response(200, text="example page content")

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    response = await adapter.fetch_page(FetchPageRequest(url="https://example.com"))

    assert response.data[0].url == "https://example.com"
    assert response.data[0].content == "example page content"
    assert response.attempts == 1
    assert response.retry_reasons == ()
    assert captured["method"] == "GET"
    assert captured["url"] == "https://api.desearch.ai/web/crawl?url=https%3A%2F%2Fexample.com&format=text"


async def test_desearch_client_fetch_page_raises_on_error_status() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "failure"})

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    with pytest.raises(RuntimeError):
        await adapter.fetch_page(FetchPageRequest(url="https://example.com"))


async def test_desearch_client_fetch_page_rejects_blank_text() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="   \n  ")

    client = httpx.AsyncClient(
        base_url="https://api.desearch.ai",
        transport=httpx.MockTransport(handler),
    )
    adapter = DeSearchClient(base_url="https://api.desearch.ai", api_key="key", client=client)

    with pytest.raises(ValueError):
        await adapter.fetch_page(FetchPageRequest(url="https://example.com"))
