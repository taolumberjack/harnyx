from __future__ import annotations

import pytest

from harnyx_commons.clients import DESEARCH
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.tools.desearch import DeSearchAiDateFilter, DeSearchClient
from harnyx_commons.tools.desearch_ai_protocol import DeSearchAiDocsResponse
from harnyx_commons.tools.search_models import SearchAiSearchRequest

pytestmark = [pytest.mark.anyio("asyncio"), pytest.mark.integration]


async def test_desearch_ai_search_live() -> None:
    settings = LlmSettings()
    assert settings.desearch_api_key_value, "DESEARCH_API_KEY must be set"

    desearch = DeSearchClient(
        base_url=DESEARCH.base_url,
        api_key=settings.desearch_api_key_value,
        timeout=settings.llm_timeout_seconds,
        max_concurrent=1,
    )
    try:
        response = await desearch.ai_search_twitter_posts(
            prompt="Bittensor",
            count=10,
            date_filter=DeSearchAiDateFilter.PAST_WEEK,
        )
        assert isinstance(response, DeSearchAiDocsResponse)
    finally:
        await desearch.aclose()


async def test_desearch_search_ai_live() -> None:
    settings = LlmSettings()
    assert settings.desearch_api_key_value, "DESEARCH_API_KEY must be set"

    desearch = DeSearchClient(
        base_url=DESEARCH.base_url,
        api_key=settings.desearch_api_key_value,
        timeout=settings.llm_timeout_seconds,
        max_concurrent=1,
    )
    try:
        response = await desearch.search_ai(
            SearchAiSearchRequest(
                prompt="Find the official Python documentation homepage",
                count=8,
            )
        )
        assert isinstance(response.data, list)
    finally:
        await desearch.aclose()
