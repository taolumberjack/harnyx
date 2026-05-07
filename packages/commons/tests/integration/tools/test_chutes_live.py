from __future__ import annotations

import pytest

from harnyx_commons.clients import CHUTES
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.providers.chutes import ChutesLlmProvider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]

DEEPSEEK_TOOL_MODELS = (
    "deepseek-ai/DeepSeek-V3.1-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
)
CHUTES_TOOL_MODELS = DEEPSEEK_TOOL_MODELS + ("zai-org/GLM-5-TEE", "google/gemma-4-31B-turbo-TEE")


def _provider_settings() -> tuple[str, float]:
    settings = LlmSettings()
    api_key = settings.chutes_api_key_value
    assert api_key, "CHUTES_API_KEY must be configured"
    timeout = float(CHUTES.timeout_seconds)
    return api_key, timeout


@pytest.mark.parametrize("model", CHUTES_TOOL_MODELS)
async def test_chutes_tool_model_completion_live(model: str) -> None:
    api_key, timeout = _provider_settings()
    provider = ChutesLlmProvider(
        base_url=CHUTES.base_url,
        api_key=api_key,
        timeout=timeout,
    )
    request = LlmRequest(
        provider="chutes",
        model=model,
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=32,
        timeout_seconds=180.0,
    )

    try:
        response = await provider.invoke(request)
    finally:
        await provider.aclose()

    assert response.raw_text
