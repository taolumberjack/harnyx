from __future__ import annotations

import json
import os

import pytest

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm.provider_factory import build_cached_llm_provider_registry, build_routed_llm_provider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_gemma_cloud_run_custom_openai_compatible_live() -> None:
    service_url = _require_env("GEMMA_CLOUD_RUN_SERVICE_URL").rstrip("/")
    _require_env("GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64")
    model = os.environ.get("GEMMA_CLOUD_RUN_MODEL", "google/gemma-4-31B-it").strip()
    if not model:
        raise RuntimeError("GEMMA_CLOUD_RUN_MODEL must be non-empty when configured")
    settings = LlmSettings(
        LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON=json.dumps(
            [
                {
                    "id": "gemma4-cloud-run",
                    "base_url": f"{service_url}/v1",
                    "auth": {
                        "type": "google_id_token",
                        "audience": service_url,
                        "credential_source": "service_account_json_b64_env",
                        "credential_env": "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64",
                    },
                    "timeout_seconds": 180.0,
                }
            ]
        ),
        LLM_MODEL_PROVIDER_OVERRIDES_JSON=json.dumps(
            {"tool": {model: "custom-openai-compatible:gemma4-cloud-run"}}
        ),
    )
    registry = build_cached_llm_provider_registry(
        llm_settings=settings,
        bedrock_settings=BedrockSettings.model_construct(region="us-east-1"),
        vertex_settings=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64="",
        ),
    )
    provider = build_routed_llm_provider(
        surface="tool",
        default_provider="chutes",
        llm_settings=settings,
        allowed_providers={"chutes", "vertex"},
        allow_custom_openai_compatible=True,
        provider_registry=registry,
    )

    try:
        response = await provider.invoke(
            LlmRequest(
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
        )
    finally:
        await registry.aclose()

    assert response.raw_text
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == "custom-openai-compatible:gemma4-cloud-run"
    assert response.metadata["effective_model"] == model


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value
