from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import BaseModel

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm.json_utils import pydantic_postprocessor
from harnyx_commons.llm.provider_factory import build_cached_llm_provider_registry, build_routed_llm_provider
from harnyx_commons.llm.schema import (
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
)

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]
_GEMMA_MODEL = "google/gemma-4-31B-it"
_GEMMA_ROUTE_TARGET = "custom-openai-compatible:gemma4-cloud-run"


class JsonObjectAnswer(BaseModel):
    ping: str
    count: int


class ThrowawayStructuredAnswer(BaseModel):
    animal: str
    count: int
    approved: bool


def test_gemma_live_test_source_has_no_test_only_env_contract() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden_env_names = (
        "GEMMA" + "_CLOUD_RUN_SERVICE_URL",
        "GEMMA" + "_CLOUD_RUN_MODEL",
    )
    assert not any(name in source for name in forbidden_env_names)


async def test_gemma_cloud_run_custom_openai_compatible_live() -> None:
    response = await _invoke_live_gemma(
        LlmRequest(
            provider="chutes",
            model=_GEMMA_MODEL,
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

    assert response.raw_text
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == "custom-openai-compatible:gemma4-cloud-run"
    assert response.metadata["effective_model"] == _GEMMA_MODEL


async def test_gemma_cloud_run_json_object_live() -> None:
    response = await _invoke_live_gemma(
        LlmRequest(
            provider="chutes",
            model=_GEMMA_MODEL,
            messages=(
                LlmMessage(
                    role="user",
                    content=(
                        LlmMessageContentPart.input_text(
                            'Return JSON only with {"ping":"pong","count":2}.'
                        ),
                    ),
                ),
            ),
            temperature=0.0,
            max_output_tokens=128,
            timeout_seconds=180.0,
            output_mode="json_object",
            postprocessor=pydantic_postprocessor(JsonObjectAnswer),
        )
    )

    parsed = response.postprocessed
    assert isinstance(parsed, JsonObjectAnswer)
    assert parsed.ping == "pong"
    assert parsed.count == 2
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == _GEMMA_ROUTE_TARGET
    assert response.metadata["effective_model"] == _GEMMA_MODEL


async def test_gemma_cloud_run_json_schema_live_with_throwaway_schema() -> None:
    response = await _invoke_live_gemma(
        LlmRequest(
            provider="chutes",
            model=_GEMMA_MODEL,
            messages=(
                LlmMessage(
                    role="user",
                    content=(
                        LlmMessageContentPart.input_text(
                            'Return JSON only with {"animal":"otter",'
                            '"count":3,"approved":true}.'
                        ),
                    ),
                ),
            ),
            temperature=0.0,
            max_output_tokens=128,
            timeout_seconds=180.0,
            output_mode="structured",
            output_schema=ThrowawayStructuredAnswer,
            postprocessor=pydantic_postprocessor(ThrowawayStructuredAnswer),
        )
    )

    parsed = response.postprocessed
    assert isinstance(parsed, ThrowawayStructuredAnswer)
    assert parsed.animal == "otter"
    assert parsed.count == 3
    assert parsed.approved is True
    assert response.metadata is not None
    assert response.metadata["effective_provider"] == _GEMMA_ROUTE_TARGET
    assert response.metadata["effective_model"] == _GEMMA_MODEL


def test_gemma_live_settings_use_runtime_env_contract() -> None:
    settings = _build_live_gemma_settings(
        {
            "LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON": json.dumps(
                [
                    {
                        "id": "gemma4-cloud-run",
                        "base_url": "https://gemma.example.run.app/v1",
                        "auth": {
                            "type": "google_id_token",
                            "audience": "https://gemma.example.run.app",
                            "credential_source": "service_account_json_b64_env",
                            "credential_env": "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64",
                        },
                    }
                ]
            ),
            "LLM_MODEL_PROVIDER_OVERRIDES_JSON": json.dumps({"tool": {_GEMMA_MODEL: _GEMMA_ROUTE_TARGET}}),
            "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64": "present",
        }
    )

    assert "gemma4-cloud-run" in settings.openai_compatible_endpoints
    assert settings.llm_model_provider_overrides["tool"][_GEMMA_MODEL] == _GEMMA_ROUTE_TARGET


def test_gemma_live_settings_rejects_wrong_runtime_route() -> None:
    with pytest.raises(RuntimeError, match="must route"):
        _build_live_gemma_settings(
            {
                "LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON": json.dumps(
                    [
                        {
                            "id": "gemma4-cloud-run",
                            "base_url": "https://gemma.example.run.app/v1",
                            "auth": {
                                "type": "google_id_token",
                                "audience": "https://gemma.example.run.app",
                                "credential_source": "service_account_json_b64_env",
                                "credential_env": "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64",
                            },
                        }
                    ]
                ),
                "LLM_MODEL_PROVIDER_OVERRIDES_JSON": json.dumps({"tool": {_GEMMA_MODEL: "chutes"}}),
                "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64": "present",
            }
        )


def _build_live_gemma_settings(environ: Mapping[str, str]) -> LlmSettings:
    _require_mapping_env(environ, "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64")
    settings = LlmSettings(
        LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON=_require_mapping_env(environ, "LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON"),
        LLM_MODEL_PROVIDER_OVERRIDES_JSON=_require_mapping_env(environ, "LLM_MODEL_PROVIDER_OVERRIDES_JSON"),
    )
    configured_route = settings.llm_model_provider_overrides.get("tool", {}).get(_GEMMA_MODEL)
    if configured_route != _GEMMA_ROUTE_TARGET:
        raise RuntimeError(
            "LLM_MODEL_PROVIDER_OVERRIDES_JSON.tool must route "
            f"{_GEMMA_MODEL} to {_GEMMA_ROUTE_TARGET}"
        )
    return settings


async def _invoke_live_gemma(request: LlmRequest) -> LlmResponse:
    settings = _build_live_gemma_settings(os.environ)
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
        return await provider.invoke(request)
    finally:
        await registry.aclose()


def _require_mapping_env(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be configured")
    return value
