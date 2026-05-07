from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from harnyx_commons.config.llm import LlmSettings, parse_openai_compatible_endpoints_json
from harnyx_commons.llm.providers import openai_compatible
from harnyx_commons.llm.providers.openai_compatible import OpenAiCompatibleLlmProvider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest

pytestmark = pytest.mark.anyio("asyncio")


def test_endpoint_config_rejects_duplicate_ids() -> None:
    raw = json.dumps(
        [
            {"id": "duplicate", "base_url": "https://example.com/v1", "auth": {"type": "none"}},
            {"id": "duplicate", "base_url": "https://example.org/v1", "auth": {"type": "none"}},
        ]
    )

    with pytest.raises(ValueError, match="duplicated"):
        parse_openai_compatible_endpoints_json(raw)


def test_endpoint_config_rejects_raw_bearer_token_field() -> None:
    raw = json.dumps(
        [
            {
                "id": "local",
                "base_url": "https://example.com/v1",
                "auth": {"type": "bearer_token_env", "token_env": "LOCAL_TOKEN", "token": "secret"},
            }
        ]
    )

    with pytest.raises(ValueError, match="extra_forbidden"):
        parse_openai_compatible_endpoints_json(raw)


async def test_bearer_token_env_auth_adds_authorization_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_OPENAI_COMPATIBLE_TOKEN", "test-token")
    endpoint = _endpoint(
        auth={"type": "bearer_token_env", "token_env": "LOCAL_OPENAI_COMPATIBLE_TOKEN"},
    )
    seen_headers: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_headers["authorization"] = request.headers["authorization"]
        return _streaming_response()

    provider = OpenAiCompatibleLlmProvider(
        endpoint=endpoint,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        response = await provider.invoke(_request())
    finally:
        await provider.aclose()

    assert seen_headers["authorization"] == "Bearer test-token"
    assert response.raw_text == "ok"
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 2


async def test_google_id_token_service_account_b64_auth_refreshes_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_account_info = {
        "type": "service_account",
        "client_email": "test@example.iam.gserviceaccount.com",
        "private_key": "unused",
    }
    encoded = base64.b64encode(json.dumps(service_account_info).encode("utf-8")).decode("ascii")
    monkeypatch.setenv("GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64", encoded)
    captured: dict[str, object] = {"credential_count": 0, "refresh_count": 0}

    class _FakeCredentials:
        token: str | None = None

        def refresh(self, request: object) -> None:
            captured["refresh_request_type"] = type(request).__name__
            captured["refresh_count"] = int(captured["refresh_count"]) + 1
            self.token = f"google-id-token-{captured['refresh_count']}"

    def fake_from_service_account_info(info: dict[str, object], *, target_audience: str) -> _FakeCredentials:
        captured["credential_count"] = int(captured["credential_count"]) + 1
        captured["info"] = info
        captured["target_audience"] = target_audience
        return _FakeCredentials()

    monkeypatch.setattr(
        openai_compatible.service_account,
        "IDTokenCredentials",
        SimpleNamespace(from_service_account_info=fake_from_service_account_info),
    )
    endpoint = _endpoint(
        auth={
            "type": "google_id_token",
            "audience": "https://gemma.example.run.app",
            "credential_source": "service_account_json_b64_env",
            "credential_env": "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64",
        }
    )
    seen_headers: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers["authorization"])
        return _streaming_response()

    provider = OpenAiCompatibleLlmProvider(
        endpoint=endpoint,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        response = await provider.invoke(_request())
        second_response = await provider.invoke(_request())
    finally:
        await provider.aclose()

    assert seen_headers == ["Bearer google-id-token-1", "Bearer google-id-token-2"]
    assert captured["credential_count"] == 2
    assert captured["refresh_count"] == 2
    assert captured["info"] == service_account_info
    assert captured["target_audience"] == "https://gemma.example.run.app"
    assert response.raw_text == "ok"
    assert second_response.raw_text == "ok"


async def test_openai_compatible_provider_normalizes_streamed_chat_response() -> None:
    provider = OpenAiCompatibleLlmProvider(
        endpoint=_endpoint(auth={"type": "none"}),
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: _streaming_response())),
    )

    try:
        response = await provider.invoke(_request())
    finally:
        await provider.aclose()

    assert response.id == "chatcmpl-1"
    assert response.raw_text == "ok"
    assert response.finish_reason == "stop"
    assert response.usage.total_tokens == 5
    assert response.metadata is not None
    assert response.metadata["raw_response"]["id"] == "chatcmpl-1"


def _endpoint(*, auth: dict[str, object]):
    return LlmSettings(
        LLM_OPENAI_COMPATIBLE_ENDPOINTS_JSON=json.dumps(
            [{"id": "local", "base_url": "https://example.com/v1", "auth": auth}]
        )
    ).openai_compatible_endpoints["local"]


def _request() -> LlmRequest:
    return LlmRequest(
        provider="custom-openai-compatible:local",
        model="nvidia/Gemma-4-31B-IT-NVFP4",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=8,
    )


def _streaming_response() -> httpx.Response:
    payload = "\n\n".join(
        (
            'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"ok"}}]}',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
            "data: [DONE]",
            "",
        )
    )
    return httpx.Response(200, content=payload.encode("utf-8"))
