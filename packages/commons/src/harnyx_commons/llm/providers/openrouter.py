"""Hardcoded OpenRouter provider for repo-owned OpenRouter routes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import httpx
from pydantic import SecretStr

from harnyx_commons.config.llm import OpenAiCompatibleEndpointConfig, OpenRouterModelProviderOptions
from harnyx_commons.llm.provider import LlmProviderPort
from harnyx_commons.llm.provider_types import OPENROUTER_PROVIDER
from harnyx_commons.llm.providers.openai_compatible import OpenAiCompatibleLlmProvider
from harnyx_commons.llm.schema import AbstractLlmRequest, LlmResponse

OPENROUTER_ENDPOINT_ID = "openrouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_ONLY_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")


OpenRouterChatProviderFactory = Callable[[str], tuple[OpenAiCompatibleLlmProvider, httpx.AsyncClient]]


class OpenRouterLlmProvider(LlmProviderPort):
    def __init__(
        self,
        *,
        openrouter_api_key: SecretStr,
        model_provider_options: Mapping[str, OpenRouterModelProviderOptions],
        openrouter_chat_provider_factory: OpenRouterChatProviderFactory | None = None,
    ) -> None:
        unknown_models = set(model_provider_options) - set(OPENROUTER_ONLY_MODELS)
        if unknown_models:
            unknown = ", ".join(sorted(unknown_models))
            raise ValueError(f"OpenRouter provider options configured for unsupported models: {unknown}")
        self._openrouter_api_key = openrouter_api_key
        self._model_provider_options = dict(model_provider_options)
        self._openrouter_chat_provider_factory = openrouter_chat_provider_factory or build_openrouter_chat_provider
        self._openrouter_provider: OpenAiCompatibleLlmProvider | None = None
        self._openrouter_client: httpx.AsyncClient | None = None

    async def invoke(self, request: AbstractLlmRequest) -> LlmResponse:
        model = request.model.strip()
        if model not in OPENROUTER_ONLY_MODELS:
            raise ValueError(f"OpenRouter provider does not support model {request.model!r}")
        openrouter_provider = self._ensure_openrouter_provider(model=model)
        response = await openrouter_provider.invoke(self._openrouter_request(request, model=model))
        metadata = dict(response.metadata or {})
        metadata["effective_provider"] = OPENROUTER_PROVIDER
        metadata["effective_model"] = model
        return replace(response, metadata=metadata)

    async def aclose(self) -> None:
        errors: list[Exception] = []
        if self._openrouter_provider is not None:
            try:
                await self._openrouter_provider.aclose()
            except Exception as exc:
                exc.add_note("OpenRouter OpenAI-compatible delegate close failed")
                errors.append(exc)
        if self._openrouter_client is not None:
            try:
                await self._openrouter_client.aclose()
            except Exception as exc:
                exc.add_note("OpenRouter HTTP client close failed")
                errors.append(exc)
        if errors:
            raise ExceptionGroup("OpenRouter provider cleanup failed", errors)

    def _ensure_openrouter_provider(self, *, model: str) -> OpenAiCompatibleLlmProvider:
        if self._openrouter_provider is not None:
            return self._openrouter_provider
        normalized_key = self._openrouter_api_key.get_secret_value().strip()
        if not normalized_key:
            raise ValueError(f"OPENROUTER_API_KEY must be configured to use OpenRouter model {model}")
        provider, client = self._openrouter_chat_provider_factory(normalized_key)
        self._openrouter_provider = provider
        self._openrouter_client = client
        return provider

    def _openrouter_request(self, request: AbstractLlmRequest, *, model: str) -> AbstractLlmRequest:
        extra = _merge_extra(request.extra, self._model_provider_options.get(model))
        return replace(request, provider=OPENROUTER_PROVIDER, extra=extra or None)


def build_openrouter_chat_provider(api_key: str) -> tuple[OpenAiCompatibleLlmProvider, httpx.AsyncClient]:
    normalized_key = api_key.strip()
    if not normalized_key:
        raise ValueError("OPENROUTER_API_KEY must be configured to build OpenRouter provider")
    client = httpx.AsyncClient(
        base_url=OPENROUTER_BASE_URL,
        headers={"Authorization": f"Bearer {normalized_key}"},
    )
    endpoint = OpenAiCompatibleEndpointConfig.model_validate(
        {
            "id": OPENROUTER_ENDPOINT_ID,
            "base_url": OPENROUTER_BASE_URL,
            "auth": {"type": "none"},
        }
    )
    return OpenAiCompatibleLlmProvider(endpoint=endpoint, client=client), client


def _merge_extra(
    request_extra: Mapping[str, Any] | None,
    options: OpenRouterModelProviderOptions | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(request_extra or {})
    if options is None:
        return merged
    options_extra = dict(options.to_request_extra())
    provider_options = options_extra.get("provider")
    if provider_options is None:
        return merged
    if not isinstance(provider_options, Mapping):
        raise AssertionError("OpenRouter provider options must serialize provider as an object")
    request_provider = merged.get("provider")
    if request_provider is not None and not isinstance(request_provider, Mapping):
        raise ValueError("OpenRouter request extra.provider must be an object")
    merged_provider = dict(request_provider or {})
    merged_provider.update(dict(provider_options))
    merged["provider"] = merged_provider
    return merged


__all__ = [
    "OPENROUTER_BASE_URL",
    "OPENROUTER_ENDPOINT_ID",
    "OPENROUTER_ONLY_MODELS",
    "OpenRouterLlmProvider",
    "build_openrouter_chat_provider",
]
