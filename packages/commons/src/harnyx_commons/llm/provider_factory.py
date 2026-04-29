"""Shared cached LLM provider construction for platform and validator."""

from __future__ import annotations

from collections.abc import Callable

from harnyx_commons.clients import CHUTES
from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.llm.adapter import LlmProviderAdapter
from harnyx_commons.llm.provider import LlmProviderName, LlmProviderPort, parse_provider_name
from harnyx_commons.llm.providers.bedrock import BedrockLlmProvider
from harnyx_commons.llm.providers.chutes import ChutesLlmProvider
from harnyx_commons.llm.providers.vertex.provider import VertexLlmProvider
from harnyx_commons.llm.routing import LlmRouteSurface, RoutedLlmProvider


class CachedLlmProviderRegistry:
    def __init__(
        self,
        *,
        llm_settings: LlmSettings,
        bedrock_settings: BedrockSettings,
        vertex_settings: VertexSettings,
    ) -> None:
        self._llm_settings = llm_settings
        self._bedrock_settings = bedrock_settings
        self._vertex_settings = vertex_settings
        self._cache: dict[LlmProviderName, LlmProviderPort] = {}

    def resolve(self, name: str) -> LlmProviderPort:
        provider_name = parse_provider_name(name, component="shared")
        provider = self._cache.get(provider_name)
        if provider is None:
            provider = _build_provider(
                provider_name=provider_name,
                llm_settings=self._llm_settings,
                bedrock_settings=self._bedrock_settings,
                vertex_settings=self._vertex_settings,
            )
            self._cache[provider_name] = provider
        return provider

    async def aclose(self) -> None:
        errors: list[Exception] = []
        for provider_name, provider in self._cache.items():
            try:
                await provider.aclose()
            except Exception as exc:
                exc.add_note(f"cached llm provider close failed: {provider_name}")
                errors.append(exc)
        if errors:
            raise ExceptionGroup("cached llm provider cleanup failed", errors)


def build_cached_llm_provider_registry(
    *,
    llm_settings: LlmSettings,
    bedrock_settings: BedrockSettings,
    vertex_settings: VertexSettings,
) -> CachedLlmProviderRegistry:
    """Return a cached provider registry that applies shared concurrency settings."""

    return CachedLlmProviderRegistry(
        llm_settings=llm_settings,
        bedrock_settings=bedrock_settings,
        vertex_settings=vertex_settings,
    )


def build_cached_llm_provider_resolver(
    *,
    llm_settings: LlmSettings,
    bedrock_settings: BedrockSettings,
    vertex_settings: VertexSettings,
) -> Callable[[str], LlmProviderPort]:
    registry = build_cached_llm_provider_registry(
        llm_settings=llm_settings,
        bedrock_settings=bedrock_settings,
        vertex_settings=vertex_settings,
    )
    return registry.resolve


def build_routed_llm_provider(
    *,
    surface: LlmRouteSurface,
    default_provider: LlmProviderName,
    llm_settings: LlmSettings,
    allowed_providers: set[LlmProviderName],
    provider_registry: CachedLlmProviderRegistry,
) -> RoutedLlmProvider:
    return RoutedLlmProvider(
        surface=surface,
        default_provider=default_provider,
        overrides=llm_settings.llm_model_provider_overrides,
        allowed_providers=allowed_providers,
        resolve_provider=provider_registry.resolve,
    )


def _build_provider(
    *,
    provider_name: LlmProviderName,
    llm_settings: LlmSettings,
    bedrock_settings: BedrockSettings,
    vertex_settings: VertexSettings,
) -> LlmProviderPort:
    max_concurrent = _max_concurrent_for_provider(provider_name, llm_settings)

    if provider_name == "bedrock":
        return LlmProviderAdapter(
            provider_name=provider_name,
            delegate=BedrockLlmProvider(
                region=bedrock_settings.region_value,
                connect_timeout_seconds=bedrock_settings.connect_timeout_seconds,
                read_timeout_seconds=bedrock_settings.read_timeout_seconds,
                max_concurrent=max_concurrent,
            ),
        )

    if provider_name == "chutes":
        return LlmProviderAdapter(
            provider_name=provider_name,
            delegate=ChutesLlmProvider(
                base_url=CHUTES.base_url,
                api_key=llm_settings.chutes_api_key_value,
                timeout=CHUTES.timeout_seconds,
                max_concurrent=max_concurrent,
            ),
        )

    if provider_name == "vertex":
        return LlmProviderAdapter(
            provider_name=provider_name,
            delegate=VertexLlmProvider(
                project=vertex_settings.gcp_project_id,
                location=vertex_settings.gcp_location,
                timeout=vertex_settings.vertex_timeout_seconds,
                service_account_b64=vertex_settings.gcp_sa_credential_b64_value,
                max_concurrent=max_concurrent,
            ),
        )

    raise ValueError(f"unsupported llm provider: {provider_name}")


def _max_concurrent_for_provider(
    provider_name: LlmProviderName,
    llm_settings: LlmSettings,
) -> int:
    if provider_name == "bedrock":
        return llm_settings.bedrock_max_concurrent
    if provider_name == "vertex":
        return llm_settings.vertex_max_concurrent
    if provider_name == "chutes":
        return llm_settings.chutes_max_concurrent
    raise ValueError(f"unsupported llm provider: {provider_name}")


__all__ = [
    "CachedLlmProviderRegistry",
    "build_cached_llm_provider_registry",
    "build_cached_llm_provider_resolver",
    "build_routed_llm_provider",
]
