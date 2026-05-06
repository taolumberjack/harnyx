from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from harnyx_commons.config.bedrock import BedrockSettings
from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.config.observability import ObservabilitySettings
from harnyx_commons.config.platform_api import PlatformApiSettings
from harnyx_commons.config.sandbox import SandboxSettings
from harnyx_commons.config.subtensor import SubtensorSettings
from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.errors import ConcurrencyLimitError
from harnyx_commons.llm.routing import ResolvedLlmRoute
from harnyx_commons.llm.schema import (
    LlmChoice,
    LlmChoiceMessage,
    LlmMessage,
    LlmMessageContentPart,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)
from harnyx_validator.runtime import bootstrap
from harnyx_validator.runtime.bootstrap import (
    _build_llm_clients,
    _build_local_eval_tooling_clients,
    _create_scoring_service,
    _create_search_client,
    close_runtime_resources,
)
from harnyx_validator.runtime.settings import Settings


class _FakeLlmProvider:
    def __init__(self) -> None:
        self.requests: list[LlmRequest] = []

    async def invoke(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            id="resp-1",
            choices=(
                LlmChoice(
                    index=0,
                    message=LlmChoiceMessage(
                        role="assistant",
                        content=(LlmMessageContentPart(type="text", text="ok"),),
                    ),
                    finish_reason="stop",
                ),
            ),
            usage=LlmUsage(),
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        return None


class _FakeLlmRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, _FakeLlmProvider] = {}

    @property
    def requests_by_provider(self) -> dict[str, list[LlmRequest]]:
        return {provider_name: provider.requests for provider_name, provider in self._providers.items()}

    def resolve(self, name: str) -> _FakeLlmProvider:
        provider = self._providers.get(name)
        if provider is None:
            provider = _FakeLlmProvider()
            self._providers[name] = provider
        return provider


def test_llm_settings_default_scoring_timeout_is_two_minutes() -> None:
    assert LlmSettings().scoring_llm_timeout_seconds == pytest.approx(120.0)


def _settings_with_gemma_tool_route() -> Settings:
    return Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider=None,
            tool_llm_provider="chutes",
            scoring_llm_provider="chutes",
            chutes_api_key=SecretStr("test-key"),
            llm_model_provider_overrides_json=json.dumps(
                {"tool": {"google/gemma-4-31B-it": "custom-openai-compatible:gemma4-cloud-run"}}
            ),
            openai_compatible_endpoints_json=json.dumps(
                [
                    {
                        "id": "gemma4-cloud-run",
                        "base_url": "https://gemma.example.run.app/v1",
                        "auth": {"type": "none"},
                    }
                ]
            ),
        ),
        bedrock=BedrockSettings.model_construct(region="us-east-1"),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
    )


def _gemma_tool_request() -> LlmRequest:
    return LlmRequest(
        provider="chutes",
        model="google/gemma-4-31B-it",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=8,
    )


def test_create_search_client_requires_search_provider() -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider=None,
        )
    )

    with pytest.raises(RuntimeError, match="SEARCH_PROVIDER must be configured"):
        _create_search_client(settings)


def test_create_search_client_uses_configured_parallel_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeParallelClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(bootstrap, "ParallelClient", _FakeParallelClient)
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider="parallel",
            parallel_base_url="https://proxy.parallel.test",
            parallel_api_key=SecretStr("parallel-key"),
            parallel_max_concurrent=7,
        )
    )

    _create_search_client(settings)

    assert captured["base_url"] == "https://proxy.parallel.test"
    assert captured["api_key"] == "parallel-key"
    assert captured["max_concurrent"] == 7


def test_build_llm_clients_uses_shared_provider_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider="parallel",
            parallel_base_url="https://proxy.parallel.test",
            parallel_api_key=SecretStr("parallel-key"),
            parallel_max_concurrent=7,
            tool_llm_provider="chutes",
            scoring_llm_provider="vertex",
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
        bedrock=BedrockSettings.model_construct(region="us-east-1"),
    )
    calls: list[str] = []

    class _FakeRegistry:
        def resolve(self, name: str) -> str:
            calls.append(name)
            return f"provider:{name}"

    def fake_build_cached_llm_provider_registry(*, llm_settings, bedrock_settings, vertex_settings):
        assert llm_settings is settings.llm
        assert bedrock_settings is settings.bedrock
        assert vertex_settings is settings.vertex
        return _FakeRegistry()

    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", fake_build_cached_llm_provider_registry)

    _, provider_registry, tool_provider, scoring_provider, scoring_route = _build_llm_clients(settings)

    assert type(tool_provider).__name__ == "RoutedLlmProvider"
    assert scoring_provider == "provider:vertex"
    assert type(provider_registry).__name__ == "_FakeRegistry"
    assert scoring_route == ResolvedLlmRoute(
        surface="scoring",
        provider="vertex",
        model=bootstrap._SCORING_LLM_MODEL,
    )
    assert calls == ["vertex"]


def test_build_local_eval_tooling_clients_allows_missing_search_provider() -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider=None,
            tool_llm_provider="chutes",
            scoring_llm_provider="chutes",
            chutes_api_key=SecretStr("test-key"),
        ),
        bedrock=BedrockSettings.model_construct(region="us-east-1"),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
    )

    (
        search_client,
        provider_registry,
        tool_provider,
        scoring_provider,
        scoring_route,
    ) = _build_local_eval_tooling_clients(settings)

    assert search_client is None
    assert provider_registry is not None
    assert tool_provider is not None
    assert scoring_provider is not None
    assert type(tool_provider).__name__ == "_LazyLlmProvider"
    assert scoring_route == ResolvedLlmRoute(
        surface="scoring",
        provider="chutes",
        model=bootstrap._SCORING_LLM_MODEL,
    )


@pytest.mark.anyio("asyncio")
async def test_build_llm_clients_routes_gemma_tool_model_to_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings_with_gemma_tool_route()
    registry = _FakeLlmRegistry()
    monkeypatch.setattr(bootstrap, "_create_search_client", lambda _: None)
    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", lambda **_: registry)

    _, _, tool_provider, _, _ = _build_llm_clients(settings)

    assert tool_provider is not None
    await tool_provider.invoke(_gemma_tool_request())

    assert registry.requests_by_provider["custom-openai-compatible:gemma4-cloud-run"][0].provider == (
        "custom-openai-compatible:gemma4-cloud-run"
    )


@pytest.mark.anyio("asyncio")
async def test_build_local_eval_tooling_clients_routes_gemma_tool_model_to_custom_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings_with_gemma_tool_route()
    registry = _FakeLlmRegistry()
    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", lambda **_: registry)

    _, _, tool_provider, _, _ = _build_local_eval_tooling_clients(settings)

    assert tool_provider is not None
    await tool_provider.invoke(_gemma_tool_request())

    assert registry.requests_by_provider["custom-openai-compatible:gemma4-cloud-run"][0].provider == (
        "custom-openai-compatible:gemma4-cloud-run"
    )


def test_build_llm_clients_uses_scoring_model_override_for_route_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider="parallel",
            parallel_base_url="https://proxy.parallel.test",
            parallel_api_key=SecretStr("parallel-key"),
            parallel_max_concurrent=7,
            tool_llm_provider="chutes",
            scoring_llm_provider="vertex",
            scoring_llm_model_override="custom/internal-model",
            llm_model_provider_overrides_json=json.dumps({"scoring": {"custom/internal-model": "bedrock"}}),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
        bedrock=BedrockSettings.model_construct(region="us-east-1"),
    )

    class _FakeRegistry:
        def resolve(self, name: str) -> str:
            return f"provider:{name}"

    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", lambda **_: _FakeRegistry())

    _, _, _, scoring_provider, scoring_route = _build_llm_clients(settings)

    assert scoring_provider == "provider:bedrock"
    assert scoring_route == ResolvedLlmRoute(
        surface="scoring",
        provider="bedrock",
        model="custom/internal-model",
    )


def test_build_state_activates_two_parallel_tool_calls_per_token() -> None:
    state = bootstrap._build_state()

    state.token_semaphore.acquire("token")
    state.token_semaphore.acquire("token")
    with pytest.raises(ConcurrencyLimitError):
        state.token_semaphore.acquire("token")

    state.token_semaphore.release("token")
    state.token_semaphore.release("token")


def test_create_scoring_service_does_not_require_vertex_config_at_bootstrap() -> None:
    settings = Settings.model_construct(
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
        llm=LlmSettings.model_construct(
            scoring_llm_provider="chutes",
            scoring_llm_temperature=None,
            scoring_llm_max_output_tokens=1024,
            scoring_llm_timeout_seconds=30.0,
            chutes_api_key=SecretStr("test-key"),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id=None,
            gcp_location=None,
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr(""),
        ),
        sandbox=SandboxSettings.model_construct(
            sandbox_image="harnyx-sandbox:test",
            sandbox_network="harnyx-sandbox-net",
            sandbox_pull_policy="always",
        ),
        platform_api=PlatformApiSettings.model_construct(
            platform_base_url=None,
            validator_public_base_url=None,
        ),
        observability=ObservabilitySettings.model_construct(
            enable_cloud_logging=False,
            gcp_project_id=None,
        ),
        subtensor=SubtensorSettings.model_construct(
            network="local",
            endpoint="ws://127.0.0.1:9945",
            netuid=1,
            wallet_name="harnyx-validator",
            hotkey_name="default",
            hotkey_mnemonic=None,
            wait_for_inclusion=True,
            wait_for_finalization=False,
            transaction_mode="immortal",
            transaction_period=None,
        ),
    )

    service = _create_scoring_service(
        settings,
        provider=SimpleNamespace(),
        scoring_route=ResolvedLlmRoute(
            surface="scoring",
            provider="chutes",
            model=bootstrap._SCORING_LLM_MODEL,
        ),
    )

    assert service is not None
    assert service._config.provider == "chutes"
    assert service._config.model == bootstrap._SCORING_LLM_MODEL
    assert service._config.reasoning_effort == bootstrap._SCORING_LLM_REASONING_EFFORT


def test_create_scoring_service_uses_effective_route_model_and_provider() -> None:
    settings = Settings.model_construct(
        rpc_listen_host="127.0.0.1",
        rpc_port=8100,
        llm=LlmSettings.model_construct(
            scoring_llm_provider="vertex",
            scoring_llm_temperature=None,
            scoring_llm_max_output_tokens=1024,
            scoring_llm_timeout_seconds=30.0,
            chutes_api_key=SecretStr("test-key"),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
        sandbox=SandboxSettings.model_construct(
            sandbox_image="harnyx-sandbox:test",
            sandbox_network="harnyx-sandbox-net",
            sandbox_pull_policy="always",
        ),
        platform_api=PlatformApiSettings.model_construct(
            platform_base_url=None,
            validator_public_base_url=None,
        ),
        observability=ObservabilitySettings.model_construct(
            enable_cloud_logging=False,
            gcp_project_id=None,
        ),
        subtensor=SubtensorSettings.model_construct(
            network="local",
            endpoint="ws://127.0.0.1:9945",
            netuid=1,
            wallet_name="harnyx-validator",
            hotkey_name="default",
            hotkey_mnemonic=None,
            wait_for_inclusion=True,
            wait_for_finalization=False,
            transaction_mode="immortal",
            transaction_period=None,
        ),
    )

    service = _create_scoring_service(
        settings,
        provider=SimpleNamespace(),
        scoring_route=ResolvedLlmRoute(
            surface="scoring",
            provider="bedrock",
            model="custom/internal-model",
        ),
    )

    assert service._config.provider == "bedrock"
    assert service._config.model == "custom/internal-model"


def test_build_llm_clients_allows_bedrock_scoring_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.model_construct(
        llm=LlmSettings.model_construct(
            search_provider="parallel",
            parallel_base_url="https://proxy.parallel.test",
            parallel_api_key=SecretStr("parallel-key"),
            parallel_max_concurrent=7,
            tool_llm_provider="chutes",
            scoring_llm_provider="vertex",
            llm_model_provider_overrides_json=json.dumps(
                {"scoring": {bootstrap._SCORING_LLM_MODEL: "bedrock"}}
            ),
        ),
        vertex=VertexSettings.model_construct(
            gcp_project_id="project",
            gcp_location="us-central1",
            vertex_timeout_seconds=60.0,
            gcp_service_account_credential_b64=SecretStr("vertex-creds"),
        ),
        bedrock=BedrockSettings.model_construct(region="us-east-1"),
    )

    class _FakeRegistry:
        def resolve(self, name: str) -> str:
            return f"provider:{name}"

    monkeypatch.setattr(bootstrap, "build_cached_llm_provider_registry", lambda **_: _FakeRegistry())

    _, _, _, scoring_provider, scoring_route = _build_llm_clients(settings)

    assert scoring_provider == "provider:bedrock"
    assert scoring_route == ResolvedLlmRoute(
        surface="scoring",
        provider="bedrock",
        model=bootstrap._SCORING_LLM_MODEL,
    )


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _CountingClosable:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


class _ShutdownSpyExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[bool, bool]] = []

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.calls.append((wait, cancel_futures))


@pytest.mark.anyio
async def test_close_runtime_resources_closes_llm_provider_registry() -> None:
    llm_provider_registry = _Closable()
    blocking_executor = _ShutdownSpyExecutor()
    runtime = SimpleNamespace(
        batch_blocking_executor=blocking_executor,
        search_client=None,
        llm_provider_registry=llm_provider_registry,
        tool_llm_provider=None,
        scoring_llm_provider=None,
    )

    await close_runtime_resources(runtime)

    assert blocking_executor.calls == [(False, True)]
    assert llm_provider_registry.closed is True


@pytest.mark.anyio
async def test_close_runtime_resources_closes_registry_once() -> None:
    llm_provider_registry = _CountingClosable()
    blocking_executor = _ShutdownSpyExecutor()
    runtime = SimpleNamespace(
        batch_blocking_executor=blocking_executor,
        search_client=None,
        llm_provider_registry=llm_provider_registry,
        tool_llm_provider=None,
        scoring_llm_provider=None,
    )

    await close_runtime_resources(runtime)

    assert blocking_executor.calls == [(False, True)]
    assert llm_provider_registry.close_calls == 1
