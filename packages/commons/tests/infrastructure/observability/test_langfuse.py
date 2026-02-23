from __future__ import annotations

import pytest

from caster_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest, LlmUsage
from caster_commons.observability import langfuse

_LANGFUSE_ENV_VARS = ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


@pytest.fixture(autouse=True)
def _reset_langfuse_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(langfuse, "_LANGFUSE_CLIENT", None)


def _request() -> LlmRequest:
    return LlmRequest(
        provider="openai",
        model="gpt-5-mini",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
        output_mode="text",
    )


def _request_with_metadata(metadata: dict[str, object]) -> LlmRequest:
    return LlmRequest(
        provider="openai",
        model="gpt-5-mini",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("hello"),),
            ),
        ),
        temperature=None,
        max_output_tokens=64,
        output_mode="text",
        internal_metadata=metadata,
    )


def test_read_config_returns_none_when_all_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _LANGFUSE_ENV_VARS:
        monkeypatch.delenv(key, raising=False)

    assert langfuse._read_config() is None


def test_read_config_raises_runtime_error_for_partial_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Langfuse configuration is partial"):
        langfuse._read_config()


def test_read_config_returns_mapping_for_full_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", " https://langfuse.example ")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", " pk-test ")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", " sk-test ")

    assert langfuse._read_config() == {
        "LANGFUSE_HOST": "https://langfuse.example",
        "LANGFUSE_PUBLIC_KEY": "pk-test",
        "LANGFUSE_SECRET_KEY": "sk-test",
    }


def test_start_llm_generation_returns_none_scope_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in _LANGFUSE_ENV_VARS:
        monkeypatch.delenv(key, raising=False)

    scope = langfuse.start_llm_generation(
        trace_id="trace-id",
        provider_label="openai",
        request=_request(),
    )
    with scope as generation:
        assert generation is None


def test_update_generation_best_effort_swallows_update_exception() -> None:
    class RaisingGeneration:
        def update(self, **kwargs: object) -> None:
            raise RuntimeError("update failed")

    langfuse.update_generation_best_effort(
        RaisingGeneration(),
        output={"ok": True},
        usage=LlmUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        metadata={"provider": "openai"},
    )


def test_build_generation_metadata_merges_internal_metadata_with_canonical_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_SERVICE_NAME", "caster-platform-worker")
    request = _request_with_metadata(
        {
            "use_case": "claim_generation",
            "feed_run_id": "feed-run-123",
            "server": "caller-supplied-server",
        }
    )

    metadata = langfuse.build_generation_metadata(
        provider_label="openai",
        request=request,
        metadata={"elapsed_ms": 12.3},
    )

    assert metadata["provider"] == "openai"
    assert metadata["server"] == "caster-platform-worker"
    assert metadata["use_case"] == "claim_generation"
    assert metadata["feed_run_id"] == "feed-run-123"
    assert metadata["elapsed_ms"] == 12.3


def test_derive_tags_uses_only_low_cardinality_dimensions() -> None:
    tags = langfuse._derive_tags(
        {
            "server": "caster-platform-worker",
            "use_case": "claim_generation",
            "feed_run_id": "feed-run-123",
            "user_id": "u-99",
        }
    )

    assert tags == ["server:caster-platform-worker", "use_case:claim_generation"]


def test_close_propagate_scope_swallows_exit_exception_and_clears_state(caplog: pytest.LogCaptureFixture) -> None:
    class RaisingPropagateContextManager:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            raise RuntimeError("propagate cleanup failed")

    scope = langfuse._LangfuseGenerationScope(
        client=None,
        trace_id="trace-id",
        provider_label="openai",
        request=_request(),
    )
    scope._propagate_cm = RaisingPropagateContextManager()

    caplog.set_level("ERROR", logger="caster_commons.observability.langfuse")
    scope._close_propagate_scope()

    assert scope._propagate_cm is None
    assert "langfuse.generation.propagate_cleanup_failed" in [record.message for record in caplog.records]
    assert {"provider": "openai", "model": "gpt-5-mini"} in [
        record.__dict__.get("data") for record in caplog.records
    ]


def test_generation_scope_enter_error_path_does_not_raise_when_propagate_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class RaisingPropagateContextManager:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            raise RuntimeError("propagate cleanup failed")

    class FailingObservationContextManager:
        def __enter__(self) -> object:
            raise RuntimeError("observation start failed")

        def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            return False

    class FakeClient:
        def start_as_current_observation(self, **kwargs: object) -> FailingObservationContextManager:
            return FailingObservationContextManager()

    monkeypatch.setattr(
        langfuse,
        "propagate_attributes",
        lambda *, tags: RaisingPropagateContextManager(),
    )
    caplog.set_level("ERROR", logger="caster_commons.observability.langfuse")

    scope = langfuse._LangfuseGenerationScope(
        client=FakeClient(),
        trace_id="trace-id",
        provider_label="openai",
        request=_request(),
    )
    generation = scope.__enter__()

    assert generation is None
    assert scope._propagate_cm is None
    messages = [record.message for record in caplog.records]
    assert "langfuse.generation.propagate_cleanup_failed" in messages
    assert "langfuse.generation.start_failed" in messages


def test_generation_scope_exit_path_swallows_propagate_cleanup_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class RaisingPropagateContextManager:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            raise RuntimeError("propagate cleanup failed")

    class SuccessfulObservationContextManager:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, exc_type: object, exc: object, exc_tb: object) -> bool:
            return True

    scope = langfuse._LangfuseGenerationScope(
        client=None,
        trace_id="trace-id",
        provider_label="openai",
        request=_request(),
    )
    scope._observation_cm = SuccessfulObservationContextManager()
    scope._propagate_cm = RaisingPropagateContextManager()

    caplog.set_level("ERROR", logger="caster_commons.observability.langfuse")
    result = scope.__exit__(None, None, None)

    assert result is True
    assert scope._propagate_cm is None
    assert "langfuse.generation.propagate_cleanup_failed" in [record.message for record in caplog.records]
