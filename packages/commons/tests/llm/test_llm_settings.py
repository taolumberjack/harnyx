from __future__ import annotations

import pytest

from harnyx_commons.config.llm import (
    LlmSettings,
    OpenRouterModelProviderOptions,
    parse_openrouter_model_provider_options_json,
)


def test_openrouter_api_key_value_strips_secret() -> None:
    settings = LlmSettings(OPENROUTER_API_KEY=" test-openrouter-key ")

    assert settings.openrouter_api_key_value == "test-openrouter-key"


def test_parse_openrouter_model_provider_options_json_normalizes_provider_order() -> None:
    parsed = parse_openrouter_model_provider_options_json(
        '{" openai/gpt-oss-120b ":{"order":[" Cerebras ","Groq"],"require_parameters":true}}'
    )

    assert parsed == {
        "openai/gpt-oss-120b": OpenRouterModelProviderOptions(
            order=("Cerebras", "Groq"),
            require_parameters=True,
        )
    }
    assert parsed["openai/gpt-oss-120b"].to_request_extra() == {
        "provider": {"order": ["Cerebras", "Groq"], "require_parameters": True}
    }


def test_parse_openrouter_model_provider_options_json_omits_unset_options() -> None:
    parsed = parse_openrouter_model_provider_options_json('{"openai/gpt-oss-120b":{}}')

    assert parsed["openai/gpt-oss-120b"].to_request_extra() == {}


def test_parse_openrouter_model_provider_options_json_rejects_non_object_payload() -> None:
    with pytest.raises(ValueError, match="decode to a JSON object"):
        parse_openrouter_model_provider_options_json("[]")


def test_parse_openrouter_model_provider_options_json_rejects_blank_provider_order_entry() -> None:
    with pytest.raises(ValueError, match="provider.order entries must be non-empty"):
        parse_openrouter_model_provider_options_json('{"openai/gpt-oss-120b":{"order":[" "]}}')
