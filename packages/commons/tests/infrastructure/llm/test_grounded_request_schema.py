from __future__ import annotations

import pytest

from caster_commons.llm.schema import GroundedLlmRequest, LlmMessage, LlmMessageContentPart, LlmTool


def _tools() -> tuple[LlmTool, ...]:
    return (
        LlmTool(
            type="provider_native",
            config={"retrieval": {"external_api": {"api_spec": "ELASTIC_SEARCH"}}},
        ),
    )


def _messages() -> tuple[LlmMessage, ...]:
    return (
        LlmMessage(
            role="user",
            content=(LlmMessageContentPart.input_text("hello"),),
        ),
    )


def test_grounded_openai_rejects_additional_tools() -> None:
    with pytest.raises(
        ValueError,
        match="grounded requests with additional tools are only supported for provider 'vertex'",
    ):
        GroundedLlmRequest(
            provider="openai",
            model="gpt-5-mini",
            messages=_messages(),
            temperature=None,
            max_output_tokens=None,
            tools=_tools(),
        )


def test_grounded_vertex_gemini_allows_additional_tools() -> None:
    request = GroundedLlmRequest(
        provider="vertex",
        model="gemini-2.5-flash",
        messages=_messages(),
        temperature=None,
        max_output_tokens=None,
        tools=_tools(),
    )
    assert request.tools is not None


def test_grounded_vertex_claude_rejects_additional_tools() -> None:
    with pytest.raises(
        ValueError,
        match="grounded requests with additional tools are not supported for Vertex Claude models",
    ):
        GroundedLlmRequest(
            provider="vertex",
            model="publishers/anthropic/models/claude-sonnet-4-5@20250929",
            messages=_messages(),
            temperature=None,
            max_output_tokens=None,
            tools=_tools(),
        )
