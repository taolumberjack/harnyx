from __future__ import annotations

from harnyx_commons.llm.tool_models import resolve_tool_model, tool_model_thinking_capability


def test_tool_model_thinking_capabilities_share_the_canonical_model_owner() -> None:
    deepseek = tool_model_thinking_capability("deepseek-ai/deepseek-v3.2-tee", provider_name="chutes")
    glm = tool_model_thinking_capability("zai-org/GLM-5-TEE", provider_name="vertex")
    qwen = tool_model_thinking_capability("Qwen/Qwen3-Next-80B-A3B-Instruct", provider_name="chutes")
    qwen36 = tool_model_thinking_capability(
        "Qwen/Qwen3.6-27B-TEE",
        provider_name="custom-openai-compatible:qwen36-cloud-run",
    )
    gemma_chutes = tool_model_thinking_capability("google/gemma-4-31B-turbo-TEE", provider_name="chutes")
    gemma_custom = tool_model_thinking_capability(
        "google/gemma-4-31B-turbo-TEE",
        provider_name="custom-openai-compatible:gemma4-cloud-run-turbo",
    )

    assert resolve_tool_model("deepseek-ai/deepseek-v3.2-tee") == "deepseek-ai/DeepSeek-V3.2-TEE"
    assert resolve_tool_model("qwen/qwen3.6-27b-tee") == "Qwen/Qwen3.6-27B-TEE"
    assert deepseek is not None
    assert deepseek.chat_template_kwargs(enabled=True) == {"thinking": True}
    assert glm is not None
    assert glm.chat_template_kwargs(enabled=False) == {"enable_thinking": False}
    assert qwen is None
    assert qwen36 is not None
    assert qwen36.chat_template_kwargs(enabled=False) == {"enable_thinking": False}
    assert gemma_chutes is None
    assert gemma_custom is not None
    assert gemma_custom.chat_template_kwargs(enabled=True) == {"enable_thinking": True}
