from caster_commons.llm.provider import _request_snapshot
from caster_commons.llm.schema import GroundedLlmRequest, LlmMessage, LlmMessageContentPart, LlmTool


def test_request_snapshot_keeps_nested_api_key_fields() -> None:
    request = GroundedLlmRequest(
        provider="vertex",
        model="gemini-2.0",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text("check novelty"),),
            ),
        ),
        temperature=None,
        max_output_tokens=None,
        tools=(
            LlmTool(
                type="provider_native",
                config={
                    "retrieval": {
                        "external_api": {
                            "api_auth": {
                                "api_key_config": {
                                    "api_key_string": "ApiKey secret-snake",
                                    "apiKeyString": "ApiKey secret-camel",
                                }
                            }
                        }
                    }
                },
            ),
        ),
    )

    snapshot = _request_snapshot(request)
    tool_config = snapshot["tools"][0]["config"]["retrieval"]["external_api"]["api_auth"]["api_key_config"]

    assert tool_config["api_key_string"] == "ApiKey secret-snake"
    assert tool_config["apiKeyString"] == "ApiKey secret-camel"
