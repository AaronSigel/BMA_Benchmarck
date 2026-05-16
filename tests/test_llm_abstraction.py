from typing import Any

import pytest
from pydantic import ValidationError

from benchmark.agent.llm.base import LlmClient, LlmMessage, LlmResponse, LlmToolCall, LlmUsage


class StaticLlmClient:
    def __init__(self, response: LlmResponse) -> None:
        self.response = response

    def complete(
        self,
        messages: list[LlmMessage],
        tools: list[dict[str, Any]] | None = None,
        timeout_sec: int | float | None = None,
    ) -> LlmResponse:
        return self.response


def test_llm_models_support_tool_calls() -> None:
    response = LlmResponse(
        content=None,
        tool_calls=[
            LlmToolCall(
                id="call-1",
                name="get_scene_info",
                arguments={"include": "objects"},
                raw={"type": "function"},
            )
        ],
        finish_reason="tool_calls",
        usage=LlmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="mock",
    )

    assert response.tool_calls[0].name == "get_scene_info"
    assert response.tool_calls[0].arguments == {"include": "objects"}
    assert response.usage is not None
    assert response.usage.total_tokens == 15
    assert response.has_action() is True


def test_llm_response_supports_fallback_json_action_in_content() -> None:
    response = LlmResponse(
        content='{"tool_name": "get_scene_info", "arguments": {"include": "objects"}}'
    )

    assert response.tool_calls == []
    assert response.json_action() == {
        "tool_name": "get_scene_info",
        "arguments": {"include": "objects"},
    }
    assert response.has_action() is True


def test_llm_response_usage_is_optional() -> None:
    response = LlmResponse(content="final answer")

    assert response.usage is None
    assert response.tool_calls == []
    assert response.json_action() is None
    assert response.has_action() is False


def test_llm_client_protocol_is_provider_neutral() -> None:
    client = StaticLlmClient(LlmResponse(content="ok"))

    assert isinstance(client, LlmClient)
    assert client.complete([LlmMessage(role="user", content="hello")]).content == "ok"


def test_llm_abstraction_validation() -> None:
    with pytest.raises(ValidationError):
        LlmMessage(role="")
    with pytest.raises(ValidationError):
        LlmToolCall(name="")
    with pytest.raises(ValidationError):
        LlmUsage(total_tokens=-1)
