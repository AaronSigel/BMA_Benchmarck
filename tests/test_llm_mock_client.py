import pytest

from benchmark.agent.errors import AgentTimeoutError, LlmClientError
from benchmark.agent.llm import LlmMessage, LlmResponse, LlmToolCall, MockLlmClient


def test_mock_llm_client_returns_responses_in_order() -> None:
    client = MockLlmClient(
        [
            LlmResponse(content="first"),
            LlmResponse(content="second"),
        ]
    )

    assert client.complete([LlmMessage(role="user", content="one")]).content == "first"
    assert client.complete([LlmMessage(role="user", content="two")]).content == "second"
    assert client.remaining_responses == 0
    assert len(client.calls) == 2
    assert client.calls[0].messages[0].content == "one"


def test_mock_llm_client_returns_tool_calls() -> None:
    client = MockLlmClient(
        [
            LlmResponse(
                tool_calls=[
                    LlmToolCall(name="get_scene_info", arguments={"include": "objects"})
                ]
            )
        ]
    )

    response = client.complete(
        [LlmMessage(role="user", content="inspect")],
        tools=[{"name": "get_scene_info"}],
        timeout_sec=10,
    )

    assert response.tool_calls[0].name == "get_scene_info"
    assert response.tool_calls[0].arguments == {"include": "objects"}
    assert client.calls[0].tools == [{"name": "get_scene_info"}]
    assert client.calls[0].timeout_sec == 10


def test_mock_llm_client_returns_json_action() -> None:
    client = MockLlmClient(
        [
            LlmResponse(
                content='{"tool_name": "get_scene_info", "arguments": {"include": "objects"}}'
            )
        ]
    )

    response = client.complete([LlmMessage(role="user", content="inspect")])

    assert response.json_action() == {
        "tool_name": "get_scene_info",
        "arguments": {"include": "objects"},
    }


def test_mock_llm_client_can_raise_configured_error() -> None:
    client = MockLlmClient(error="planned failure")

    with pytest.raises(LlmClientError, match="planned failure"):
        client.complete([LlmMessage(role="user", content="hello")])


def test_mock_llm_client_can_raise_sequence_error() -> None:
    client = MockLlmClient([LlmClientError("step failed")])

    with pytest.raises(LlmClientError, match="step failed"):
        client.complete([LlmMessage(role="user", content="hello")])


def test_mock_llm_client_can_raise_timeout() -> None:
    client = MockLlmClient(timeout=True)

    with pytest.raises(AgentTimeoutError, match="Mock LLM timeout"):
        client.complete([LlmMessage(role="user", content="hello")])


def test_mock_llm_client_exhaustion_has_clear_error() -> None:
    client = MockLlmClient([])

    with pytest.raises(LlmClientError, match="response sequence exhausted"):
        client.complete([LlmMessage(role="user", content="hello")])
