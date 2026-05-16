from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from benchmark.agent.errors import LlmClientError, LlmResponseParseError
from benchmark.agent.llm.anthropic_client import AnthropicClient, _adapt_tools, _split_system_messages
from benchmark.agent.llm.base import LlmMessage
from benchmark.agent.models import LlmConfig, LlmProvider


def _make_config(**kwargs: Any) -> LlmConfig:
    defaults = dict(provider=LlmProvider.ANTHROPIC, model="claude-3-5-haiku-20241022")
    return LlmConfig(**{**defaults, **kwargs})


def _make_response(
    content_blocks: list[dict] | None = None,
    stop_reason: str = "end_turn",
    model: str = "claude-3-5-haiku-20241022",
    usage: dict | None = None,
    status_code: int = 200,
) -> MagicMock:
    if content_blocks is None:
        content_blocks = [{"type": "text", "text": "Hello"}]
    data = {
        "id": "msg_abc",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": usage or {"input_tokens": 10, "output_tokens": 5},
    }
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    mock.json.return_value = data
    mock.text = json.dumps(data)
    return mock


def _messages() -> list[LlmMessage]:
    return [LlmMessage(role="user", content="Inspect the scene")]


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LlmClientError, match="ANTHROPIC_API_KEY"):
        AnthropicClient(_make_config()).complete(_messages())


def test_custom_api_key_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MY_CLAUDE_KEY", raising=False)
    with pytest.raises(LlmClientError, match="MY_CLAUDE_KEY"):
        AnthropicClient(_make_config(api_key_env="MY_CLAUDE_KEY")).complete(_messages())


def test_api_key_sent_in_x_api_key_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        AnthropicClient(_make_config()).complete(_messages())
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["x-api-key"] == "sk-ant-secret"


def test_api_key_not_in_json_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        AnthropicClient(_make_config()).complete(_messages())
    _, kwargs = mock_post.call_args
    assert "sk-ant-secret" not in json.dumps(kwargs.get("json", {}))


# ---------------------------------------------------------------------------
# Request format
# ---------------------------------------------------------------------------

def test_anthropic_version_header_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        AnthropicClient(_make_config()).complete(_messages())
    _, kwargs = mock_post.call_args
    assert "anthropic-version" in kwargs["headers"]


def test_endpoint_is_v1_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        AnthropicClient(_make_config()).complete(_messages())
    url = mock_post.call_args[0][0]
    assert url.endswith("/v1/messages")


def test_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    config = _make_config(base_url="http://proxy:8080")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        AnthropicClient(config).complete(_messages())
    assert mock_post.call_args[0][0] == "http://proxy:8080/v1/messages"


def test_system_message_extracted_to_separate_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    msgs = [
        LlmMessage(role="system", content="You are helpful."),
        LlmMessage(role="user", content="Hi"),
    ]
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        AnthropicClient(_make_config()).complete(msgs)
    payload = mock_post.call_args[1]["json"]
    assert payload["system"] == "You are helpful."
    assert all(m["role"] != "system" for m in payload["messages"])


def test_no_system_field_when_no_system_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        AnthropicClient(_make_config()).complete(_messages())
    payload = mock_post.call_args[1]["json"]
    assert "system" not in payload


def test_tool_result_message_converted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    msgs = [
        LlmMessage(role="user", content="Do it"),
        LlmMessage(role="tool", content='{"objects": []}', tool_call_id="call_1"),
    ]
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        AnthropicClient(_make_config()).complete(msgs)
    payload = mock_post.call_args[1]["json"]
    tool_msg = payload["messages"][1]
    assert tool_msg["role"] == "user"
    assert tool_msg["content"][0]["type"] == "tool_result"
    assert tool_msg["content"][0]["tool_use_id"] == "call_1"


def test_custom_timeout_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        AnthropicClient(_make_config()).complete(_messages(), timeout_sec=45)
    assert mock_post.call_args[1]["timeout"] == 45


def test_extra_headers_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    config = _make_config(extra_headers={"anthropic-beta": "tools-2024-05-16"})
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        AnthropicClient(config).complete(_messages())
    assert mock_post.call_args[1]["headers"]["anthropic-beta"] == "tools-2024-05-16"


# ---------------------------------------------------------------------------
# Tool schema adaptation
# ---------------------------------------------------------------------------

def test_adapt_tools_converts_openai_to_anthropic_format() -> None:
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_scene_info",
                "description": "Get scene details",
                "parameters": {"type": "object", "properties": {"detail": {"type": "string"}}},
            },
        }
    ]
    adapted = _adapt_tools(openai_tools)
    assert len(adapted) == 1
    t = adapted[0]
    assert t["name"] == "get_scene_info"
    assert t["description"] == "Get scene details"
    assert "input_schema" in t
    assert "parameters" not in t
    assert "function" not in t


def test_adapt_tools_passthrough_anthropic_format() -> None:
    anthropic_tools = [{"name": "foo", "description": "bar", "input_schema": {}}]
    assert _adapt_tools(anthropic_tools) == anthropic_tools


def test_tools_sent_in_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    tools = [{"type": "function", "function": {"name": "foo", "parameters": {}}}]
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        AnthropicClient(_make_config()).complete(_messages(), tools=tools)
    payload = mock_post.call_args[1]["json"]
    assert "tools" in payload
    assert payload["tools"][0]["name"] == "foo"


def test_no_tools_key_when_tools_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        AnthropicClient(_make_config()).complete(_messages(), tools=None)
    assert "tools" not in mock_post.call_args[1]["json"]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def test_text_block_returned_as_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_make_response(
        content_blocks=[{"type": "text", "text": "World"}]
    )):
        result = AnthropicClient(_make_config()).complete(_messages())
    assert result.content == "World"
    assert result.tool_calls == []


def test_tool_use_block_parsed_as_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    blocks = [
        {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "create_cube",
            "input": {"size": 3, "location": [0, 0, 0]},
        }
    ]
    with patch("httpx.post", return_value=_make_response(content_blocks=blocks)):
        result = AnthropicClient(_make_config()).complete(_messages())
    assert result.content is None
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.id == "toolu_123"
    assert tc.name == "create_cube"
    assert tc.arguments == {"size": 3, "location": [0, 0, 0]}
    assert tc.raw is not None


def test_mixed_text_and_tool_use_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    blocks = [
        {"type": "text", "text": "I will call a tool."},
        {"type": "tool_use", "id": "t1", "name": "foo", "input": {}},
    ]
    with patch("httpx.post", return_value=_make_response(content_blocks=blocks)):
        result = AnthropicClient(_make_config()).complete(_messages())
    assert result.content == "I will call a tool."
    assert len(result.tool_calls) == 1


def test_usage_mapped_from_anthropic_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_make_response(
        usage={"input_tokens": 20, "output_tokens": 8}
    )):
        result = AnthropicClient(_make_config()).complete(_messages())
    assert result.usage is not None
    assert result.usage.prompt_tokens == 20
    assert result.usage.completion_tokens == 8
    assert result.usage.total_tokens == 28


def test_stop_reason_mapped_to_finish_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_make_response(stop_reason="tool_use")):
        result = AnthropicClient(_make_config()).complete(_messages())
    assert result.finish_reason == "tool_use"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def _error_response(status_code: int, message: str = "error") -> MagicMock:
    data = {"type": "error", "error": {"type": "api_error", "message": message}}
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    mock.json.return_value = data
    mock.text = json.dumps(data)
    return mock


def test_401_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_error_response(401, "Invalid API key")):
        with pytest.raises(LlmClientError, match="401"):
            AnthropicClient(_make_config()).complete(_messages())


def test_429_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_error_response(429, "Rate limit")):
        with pytest.raises(LlmClientError, match="429"):
            AnthropicClient(_make_config()).complete(_messages())


def test_4xx_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_error_response(400, "Bad request")):
        with pytest.raises(LlmClientError, match="400"):
            AnthropicClient(_make_config()).complete(_messages())


def test_5xx_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_error_response(529, "Overloaded")):
        with pytest.raises(LlmClientError, match="529"):
            AnthropicClient(_make_config()).complete(_messages())


def test_timeout_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("httpx.post", side_effect=httpx.TimeoutException("timed out")):
        with pytest.raises(LlmClientError, match="timed out"):
            AnthropicClient(_make_config()).complete(_messages())


def test_connect_error_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(LlmClientError, match="refused"):
            AnthropicClient(_make_config()).complete(_messages())


def test_non_list_content_raises_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = {"content": "not a list", "model": "x", "stop_reason": "end_turn"}
    with patch("httpx.post", return_value=mock):
        with pytest.raises(LlmResponseParseError):
            AnthropicClient(_make_config()).complete(_messages())


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

def test_split_system_messages_extracts_system() -> None:
    msgs = [
        LlmMessage(role="system", content="Be helpful."),
        LlmMessage(role="user", content="Hi"),
    ]
    system, converted = _split_system_messages(msgs)
    assert system == "Be helpful."
    assert len(converted) == 1
    assert converted[0]["role"] == "user"


def test_split_system_messages_no_system() -> None:
    msgs = [LlmMessage(role="user", content="Hi")]
    system, converted = _split_system_messages(msgs)
    assert system is None
    assert len(converted) == 1


def test_split_multiple_system_messages_joined() -> None:
    msgs = [
        LlmMessage(role="system", content="Part one."),
        LlmMessage(role="system", content="Part two."),
        LlmMessage(role="user", content="Go"),
    ]
    system, _ = _split_system_messages(msgs)
    assert system == "Part one.\nPart two."


# ---------------------------------------------------------------------------
# Real API test (disabled by default)
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_real_anthropic_completion() -> None:
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    result = AnthropicClient(_make_config(model="claude-3-5-haiku-20241022")).complete(
        [LlmMessage(role="user", content="Say hi in one word.")]
    )
    assert result.content is not None
