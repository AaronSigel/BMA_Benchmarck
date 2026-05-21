from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from benchmark.agent.errors import LlmClientError, LlmResponseParseError
from benchmark.agent.llm.base import LlmMessage, LlmToolCall
from benchmark.agent.llm.openrouter_client import OpenRouterClient
from benchmark.agent.models import LlmConfig, LlmProvider


def _make_config(**kwargs: Any) -> LlmConfig:
    defaults = dict(provider=LlmProvider.OPENROUTER, model="openai/gpt-4o-mini")
    return LlmConfig(**{**defaults, **kwargs})


def _make_response(
    content: str | None = "Hello",
    tool_calls: list[dict] | None = None,
    finish_reason: str = "stop",
    model: str = "openai/gpt-4o-mini",
    usage: dict | None = None,
    status_code: int = 200,
) -> MagicMock:
    msg: dict[str, Any] = {}
    if content is not None:
        msg["content"] = content
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    data = {
        "model": model,
        "choices": [{"message": msg, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
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
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = OpenRouterClient(_make_config())
    with pytest.raises(LlmClientError, match="OPENROUTER_API_KEY"):
        client.complete(_messages())


def test_custom_api_key_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MY_KEY", raising=False)
    client = OpenRouterClient(_make_config(api_key_env="MY_KEY"))
    with pytest.raises(LlmClientError, match="MY_KEY"):
        client.complete(_messages())


def test_api_key_not_leaked_in_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        OpenRouterClient(_make_config()).complete(_messages())
    _, kwargs = mock_post.call_args
    payload = kwargs.get("json", {})
    assert "sk-secret" not in json.dumps(payload)
    assert kwargs["headers"]["Authorization"] == "Bearer sk-secret"


# ---------------------------------------------------------------------------
# Successful responses
# ---------------------------------------------------------------------------

def test_complete_returns_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_make_response(content="World")):
        result = OpenRouterClient(_make_config()).complete(_messages())
    assert result.content == "World"
    assert result.tool_calls == []
    assert result.finish_reason == "stop"


def test_complete_with_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    raw_tc = [
        {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_scene_info", "arguments": '{"detail": "full"}'},
        }
    ]
    with patch("httpx.post", return_value=_make_response(content=None, tool_calls=raw_tc)):
        result = OpenRouterClient(_make_config()).complete(_messages())
    assert len(result.tool_calls) == 1
    tc: LlmToolCall = result.tool_calls[0]
    assert tc.id == "call_abc"
    assert tc.name == "get_scene_info"
    assert tc.arguments == {"detail": "full"}
    assert tc.raw is not None


def test_tool_calls_with_invalid_json_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    raw_tc = [{"id": "x", "function": {"name": "foo", "arguments": "not-json"}}]
    with patch("httpx.post", return_value=_make_response(content=None, tool_calls=raw_tc)):
        result = OpenRouterClient(_make_config()).complete(_messages())
    assert result.tool_calls[0].arguments == {}


def test_complete_usage_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    usage = {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "cost": 0.000123}
    with patch("httpx.post", return_value=_make_response(usage=usage)):
        result = OpenRouterClient(_make_config()).complete(_messages())
    assert result.usage is not None
    assert result.usage.prompt_tokens == 20
    assert result.usage.completion_tokens == 10
    assert result.usage.total_tokens == 30
    assert result.usage.cost == pytest.approx(0.000123)
    assert result.usage.provider_name == "openrouter"


def test_complete_passes_tools_in_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    tools = [{"type": "function", "function": {"name": "get_scene_info", "parameters": {}}}]
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        OpenRouterClient(_make_config()).complete(_messages(), tools=tools)
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["tools"] == tools


def test_complete_omits_tools_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        OpenRouterClient(_make_config()).complete(_messages(), tools=None)
    _, kwargs = mock_post.call_args
    assert "tools" not in kwargs["json"]


def test_custom_timeout_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        OpenRouterClient(_make_config()).complete(_messages(), timeout_sec=99)
    _, kwargs = mock_post.call_args
    assert kwargs["timeout"] == 99


def test_extra_headers_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    config = _make_config(extra_headers={"X-Custom": "value"})
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        OpenRouterClient(config).complete(_messages())
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["X-Custom"] == "value"


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------

def _error_response(status_code: int, message: str = "error") -> MagicMock:
    data = {"error": {"message": message}}
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    mock.json.return_value = data
    mock.text = json.dumps(data)
    return mock


def test_401_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_error_response(401, "Invalid API key")):
        with pytest.raises(LlmClientError, match="401"):
            OpenRouterClient(_make_config()).complete(_messages())


def test_429_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_error_response(429, "Rate limit")):
        with pytest.raises(LlmClientError, match="429"):
            OpenRouterClient(_make_config()).complete(_messages())


def test_4xx_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_error_response(422, "Unprocessable")):
        with pytest.raises(LlmClientError, match="422"):
            OpenRouterClient(_make_config()).complete(_messages())


def test_5xx_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    with patch("httpx.post", return_value=_error_response(503, "Service unavailable")):
        with pytest.raises(LlmClientError, match="503"):
            OpenRouterClient(_make_config()).complete(_messages())


def test_timeout_exception_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    with patch("httpx.post", side_effect=httpx.TimeoutException("timed out")):
        with pytest.raises(LlmClientError, match="timed out"):
            OpenRouterClient(_make_config()).complete(_messages())


def test_request_error_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    with patch("httpx.post", side_effect=httpx.ConnectError("connection refused")):
        with pytest.raises(LlmClientError, match="connection refused"):
            OpenRouterClient(_make_config()).complete(_messages())


def test_empty_choices_raises_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = {"choices": [], "model": "x"}
    with patch("httpx.post", return_value=mock):
        with pytest.raises(LlmResponseParseError, match="No choices"):
            OpenRouterClient(_make_config()).complete(_messages())


# ---------------------------------------------------------------------------
# Real API test (skipped unless --llm flag or OPENROUTER_API_KEY is set)
# ---------------------------------------------------------------------------

@pytest.mark.llm
@pytest.mark.api_e2e
def test_real_openrouter_completion() -> None:
    import os
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")
    config = _make_config(model="openai/gpt-4o-mini")
    client = OpenRouterClient(config)
    result = client.complete([LlmMessage(role="user", content="Say hello in one word.")])
    assert result.content is not None
    assert len(result.content) > 0
