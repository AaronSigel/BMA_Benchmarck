from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from benchmark.agent.errors import LlmClientError, LlmResponseParseError
from benchmark.agent.llm.base import LlmMessage
from benchmark.agent.llm.openai_compatible_client import OpenAICompatibleClient
from benchmark.agent.models import LlmConfig, LlmProvider


def _make_config(**kwargs: Any) -> LlmConfig:
    defaults = dict(
        provider=LlmProvider.OPENAI_COMPATIBLE,
        model="gpt-4o-mini",
        base_url="http://localhost:11434/v1",
    )
    return LlmConfig(**{**defaults, **kwargs})


def _make_response(
    content: str | None = "Hello",
    tool_calls: list[dict] | None = None,
    finish_reason: str = "stop",
    model: str = "gpt-4o-mini",
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
        "usage": usage or {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
    }
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    mock.json.return_value = data
    mock.text = json.dumps(data)
    return mock


def _messages() -> list[LlmMessage]:
    return [LlmMessage(role="user", content="List scene objects")]


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_missing_base_url_raises_on_complete() -> None:
    config = LlmConfig(provider=LlmProvider.OPENAI_COMPATIBLE, model="x", base_url=None)
    with pytest.raises(LlmClientError, match="base_url"):
        OpenAICompatibleClient(config).complete(_messages())


def test_empty_base_url_raises_on_complete() -> None:
    config = LlmConfig(provider=LlmProvider.OPENAI_COMPATIBLE, model="x", base_url="")
    with pytest.raises(LlmClientError, match="base_url"):
        OpenAICompatibleClient(config).complete(_messages())


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------

def test_no_api_key_still_sends_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        OpenAICompatibleClient(_make_config()).complete(_messages())
    _, kwargs = mock_post.call_args
    assert "Authorization" not in kwargs["headers"]


def test_api_key_from_env_sent_as_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-local")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        OpenAICompatibleClient(_make_config()).complete(_messages())
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer sk-local"


def test_custom_api_key_env_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_LOCAL_KEY", "sk-custom")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _make_config(api_key_env="MY_LOCAL_KEY")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        OpenAICompatibleClient(config).complete(_messages())
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer sk-custom"


def test_api_key_not_leaked_into_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        OpenAICompatibleClient(_make_config()).complete(_messages())
    _, kwargs = mock_post.call_args
    assert "sk-secret" not in json.dumps(kwargs.get("json", {}))


# ---------------------------------------------------------------------------
# Endpoint construction
# ---------------------------------------------------------------------------

def test_endpoint_built_from_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _make_config(base_url="http://myserver:8080/v1")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        OpenAICompatibleClient(config).complete(_messages())
    url = mock_post.call_args[0][0]
    assert url == "http://myserver:8080/v1/chat/completions"


def test_trailing_slash_stripped_from_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _make_config(base_url="http://myserver/v1/")
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        OpenAICompatibleClient(config).complete(_messages())
    url = mock_post.call_args[0][0]
    assert url == "http://myserver/v1/chat/completions"


# ---------------------------------------------------------------------------
# Successful responses
# ---------------------------------------------------------------------------

def test_complete_returns_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("httpx.post", return_value=_make_response(content="Pong")):
        result = OpenAICompatibleClient(_make_config()).complete(_messages())
    assert result.content == "Pong"
    assert result.tool_calls == []


def test_complete_parses_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    raw_tc = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "create_cube", "arguments": '{"size": 2}'},
        }
    ]
    with patch("httpx.post", return_value=_make_response(content=None, tool_calls=raw_tc)):
        result = OpenAICompatibleClient(_make_config()).complete(_messages())
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "create_cube"
    assert tc.arguments == {"size": 2}


def test_complete_passes_tools_in_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tools = [{"type": "function", "function": {"name": "get_scene_info", "parameters": {}}}]
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        OpenAICompatibleClient(_make_config()).complete(_messages(), tools=tools)
    assert mock_post.call_args[1]["json"]["tools"] == tools


def test_complete_omits_tools_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        OpenAICompatibleClient(_make_config()).complete(_messages(), tools=None)
    assert "tools" not in mock_post.call_args[1]["json"]


def test_custom_timeout_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        OpenAICompatibleClient(_make_config()).complete(_messages(), timeout_sec=30)
    assert mock_post.call_args[1]["timeout"] == 30


def test_extra_headers_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _make_config(extra_headers={"X-Org": "bench"})
    with patch("httpx.post", return_value=_make_response()) as mock_post:
        OpenAICompatibleClient(config).complete(_messages())
    assert mock_post.call_args[1]["headers"]["X-Org"] == "bench"


def test_usage_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    usage = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
    with patch("httpx.post", return_value=_make_response(usage=usage)):
        result = OpenAICompatibleClient(_make_config()).complete(_messages())
    assert result.usage is not None
    assert result.usage.total_tokens == 8


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def _error_response(status_code: int, message: str = "error") -> MagicMock:
    data = {"error": {"message": message}}
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    mock.json.return_value = data
    mock.text = json.dumps(data)
    return mock


def test_401_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("httpx.post", return_value=_error_response(401)):
        with pytest.raises(LlmClientError, match="401"):
            OpenAICompatibleClient(_make_config()).complete(_messages())


def test_429_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("httpx.post", return_value=_error_response(429)):
        with pytest.raises(LlmClientError, match="429"):
            OpenAICompatibleClient(_make_config()).complete(_messages())


def test_4xx_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("httpx.post", return_value=_error_response(400)):
        with pytest.raises(LlmClientError, match="400"):
            OpenAICompatibleClient(_make_config()).complete(_messages())


def test_5xx_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("httpx.post", return_value=_error_response(500)):
        with pytest.raises(LlmClientError, match="500"):
            OpenAICompatibleClient(_make_config()).complete(_messages())


def test_timeout_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("httpx.post", side_effect=httpx.TimeoutException("timed out")):
        with pytest.raises(LlmClientError, match="timed out"):
            OpenAICompatibleClient(_make_config()).complete(_messages())


def test_connect_error_raises_llm_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(LlmClientError, match="refused"):
            OpenAICompatibleClient(_make_config()).complete(_messages())


def test_empty_choices_raises_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.json.return_value = {"choices": [], "model": "x"}
    with patch("httpx.post", return_value=mock):
        with pytest.raises(LlmResponseParseError, match="No choices"):
            OpenAICompatibleClient(_make_config()).complete(_messages())


# ---------------------------------------------------------------------------
# Real API test (disabled by default)
# ---------------------------------------------------------------------------

@pytest.mark.llm
@pytest.mark.api_e2e
def test_real_openai_compatible_completion() -> None:
    import os
    base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL")
    if not base_url:
        pytest.skip("OPENAI_COMPATIBLE_BASE_URL not set")
    config = _make_config(base_url=base_url, model=os.environ.get("OPENAI_COMPATIBLE_MODEL", "gpt-4o-mini"))
    result = OpenAICompatibleClient(config).complete(
        [LlmMessage(role="user", content="Say hi in one word.")]
    )
    assert result.content is not None
