from __future__ import annotations

import os
from typing import Any

import httpx

from benchmark.agent.errors import LlmClientError, LlmResponseParseError
from benchmark.agent.llm.base import LlmMessage, LlmResponse, LlmToolCall, LlmUsage
from benchmark.agent.models import LlmConfig

_DEFAULT_BASE_URL = "https://api.anthropic.com"
_DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicClient:
    """Direct HTTP client for the Anthropic Messages API."""

    def __init__(self, config: LlmConfig) -> None:
        self.config = config

    def complete(
        self,
        messages: list[LlmMessage],
        tools: list[dict[str, Any]] | None = None,
        timeout_sec: int | float | None = None,
    ) -> LlmResponse:
        api_key = self._resolve_api_key()
        base_url = (self.config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        timeout = timeout_sec if timeout_sec is not None else self.config.timeout_sec

        system, anthropic_messages = _split_system_messages(messages)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": anthropic_messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = _adapt_tools(tools)

        headers = {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
            **self.config.extra_headers,
        }

        try:
            response = httpx.post(
                f"{base_url}/v1/messages",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise LlmClientError(f"Anthropic request timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise LlmClientError(f"Anthropic request failed: {exc}") from exc

        if response.status_code >= 400:
            _raise_for_status(response)

        try:
            data = response.json()
        except Exception as exc:
            raise LlmResponseParseError(
                f"Failed to parse Anthropic response: {exc}",
                raw_response=response.text,
            ) from exc

        return _parse_response(data)

    def _resolve_api_key(self) -> str:
        env_var = self.config.api_key_env or _DEFAULT_API_KEY_ENV
        api_key = os.environ.get(env_var, "")
        if not api_key:
            raise LlmClientError(
                f"Anthropic API key not found in environment variable {env_var!r}"
            )
        return api_key


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

def _split_system_messages(
    messages: list[LlmMessage],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract system prompt and convert remaining messages to Anthropic format."""
    system_parts: list[str] = []
    anthropic_messages: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "system":
            if msg.content:
                system_parts.append(msg.content)
            continue

        if msg.role == "tool" or msg.tool_call_id is not None:
            anthropic_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "",
                        "content": msg.content or "",
                    }
                ],
            })
            continue

        anthropic_messages.append({
            "role": msg.role,
            "content": msg.content or "",
        })

    system = "\n".join(system_parts) if system_parts else None
    return system, anthropic_messages


# ---------------------------------------------------------------------------
# Tool schema adaptation (OpenAI format → Anthropic format)
# ---------------------------------------------------------------------------

def _adapt_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools:
        if "function" in tool:
            func = tool["function"]
            result.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
        else:
            result.append(tool)
    return result


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(data: dict[str, Any]) -> LlmResponse:
    content_blocks = data.get("content", [])
    if not isinstance(content_blocks, list):
        raise LlmResponseParseError(
            "Unexpected Anthropic response: 'content' is not a list", raw_response=data
        )

    text_parts: list[str] = []
    tool_calls: list[LlmToolCall] = []

    for block in content_blocks:
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(LlmToolCall(
                id=block.get("id"),
                name=block.get("name", ""),
                arguments=block.get("input") or {},
                raw=block,
            ))

    content = "\n".join(text_parts) if text_parts else None
    finish_reason = data.get("stop_reason")

    usage_data = data.get("usage")
    usage = None
    if usage_data:
        usage = LlmUsage(
            prompt_tokens=usage_data.get("input_tokens"),
            completion_tokens=usage_data.get("output_tokens"),
            total_tokens=(
                (usage_data.get("input_tokens") or 0)
                + (usage_data.get("output_tokens") or 0)
            ) or None,
        )

    return LlmResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        model=data.get("model"),
        raw_response=data,
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    try:
        body = response.json()
        detail = body.get("error", {}).get("message") or str(body)
    except Exception:
        detail = response.text
    if status == 401:
        raise LlmClientError(f"Anthropic authentication failed (401): {detail}")
    if status == 429:
        raise LlmClientError(f"Anthropic rate limit exceeded (429): {detail}")
    if 400 <= status < 500:
        raise LlmClientError(f"Anthropic client error ({status}): {detail}")
    raise LlmClientError(f"Anthropic server error ({status}): {detail}")
