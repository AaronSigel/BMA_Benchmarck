from __future__ import annotations

import json
import os
from typing import Any

import httpx

from benchmark.agent.errors import LlmClientError, LlmResponseParseError
from benchmark.agent.llm.base import LlmMessage, LlmResponse, LlmToolCall, LlmUsage
from benchmark.agent.models import LlmConfig

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
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

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [_message_to_dict(m) for m in messages],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }

        try:
            response = httpx.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise LlmClientError(f"OpenRouter request timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise LlmClientError(f"OpenRouter request failed: {exc}") from exc

        if response.status_code >= 400:
            _raise_for_status(response)

        try:
            data = response.json()
        except Exception as exc:
            raise LlmResponseParseError(
                f"Failed to parse OpenRouter response: {exc}",
                raw_response=response.text,
            ) from exc

        return _parse_response(data, provider_name="openrouter")

    def _resolve_api_key(self) -> str:
        env_var = self.config.api_key_env or "OPENROUTER_API_KEY"
        api_key = os.environ.get(env_var, "")
        if not api_key:
            raise LlmClientError(
                f"OpenRouter API key not found in environment variable {env_var!r}"
            )
        return api_key


def _message_to_dict(message: LlmMessage) -> dict[str, Any]:
    result: dict[str, Any] = {"role": message.role}
    if message.content is not None:
        result["content"] = message.content
    if message.name is not None:
        result["name"] = message.name
    if message.tool_call_id is not None:
        result["tool_call_id"] = message.tool_call_id
    return result


def _parse_response(data: dict[str, Any], provider_name: str | None = None) -> LlmResponse:
    choices = data.get("choices", [])
    if not choices:
        raise LlmResponseParseError("No choices in OpenRouter response", raw_response=data)

    choice = choices[0]
    message = choice.get("message", {})
    content = message.get("content")
    finish_reason = choice.get("finish_reason")

    tool_calls: list[LlmToolCall] = []
    for tc in message.get("tool_calls") or []:
        func = tc.get("function", {})
        name = func.get("name", "")
        try:
            arguments = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            arguments = {}
        tool_calls.append(LlmToolCall(
            id=tc.get("id"),
            name=name,
            arguments=arguments,
            raw=tc,
        ))

    usage_data = data.get("usage")
    usage = None
    if usage_data:
        usage = LlmUsage(
            prompt_tokens=usage_data.get("prompt_tokens"),
            completion_tokens=usage_data.get("completion_tokens"),
            total_tokens=usage_data.get("total_tokens"),
            cost=usage_data.get("cost"),
            provider_name=provider_name,
            metadata={"raw_usage": usage_data},
        )

    return LlmResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        model=data.get("model"),
        raw_response=data,
    )


def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    try:
        body = response.json()
        detail = body.get("error", {}).get("message") or str(body)
    except Exception:
        detail = response.text
    if status == 401:
        raise LlmClientError(f"OpenRouter authentication failed (401): {detail}")
    if status == 429:
        raise LlmClientError(f"OpenRouter rate limit exceeded (429): {detail}")
    if 400 <= status < 500:
        raise LlmClientError(f"OpenRouter client error ({status}): {detail}")
    raise LlmClientError(f"OpenRouter server error ({status}): {detail}")
