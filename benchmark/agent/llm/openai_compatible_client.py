from __future__ import annotations

import os
from typing import Any

import httpx

from benchmark.agent.errors import LlmClientError
from benchmark.agent.llm.base import LlmMessage, LlmResponse
from benchmark.agent.llm.openrouter_client import (
    _message_to_dict,
    _parse_response,
    _raise_for_status,
)
from benchmark.agent.models import LlmConfig

_DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"


class OpenAICompatibleClient:
    """Generic client for any OpenAI-compatible chat completions API.

    base_url is required — there is no built-in default endpoint.
    """

    def __init__(self, config: LlmConfig) -> None:
        if not config.base_url:
            raise LlmClientError(
                "OpenAICompatibleClient requires base_url to be set in LlmConfig"
            )
        self.config = config

    def complete(
        self,
        messages: list[LlmMessage],
        tools: list[dict[str, Any]] | None = None,
        timeout_sec: int | float | None = None,
    ) -> LlmResponse:
        api_key = self._resolve_api_key()
        base_url = self.config.base_url.rstrip("/")  # type: ignore[union-attr]  # checked in __init__
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

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = httpx.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise LlmClientError(f"OpenAI-compatible request timed out: {exc}") from exc
        except httpx.RequestError as exc:
            raise LlmClientError(f"OpenAI-compatible request failed: {exc}") from exc

        if response.status_code >= 400:
            _raise_for_status(response)

        try:
            data = response.json()
        except Exception as exc:
            from benchmark.agent.errors import LlmResponseParseError
            raise LlmResponseParseError(
                f"Failed to parse OpenAI-compatible response: {exc}",
                raw_response=response.text,
            ) from exc

        return _parse_response(data)

    def _resolve_api_key(self) -> str:
        env_var = self.config.api_key_env or _DEFAULT_API_KEY_ENV
        return os.environ.get(env_var, "")
