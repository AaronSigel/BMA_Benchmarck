from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator


class LlmMessage(BaseModel):
    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("role must not be empty")
        return value


class LlmToolCall(BaseModel):
    id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tool call name must not be empty")
        return value


class LlmUsage(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0.0)
    provider_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LlmResponse(BaseModel):
    content: str | None = None
    tool_calls: list[LlmToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: LlmUsage | None = None
    model: str | None = None
    raw_response: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def json_action(self) -> dict[str, Any] | None:
        """Return a fallback JSON action encoded in content, if present."""
        if self.content is None:
            return None
        try:
            parsed = json.loads(self.content)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def has_action(self) -> bool:
        return bool(self.tool_calls) or self.json_action() is not None


_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)


def parse_json_from_content(content: str | None) -> Any:
    """Извлекает JSON из content: fenced blocks, чистый JSON или встроенный фрагмент."""
    if content is None:
        return None
    text = content.strip()
    fence = _JSON_FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    return None


@runtime_checkable
class LlmClient(Protocol):
    def complete(
        self,
        messages: list[LlmMessage],
        tools: list[dict[str, Any]] | None = None,
        timeout_sec: int | float | None = None,
    ) -> LlmResponse:
        """Complete a chat-style request and return a provider-neutral response."""
