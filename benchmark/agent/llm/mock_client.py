from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from benchmark.agent.errors import AgentTimeoutError, LlmClientError
from benchmark.agent.llm.base import LlmMessage, LlmResponse


@dataclass(frozen=True)
class MockLlmCall:
    messages: list[LlmMessage]
    tools: list[dict[str, Any]] | None
    timeout_sec: int | float | None


class MockLlmClient:
    """Deterministic LLM client for strategy tests without external APIs."""

    def __init__(
        self,
        responses: Sequence[LlmResponse | Exception] | None = None,
        *,
        error: str | Exception | None = None,
        timeout: bool = False,
    ) -> None:
        self._responses = list(responses or [])
        self._error = error
        self._timeout = timeout
        self._index = 0
        self.calls: list[MockLlmCall] = []

    @property
    def remaining_responses(self) -> int:
        return max(len(self._responses) - self._index, 0)

    def complete(
        self,
        messages: list[LlmMessage],
        tools: list[dict[str, Any]] | None = None,
        timeout_sec: int | float | None = None,
    ) -> LlmResponse:
        self.calls.append(
            MockLlmCall(messages=list(messages), tools=tools, timeout_sec=timeout_sec)
        )

        if self._timeout:
            raise AgentTimeoutError("Mock LLM timeout")
        if self._error is not None:
            if isinstance(self._error, Exception):
                raise self._error
            raise LlmClientError(self._error)
        if self._index >= len(self._responses):
            raise LlmClientError("MockLlmClient response sequence exhausted")

        response = self._responses[self._index]
        self._index += 1
        if isinstance(response, Exception):
            raise response
        return response
