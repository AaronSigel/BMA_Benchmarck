from __future__ import annotations

from benchmark.agent.errors import RemoteAgentError, RemoteAgentTimeoutError
from benchmark.agent.remote.base import RemoteAgentRequest, RemoteAgentResponse


class MockRemoteAgentClient:
    """Deterministic remote agent client for tests without external infrastructure."""

    def __init__(
        self,
        response: RemoteAgentResponse | None = None,
        *,
        error: str | Exception | None = None,
        timeout: bool = False,
    ) -> None:
        self.response = response or RemoteAgentResponse(ok=True)
        self.error = error
        self.timeout = timeout
        self.requests: list[RemoteAgentRequest] = []

    def run_task(self, request: RemoteAgentRequest) -> RemoteAgentResponse:
        self.requests.append(request)
        if self.timeout:
            raise RemoteAgentTimeoutError("Mock remote agent timeout")
        if self.error is not None:
            if isinstance(self.error, Exception):
                raise self.error
            raise RemoteAgentError(self.error)
        return self.response
