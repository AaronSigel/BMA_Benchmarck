from __future__ import annotations

import datetime
from typing import Any, Protocol, runtime_checkable

from benchmark.agent.errors import ToolInvocationError
from benchmark.agent.models import ToolCallRequest, ToolCallResult, ToolCallStatus
from benchmark.agent.tool_context import AgentToolContext, ToolSchemaProvider
from benchmark.mcp.errors import McpLayerError
from benchmark.mcp.profiles import McpProfile
from benchmark.mcp.server_adapter import ExternalBlenderMcpServerAdapter
from benchmark.mcp.tool_registry import McpToolRegistry


@runtime_checkable
class ToolExecutor(Protocol):
    def call_tool(
        self,
        tool_name: str | ToolCallRequest,
        arguments: dict[str, Any] | AgentToolContext | None = None,
    ) -> ToolCallResult:
        """Execute one tool call and return a normalized result."""

    def assert_tool_allowed(self, tool_name: str) -> None:
        """Raise ToolInvocationError if a tool is blocked."""

    def normalize_tool_result(self, result: Any) -> dict[str, Any]:
        """Normalize arbitrary tool output into a JSON-compatible mapping."""


class NoopToolSchemaProvider(ToolSchemaProvider):
    def list_tool_schemas(self, profile: McpProfile | str = McpProfile.MINIMAL) -> list[dict[str, Any]]:
        return []


class MockToolExecutor:
    def __init__(
        self,
        results: dict[str, Any] | None = None,
        *,
        allowed_tools: list[str] | None = None,
        errors: dict[str, str | Exception] | None = None,
    ) -> None:
        self.results = results or {}
        self.allowed_tools = allowed_tools or []
        self.errors = errors or {}
        self.calls: list[ToolCallRequest] = []

    def assert_tool_allowed(self, tool_name: str) -> None:
        if self.allowed_tools and tool_name not in self.allowed_tools:
            raise ToolInvocationError(f"Tool is not allowed: {tool_name}", tool_name=tool_name)

    def call_tool(
        self,
        tool_name: str | ToolCallRequest,
        arguments: dict[str, Any] | AgentToolContext | None = None,
    ) -> ToolCallResult:
        request = _coerce_request(tool_name, arguments)
        started_at = datetime.datetime.now(datetime.timezone.utc)
        try:
            self.assert_tool_allowed(request.name)
            if request.name in self.errors:
                error = self.errors[request.name]
                if isinstance(error, Exception):
                    raise error
                raise ToolInvocationError(error, tool_name=request.name)
            raw_result = self.results.get(request.name, {})
            result = self.normalize_tool_result(raw_result)
            status = ToolCallStatus.SUCCEEDED
            error_message = None
        except ToolInvocationError as error:
            result = None
            status = ToolCallStatus.FAILED
            error_message = str(error)
        finished_at = datetime.datetime.now(datetime.timezone.utc)
        self.calls.append(request)
        return ToolCallResult(
            name=request.name,
            status=status,
            result=result,
            error=error_message,
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=(finished_at - started_at).total_seconds(),
        )

    def normalize_tool_result(self, result: Any) -> dict[str, Any]:
        return normalize_tool_result(result)


class McpToolExecutor:
    def __init__(
        self,
        adapter: ExternalBlenderMcpServerAdapter,
        *,
        profile: McpProfile | str,
        registry: McpToolRegistry | None = None,
    ) -> None:
        self.adapter = adapter
        self.profile = _coerce_profile(profile)
        self.registry = registry or McpToolRegistry()

    def assert_tool_allowed(self, tool_name: str) -> None:
        try:
            self.registry.assert_tool_allowed(tool_name, self.profile)
        except McpLayerError as error:
            raise ToolInvocationError(str(error), tool_name=tool_name) from error

    def call_tool(
        self,
        tool_name: str | ToolCallRequest,
        arguments: dict[str, Any] | AgentToolContext | None = None,
    ) -> ToolCallResult:
        request = _coerce_request(tool_name, arguments)
        started_at = datetime.datetime.now(datetime.timezone.utc)
        try:
            self.assert_tool_allowed(request.name)
            raw_result = self.adapter.call_tool(request.name, request.arguments)
            result = self.normalize_tool_result(raw_result)
            status = ToolCallStatus.SUCCEEDED
            error_message = None
        except ToolInvocationError as error:
            result = None
            status = ToolCallStatus.FAILED
            error_message = str(error)
        except McpLayerError as error:
            result = None
            status = ToolCallStatus.FAILED
            error_message = str(error)
        finished_at = datetime.datetime.now(datetime.timezone.utc)
        return ToolCallResult(
            name=request.name,
            status=status,
            result=result,
            error=error_message,
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=(finished_at - started_at).total_seconds(),
        )

    def normalize_tool_result(self, result: Any) -> dict[str, Any]:
        return normalize_tool_result(result)


class NoopToolExecutor(MockToolExecutor):
    def __init__(self) -> None:
        super().__init__(errors={"*": "No tool executor is configured"})

    def call_tool(
        self,
        tool_name: str | ToolCallRequest,
        arguments: dict[str, Any] | AgentToolContext | None = None,
    ) -> ToolCallResult:
        request = _coerce_request(tool_name, arguments)
        raise ToolInvocationError("No tool executor is configured", tool_name=request.name)


def normalize_tool_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if result is None:
        return {"value": None}
    if isinstance(result, list):
        return {"items": result}
    return {"value": result}


def _coerce_request(
    tool_name: str | ToolCallRequest,
    arguments: dict[str, Any] | AgentToolContext | None = None,
) -> ToolCallRequest:
    if isinstance(tool_name, ToolCallRequest):
        return tool_name
    if isinstance(arguments, AgentToolContext) or arguments is None:
        args: dict[str, Any] = {}
    else:
        args = arguments
    return ToolCallRequest(name=tool_name, arguments=args)


def _coerce_profile(profile: McpProfile | str) -> McpProfile:
    if isinstance(profile, McpProfile):
        return profile
    return McpProfile(profile)
