from typing import Any

import pytest

from benchmark.agent.errors import ToolInvocationError
from benchmark.agent.models import AgentStep, AgentStepType, ToolCallStatus
from benchmark.agent.tool_executor import MockToolExecutor, McpToolExecutor, normalize_tool_result
from benchmark.mcp.errors import McpExecutionError


class FakeMcpAdapter:
    def __init__(self, results: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.results = results or {}
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call_tool(self, tool_name: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append((tool_name, params or {}))
        if self.error is not None:
            raise self.error
        return self.results.get(tool_name, {})


def test_mock_tool_executor_works_without_mcp() -> None:
    executor = MockToolExecutor(results={"get_scene_info": {"objects": []}})

    result = executor.call_tool("get_scene_info", {})

    assert result.status == ToolCallStatus.SUCCEEDED
    assert result.result == {"objects": []}
    assert executor.calls[0].name == "get_scene_info"


def test_mock_tool_executor_blocks_disallowed_tool() -> None:
    executor = MockToolExecutor(allowed_tools=["get_scene_info"])

    result = executor.call_tool("execute_blender_code", {"code": "print(1)"})

    assert result.status == ToolCallStatus.FAILED
    assert "not allowed" in (result.error or "")


def test_mock_tool_executor_assert_tool_allowed_raises() -> None:
    executor = MockToolExecutor(allowed_tools=["get_scene_info"])

    with pytest.raises(ToolInvocationError):
        executor.assert_tool_allowed("execute_blender_code")


def test_mcp_tool_executor_checks_registry_before_calling_adapter() -> None:
    adapter = FakeMcpAdapter()
    executor = McpToolExecutor(adapter, profile="no_python")

    result = executor.call_tool("execute_blender_code", {"code": "print(1)"})

    assert result.status == ToolCallStatus.FAILED
    assert "disabled" in (result.error or "") or "not allowed" in (result.error or "")
    assert adapter.calls == []


def test_mcp_tool_executor_calls_allowed_tool_and_normalizes_result() -> None:
    adapter = FakeMcpAdapter(results={"get_scene_info": ["scene", "info"]})
    executor = McpToolExecutor(adapter, profile="no_python")

    result = executor.call_tool("get_scene_info", {})

    assert result.status == ToolCallStatus.SUCCEEDED
    assert result.result == {"items": ["scene", "info"]}
    assert adapter.calls == [("get_scene_info", {})]


def test_mcp_tool_executor_wraps_mcp_errors() -> None:
    adapter = FakeMcpAdapter(error=McpExecutionError("socket failed"))
    executor = McpToolExecutor(adapter, profile="no_python")

    result = executor.call_tool("get_scene_info", {})

    assert result.status == ToolCallStatus.FAILED
    assert "socket failed" in (result.error or "")


def test_tool_result_can_be_recorded_in_agent_step() -> None:
    executor = MockToolExecutor(results={"get_scene_info": {"objects": []}})
    result = executor.call_tool("get_scene_info", {})

    step = AgentStep(
        step_index=0,
        step_type=AgentStepType.TOOL_CALL,
        tool_name=result.name,
        tool_arguments={},
        observation=result.result,
        error=result.error,
        duration_sec=result.duration_sec,
    )

    assert step.tool_name == "get_scene_info"
    assert step.observation == {"objects": []}


def test_normalize_tool_result_returns_dict() -> None:
    assert normalize_tool_result({"ok": True}) == {"ok": True}
    assert normalize_tool_result(["a"]) == {"items": ["a"]}
    assert normalize_tool_result("text") == {"value": "text"}
    assert normalize_tool_result(None) == {"value": None}
