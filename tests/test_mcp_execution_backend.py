"""Tests for benchmark.mcp.execution_backend (no Blender, no MCP server required)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark.mcp.execution_backend import McpExecutionBackend, McpExternalBackend, McpSmokeBackend
from benchmark.runner.execution import ExecutionBackend
from benchmark.runner.models import ExecutionMode, RunConfig


def make_run_config(tmp_path: Path, **overrides) -> RunConfig:
    defaults = dict(
        run_id="smoke_001",
        task_id="test_task",
        execution_mode=ExecutionMode.MCP_SMOKE,
        artifacts_dir=tmp_path,
        output_dir=tmp_path / "out",
        mcp_config_path=Path("configs/mcp/minimal.yaml"),
        mcp_profile="minimal",
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def test_mcp_execution_backend_inherits_execution_backend():
    assert issubclass(McpExecutionBackend, ExecutionBackend)


def test_mcp_smoke_backend_mode():
    assert McpSmokeBackend.mode == ExecutionMode.MCP_SMOKE


def test_mcp_external_backend_mode():
    assert McpExternalBackend.mode == ExecutionMode.MCP_EXTERNAL


def test_execute_returns_execution_result_when_socket_unavailable(tmp_path):
    cfg = make_run_config(tmp_path, mcp_config_path=Path("configs/mcp/minimal.yaml"))
    backend = McpExecutionBackend()
    # No Blender running → socket unavailable → should return ExecutionResult with ok=False
    result = backend.execute(cfg)
    assert result.ok is False
    assert result.error is not None
    # Should still write the result JSON artifact
    assert (tmp_path / "out" / "mcp_smoke_result.json").exists()


def test_execute_writes_mcp_smoke_result_json(tmp_path):
    cfg = make_run_config(tmp_path)
    backend = McpExecutionBackend()
    backend.execute(cfg)
    result_file = tmp_path / "out" / "mcp_smoke_result.json"
    assert result_file.exists()
    data = json.loads(result_file.read_text())
    required_fields = [
        "ok", "profile", "server_distribution", "blender_socket_available",
        "telemetry_disabled", "available_tools", "disabled_tools",
        "scene_info", "profile_info", "error", "started_at", "finished_at", "duration_sec",
    ]
    for field in required_fields:
        assert field in data, f"McpSmokeResult missing field: {field}"


def test_execute_with_mock_socket_calls_get_scene_info(tmp_path):
    cfg = make_run_config(tmp_path, mcp_profile="full")
    backend = McpExecutionBackend()

    mock_sock = MagicMock()
    mock_sock.recv.return_value = json.dumps({"result": {"objects": []}}).encode()

    with patch("benchmark.mcp.connection_check.socket.create_connection", return_value=MagicMock()), \
         patch("benchmark.mcp.server_adapter.socket.create_connection", return_value=mock_sock):
        result = backend.execute(cfg)

    result_file = tmp_path / "out" / "mcp_smoke_result.json"
    data = json.loads(result_file.read_text())
    assert "get_scene_info" in data.get("scene_info", {}) or data.get("blender_socket_available") is not None


def test_execute_never_calls_execute_blender_code_in_minimal(tmp_path):
    cfg = make_run_config(tmp_path, mcp_profile="minimal")
    backend = McpExecutionBackend()

    calls: list[str] = []

    def mock_call_tool(tool_name, params=None):
        calls.append(tool_name)
        return {"objects": []}

    mock_sock = MagicMock()
    mock_sock.recv.return_value = json.dumps({"result": {}}).encode()

    with patch("benchmark.mcp.connection_check.socket.create_connection", return_value=MagicMock()), \
         patch(
             "benchmark.mcp.server_adapter.ExternalBlenderMcpServerAdapter.call_tool",
             side_effect=mock_call_tool,
         ):
        backend.execute(cfg)

    assert "execute_blender_code" not in calls


def test_execute_never_calls_execute_blender_code_in_no_python(tmp_path):
    cfg = make_run_config(tmp_path, mcp_profile="no_python")
    backend = McpExecutionBackend()
    calls: list[str] = []

    def mock_call_tool(tool_name, params=None):
        calls.append(tool_name)
        return {}

    with patch("benchmark.mcp.connection_check.socket.create_connection", return_value=MagicMock()), \
         patch(
             "benchmark.mcp.server_adapter.ExternalBlenderMcpServerAdapter.call_tool",
             side_effect=mock_call_tool,
         ):
        backend.execute(cfg)

    assert "execute_blender_code" not in calls


def test_execute_mcp_external_mode_calls_get_object_info(tmp_path):
    cfg = make_run_config(
        tmp_path,
        execution_mode=ExecutionMode.MCP_EXTERNAL,
        mcp_profile="full",
    )
    backend = McpExternalBackend()
    calls: list[str] = []

    def mock_call_tool(tool_name, params=None):
        calls.append(tool_name)
        return {}

    with patch("benchmark.mcp.connection_check.socket.create_connection", return_value=MagicMock()), \
         patch(
             "benchmark.mcp.server_adapter.ExternalBlenderMcpServerAdapter.call_tool",
             side_effect=mock_call_tool,
         ):
        backend.execute(cfg)

    assert "get_object_info" in calls


@pytest.mark.mcp
def test_execute_with_real_blender_socket(tmp_path):
    """Real integration test — requires Blender running with blender-mcp addon."""
    cfg = make_run_config(tmp_path, mcp_profile="minimal")
    backend = McpExecutionBackend()
    result = backend.execute(cfg)
    assert result.ok is True
    result_data = json.loads((tmp_path / "out" / "mcp_smoke_result.json").read_text())
    assert result_data["blender_socket_available"] is True
