"""Tests for benchmark.mcp.server_adapter (no Blender, no MCP server required)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from benchmark.mcp.config import McpServerConfig
from benchmark.mcp.errors import (
    BlenderSocketUnavailableError,
    McpExecutionError,
    McpServerStartError,
    ToolDisabledError,
)
from benchmark.mcp.profiles import McpProfile
from benchmark.mcp.server_adapter import ExternalBlenderMcpServerAdapter


def make_config(**overrides) -> McpServerConfig:
    defaults = dict(
        server_distribution="upstream",
        blender_host="localhost",
        blender_port=9876,
        profile="full",
        disable_telemetry=True,
        startup_timeout_sec=5,
        request_timeout_sec=10,
    )
    defaults.update(overrides)
    return McpServerConfig(**defaults)


# ---------------------------------------------------------------------------
# build_command
# ---------------------------------------------------------------------------

def test_build_command_upstream():
    cfg = make_config(server_distribution="upstream")
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    cmd = adapter.build_command()
    assert cmd == ["uvx", "blender-mcp"]


def test_build_command_fork_with_git_url():
    cfg = make_config(
        server_distribution="fork",
        package_source="git+https://github.com/AaronSigel/blender-mcp-bma",
    )
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    cmd = adapter.build_command()
    assert cmd == [
        "uvx", "--from",
        "git+https://github.com/AaronSigel/blender-mcp-bma",
        "blender-mcp",
    ]


def test_build_command_local():
    cfg = make_config(server_distribution="local", package_source="./vendor/bma")
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    cmd = adapter.build_command()
    assert cmd == ["uvx", "--from", "./vendor/bma", "blender-mcp"]


def test_build_command_fork_without_package_source_raises():
    cfg = make_config(server_distribution="fork", package_source=None)
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    with pytest.raises(McpServerStartError, match="package_source"):
        adapter.build_command()


# ---------------------------------------------------------------------------
# build_env
# ---------------------------------------------------------------------------

def test_build_env_contains_bma_vars():
    cfg = make_config(profile="minimal")
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    env = adapter.build_env()
    assert env["BMA_MCP_PROFILE"] == "minimal"
    assert "DISABLE_TELEMETRY" in env


# ---------------------------------------------------------------------------
# start / stop / is_running
# ---------------------------------------------------------------------------

def test_start_returns_popen():
    cfg = make_config()
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    mock_proc = MagicMock()
    mock_proc.pid = 1234

    with patch("benchmark.mcp.server_adapter.subprocess.Popen", return_value=mock_proc):
        proc = adapter.start()

    assert proc is mock_proc


def test_start_file_not_found_raises():
    cfg = make_config()
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    with patch(
        "benchmark.mcp.server_adapter.subprocess.Popen",
        side_effect=FileNotFoundError("uvx not found"),
    ):
        with pytest.raises(McpServerStartError, match="not found"):
            adapter.start()


def test_is_running_true_when_poll_none():
    cfg = make_config()
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    assert adapter.is_running(mock_proc) is True


def test_is_running_false_when_exited():
    cfg = make_config()
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 0
    assert adapter.is_running(mock_proc) is False


def test_stop_terminates_running_process():
    cfg = make_config()
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.pid = 999

    with patch("benchmark.mcp.server_adapter.os.getpgid", return_value=999), \
         patch("benchmark.mcp.server_adapter.os.killpg") as mock_kill, \
         patch.object(mock_proc, "wait"):
        adapter.stop(mock_proc)

    assert mock_kill.called


# ---------------------------------------------------------------------------
# call_tool (socket)
# ---------------------------------------------------------------------------

def _make_socket_mock(response: dict):
    mock_sock = MagicMock()
    mock_sock.recv.return_value = json.dumps(response).encode()
    return mock_sock


def test_call_tool_returns_result():
    cfg = make_config(profile="full")
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    mock_sock = _make_socket_mock({"result": {"objects": []}})

    with patch("benchmark.mcp.server_adapter.socket.create_connection", return_value=mock_sock):
        result = adapter.call_tool("get_scene_info")

    assert result == {"objects": []}


def test_call_tool_disabled_in_profile_raises():
    cfg = make_config(profile="minimal")
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    with pytest.raises(ToolDisabledError):
        adapter.call_tool("execute_blender_code")


def test_call_tool_socket_error_raises():
    cfg = make_config(profile="full")
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    with patch(
        "benchmark.mcp.server_adapter.socket.create_connection",
        side_effect=OSError("refused"),
    ):
        with pytest.raises(BlenderSocketUnavailableError):
            adapter.call_tool("get_scene_info")


def test_call_tool_error_response_raises():
    cfg = make_config(profile="full")
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    mock_sock = _make_socket_mock({"status": "error", "error": "boom"})

    with patch("benchmark.mcp.server_adapter.socket.create_connection", return_value=mock_sock):
        with pytest.raises(McpExecutionError, match="boom"):
            adapter.call_tool("get_scene_info")


def test_list_tools_returns_profile_tools():
    cfg = make_config(profile="minimal")
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    tools = adapter.list_tools()
    assert "get_scene_info" in tools
    assert "execute_blender_code" not in tools
