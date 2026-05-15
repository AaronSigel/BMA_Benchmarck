"""Tests for benchmark.mcp.connection_check (no Blender, no MCP server required)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from benchmark.mcp.config import McpServerConfig
from benchmark.mcp.connection_check import (
    ConnectionCheckResult,
    check_blender_socket,
    check_mcp_server_config,
    is_blender_socket_available,
)
from benchmark.mcp.errors import BlenderSocketUnavailableError


def make_config(**overrides) -> McpServerConfig:
    defaults = dict(
        blender_host="localhost",
        blender_port=9876,
        profile="full",
        disable_telemetry=True,
        startup_timeout_sec=2,
    )
    defaults.update(overrides)
    return McpServerConfig(**defaults)


# ---------------------------------------------------------------------------
# check_blender_socket
# ---------------------------------------------------------------------------

def test_check_blender_socket_raises_when_refused():
    with pytest.raises(BlenderSocketUnavailableError):
        check_blender_socket("localhost", 19999, timeout_sec=0.2)


def test_check_blender_socket_succeeds_with_mock():
    mock_sock = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    with patch("benchmark.mcp.connection_check.socket.create_connection", return_value=mock_sock):
        check_blender_socket("localhost", 9876, timeout_sec=1.0)
    mock_sock.close.assert_called_once()


def test_check_blender_socket_oserror_raises():
    with patch(
        "benchmark.mcp.connection_check.socket.create_connection",
        side_effect=OSError("refused"),
    ):
        with pytest.raises(BlenderSocketUnavailableError, match="refused"):
            check_blender_socket("localhost", 9876)


# ---------------------------------------------------------------------------
# is_blender_socket_available
# ---------------------------------------------------------------------------

def test_is_blender_socket_available_false_when_refused():
    cfg = make_config(blender_port=19999, startup_timeout_sec=1)
    assert is_blender_socket_available(cfg) is False


def test_is_blender_socket_available_true_with_mock():
    cfg = make_config()
    mock_sock = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    with patch("benchmark.mcp.connection_check.socket.create_connection", return_value=mock_sock):
        assert is_blender_socket_available(cfg) is True


# ---------------------------------------------------------------------------
# check_mcp_server_config
# ---------------------------------------------------------------------------

def test_check_mcp_server_config_socket_unavailable():
    cfg = make_config(blender_port=19999, startup_timeout_sec=1)
    result = check_mcp_server_config(cfg)
    assert isinstance(result, ConnectionCheckResult)
    assert result.blender_socket_available is False
    assert result.ok is False
    assert result.host == "localhost"
    assert result.port == 19999
    assert result.profile == "full"
    assert result.telemetry_disabled is True
    assert len(result.issues) > 0


def test_check_mcp_server_config_socket_available():
    cfg = make_config()
    mock_sock = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    with patch("benchmark.mcp.connection_check.socket.create_connection", return_value=mock_sock):
        result = check_mcp_server_config(cfg)
    assert result.blender_socket_available is True
    assert result.issues == []
    assert result.ok is True


def test_check_mcp_server_config_bad_profile():
    cfg = make_config(profile="nonexistent_profile")
    result = check_mcp_server_config(cfg)
    assert result.ok is False
    assert any("Unknown profile" in issue for issue in result.issues)


def test_check_mcp_server_config_fork_without_source():
    cfg = make_config(server_distribution="fork", package_source=None)
    result = check_mcp_server_config(cfg)
    assert any("package_source" in issue for issue in result.issues)


def test_connection_check_result_ok_property():
    good = ConnectionCheckResult(
        blender_socket_available=True,
        host="h", port=9876, profile="full", telemetry_disabled=True, issues=[],
    )
    assert good.ok is True

    bad = ConnectionCheckResult(
        blender_socket_available=False,
        host="h", port=9876, profile="full", telemetry_disabled=True,
        issues=["socket unavailable"],
    )
    assert bad.ok is False
