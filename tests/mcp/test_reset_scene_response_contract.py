from __future__ import annotations

from unittest.mock import MagicMock, patch

from benchmark.mcp.config import McpServerConfig
from benchmark.mcp.server_adapter import ExternalBlenderMcpServerAdapter


def test_reset_scene_success_envelope() -> None:
    cfg = McpServerConfig(profile="full", blender_host="localhost", blender_port=9876)
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    mock_sock = MagicMock()
    mock_sock.recv.return_value = b'{"status": "success", "result": {"ok": true}}'

    with patch("benchmark.mcp.server_adapter.socket.create_connection", return_value=mock_sock):
        result = adapter.reset_scene()
    assert result["ok"] is True
    assert result["tool"] == "reset_scene"


def test_reset_scene_empty_socket_envelope() -> None:
    cfg = McpServerConfig(profile="full", blender_host="localhost", blender_port=9876)
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    mock_sock = MagicMock()
    mock_sock.recv.return_value = b""

    with patch("benchmark.mcp.server_adapter.socket.create_connection", return_value=mock_sock):
        result = adapter.reset_scene()
    assert result["ok"] is False
    assert result["error"]["type"] == "EmptySocketResponse"
