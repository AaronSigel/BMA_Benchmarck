"""Contract tests for bma_create_camera_look_at socket responses."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from benchmark.mcp.config import McpServerConfig
from benchmark.mcp.server_adapter import ExternalBlenderMcpServerAdapter


def _make_config(**overrides):
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


def _socket_mock(response: dict) -> MagicMock:
    mock_sock = MagicMock()
    mock_sock.recv.return_value = json.dumps(response).encode()
    return mock_sock


def test_camera_look_at_returns_json_success() -> None:
    adapter = ExternalBlenderMcpServerAdapter(_make_config())
    response = {
        "status": "success",
        "result": {
            "camera_name": "Camera",
            "location": [0.0, -5.0, 2.0],
            "target": [0.0, 0.0, 1.0],
            "is_active": True,
        },
    }
    with patch("benchmark.mcp.server_adapter.socket.create_connection", return_value=_socket_mock(response)):
        envelope = adapter.call_tool(
            "bma_create_camera_look_at",
            {"name": "Camera", "location": [0, -5, 2], "target": [0, 0, 1]},
        )
    assert envelope["ok"] is True
    assert envelope["result"]["camera_name"] == "Camera"
    assert envelope["result"]["set_active"] is True


def test_camera_look_at_missing_target_returns_json_error() -> None:
    adapter = ExternalBlenderMcpServerAdapter(_make_config())
    response = {
        "status": "error",
        "error": {
            "type": "ObjectNotFound",
            "message": "Target object not found",
            "camera_name": "Camera",
            "target": "MissingObject",
        },
    }
    with patch("benchmark.mcp.server_adapter.socket.create_connection", return_value=_socket_mock(response)):
        envelope = adapter.call_tool(
            "bma_create_camera_look_at",
            {"name": "Camera", "location": [0, -5, 2], "target_object_name": "MissingObject"},
        )
    assert envelope["ok"] is False
    assert envelope["error"]["type"] == "ObjectNotFound"


def test_empty_socket_response_classified() -> None:
    adapter = ExternalBlenderMcpServerAdapter(_make_config())
    mock_sock = MagicMock()
    mock_sock.recv.return_value = b""
    with patch("benchmark.mcp.server_adapter.socket.create_connection", return_value=mock_sock), patch.object(
        adapter,
        "execute_code_unrestricted",
        return_value={"warning": "execute_code failed: python disabled"},
    ):
        envelope = adapter.call_tool(
            "bma_create_camera_look_at",
            {"name": "Camera", "location": [0, -5, 2], "target": [0, 0, 1]},
        )
    assert envelope["ok"] is False
    assert envelope["error"]["type"] == "CameraLookAtFailed"
    assert envelope["result"]["failure_stage"] == "execute_code_fallback"


def test_empty_socket_triggers_execute_code_fallback() -> None:
    adapter = ExternalBlenderMcpServerAdapter(_make_config())
    mock_sock = MagicMock()
    mock_sock.recv.return_value = b""
    fallback_payload = {
        "stdout": json.dumps(
            {
                "ok": True,
                "tool": "bma_create_camera_look_at",
                "result": {
                    "camera_name": "Camera",
                    "location": [0.0, -5.0, 2.0],
                    "target": [0.0, 0.0, 1.0],
                    "set_active": True,
                },
            }
        )
    }
    with patch("benchmark.mcp.server_adapter.socket.create_connection", return_value=mock_sock), patch.object(
        adapter,
        "execute_code_unrestricted",
        return_value=fallback_payload,
    ) as execute_mock:
        envelope = adapter.call_tool(
            "bma_create_camera_look_at",
            {"name": "Camera", "location": [0, -5, 2], "target": [0, 0, 1]},
        )
    assert envelope["ok"] is True
    assert envelope["result"]["camera_name"] == "Camera"
    assert envelope["result"]["set_active"] is True
    execute_mock.assert_called_once()
