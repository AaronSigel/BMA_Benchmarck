"""Tests for bma_export_scene / server_adapter response contract.

These tests verify:
1. server_adapter classifies empty socket responses as EmptySocketResponse
2. server_adapter classifies invalid JSON as InvalidJsonResponse with raw_preview
3. ok=False results from addon are surfaced correctly
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import json

import pytest

from benchmark.mcp.server_adapter import ExternalBlenderMcpServerAdapter, _tool_envelope


# ---------------------------------------------------------------------------
# _tool_envelope tests
# ---------------------------------------------------------------------------

def test_tool_envelope_success():
    result = _tool_envelope("bma_export_scene", ok=True, result={"filepath": "/tmp/out.blend"})
    assert result["ok"] is True
    assert result["tool"] == "bma_export_scene"
    assert result["error"] is None


def test_tool_envelope_error_includes_type_and_message():
    result = _tool_envelope(
        "bma_export_scene", ok=False, result=None,
        error_type="ExportFailed", error_message="write error"
    )
    assert result["ok"] is False
    assert result["error"]["type"] == "ExportFailed"
    assert "write error" in result["error"]["message"]


# ---------------------------------------------------------------------------
# Empty socket response → EmptySocketResponse
# ---------------------------------------------------------------------------

def _make_adapter():
    from benchmark.mcp.config import McpServerConfig
    cfg = McpServerConfig(
        profile="minimal",
        blender_host="127.0.0.1",
        blender_port=9876,
    )
    return ExternalBlenderMcpServerAdapter(cfg)


def _mock_socket_call(adapter, *, raw_bytes: bytes):
    """Patch socket layer to return raw_bytes, then call bma_export_scene."""
    with patch("benchmark.mcp.server_adapter.socket.create_connection") as mock_conn, \
         patch("benchmark.mcp.server_adapter._recv_all", return_value=raw_bytes), \
         patch("benchmark.mcp.profiles.is_tool_allowed", return_value=True):
        mock_sock = MagicMock()
        mock_conn.return_value.__enter__ = lambda s: mock_sock
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        return adapter.call_tool("bma_export_scene", {"format": "blend", "filepath": "/tmp/out.blend"})


def test_empty_socket_response_is_classified():
    adapter = _make_adapter()
    with patch("benchmark.mcp.server_adapter.socket.create_connection"), \
         patch("benchmark.mcp.server_adapter._recv_all", return_value=b""), \
         patch("benchmark.mcp.profiles.is_tool_allowed", return_value=True):

        result = adapter.call_tool("bma_export_scene", {"format": "blend", "filepath": "/tmp/out.blend"})

    assert result["ok"] is False
    assert result["error"]["type"] == "EmptySocketResponse"


def test_invalid_json_response_keeps_raw_preview():
    adapter = _make_adapter()
    bad_bytes = b"<html>error page</html>"
    with patch("benchmark.mcp.server_adapter.socket.create_connection"), \
         patch("benchmark.mcp.server_adapter._recv_all", return_value=bad_bytes), \
         patch("benchmark.mcp.profiles.is_tool_allowed", return_value=True):

        result = adapter.call_tool("bma_export_scene", {"format": "blend", "filepath": "/tmp/out.blend"})

    assert result["ok"] is False
    assert result["error"]["type"] == "InvalidJsonResponse"
    assert "raw_preview" in result["error"]
    assert len(result["error"]["raw_preview"]) > 0
    assert result["error"].get("raw_len", 0) == len(bad_bytes)


def test_valid_export_response_with_ok_false_surfaced_as_error():
    """When addon returns ok=False inside status=success, it must be surfaced as error."""
    adapter = _make_adapter()
    addon_response = {
        "status": "success",
        "result": {
            "ok": False,
            "tool": "bma_export_scene",
            "format": "glb",
            "filepath": "/tmp/out.glb",
            "exists": False,
            "file_size_bytes": 0,
            "error": {
                "type": "ExportFailed",
                "message": "GLB subprocess failed",
                "format": "glb",
                "filepath": "/tmp/out.glb",
            },
        },
    }
    raw = json.dumps(addon_response).encode()
    with patch("benchmark.mcp.server_adapter.socket.create_connection"), \
         patch("benchmark.mcp.server_adapter._recv_all", return_value=raw), \
         patch("benchmark.mcp.profiles.is_tool_allowed", return_value=True):

        result = adapter.call_tool("bma_export_scene", {"format": "glb", "filepath": "/tmp/out.glb"})

    assert result["ok"] is False
    assert result["error"]["type"] == "ExportFailed"


def test_valid_export_success_response():
    """A proper success response with exists and file_size_bytes passes through."""
    adapter = _make_adapter()
    addon_response = {
        "status": "success",
        "result": {
            "ok": True,
            "tool": "bma_export_scene",
            "format": "blend",
            "filepath": "/tmp/out.blend",
            "exists": True,
            "file_size_bytes": 12345,
        },
    }
    raw = json.dumps(addon_response).encode()
    with patch("benchmark.mcp.server_adapter.socket.create_connection"), \
         patch("benchmark.mcp.server_adapter._recv_all", return_value=raw), \
         patch("benchmark.mcp.profiles.is_tool_allowed", return_value=True):

        result = adapter.call_tool("bma_export_scene", {"format": "blend", "filepath": "/tmp/out.blend"})

    assert result["ok"] is True
    inner = result.get("result") or {}
    assert inner.get("exists") is True
    assert inner.get("file_size_bytes") == 12345
