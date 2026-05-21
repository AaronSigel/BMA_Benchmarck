"""Tests for bma_create_object param tolerance in server_adapter.

When the LLM sends unrecognized parameters (e.g. mesh_type, color, material),
the addon handler must absorb them silently via **_ignored and still succeed.
The server_adapter must surface ok=False with CreateObjectFailed for genuine errors
(bad type string) while unknown extra params must not cause TypeError.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from benchmark.mcp.server_adapter import ExternalBlenderMcpServerAdapter, _tool_envelope


def _make_adapter():
    from benchmark.mcp.config import McpServerConfig
    cfg = McpServerConfig(
        profile="minimal",
        blender_host="127.0.0.1",
        blender_port=9876,
    )
    return ExternalBlenderMcpServerAdapter(cfg)


def _mock_addon_response(adapter, *, addon_result: dict, tool_name: str = "bma_create_object", params: dict | None = None):
    """Patch socket layer to return addon_result as the addon's socket response."""
    if params is None:
        params = {"type": "MESH_CUBE", "name": "Cube"}
    raw = json.dumps({"status": "success", "result": addon_result}).encode()
    with patch("benchmark.mcp.server_adapter.socket.create_connection"), \
         patch("benchmark.mcp.server_adapter._recv_all", return_value=raw), \
         patch("benchmark.mcp.profiles.is_tool_allowed", return_value=True):
        return adapter.call_tool(tool_name, params)


# ---------------------------------------------------------------------------
# Adapter-level: ok=True result passes through
# ---------------------------------------------------------------------------

def test_create_object_ok_result_passes_through():
    adapter = _make_adapter()
    addon_result = {
        "ok": True,
        "tool": "bma_create_object",
        "name": "Cube",
        "type": "MESH",
        "location": [0.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "dimensions": [2.0, 2.0, 2.0],
        "primitive_hint": "MESH_CUBE",
    }
    result = _mock_addon_response(adapter, addon_result=addon_result)
    assert result["ok"] is True
    inner = result.get("result") or result
    assert inner.get("name") == "Cube"


# ---------------------------------------------------------------------------
# Adapter-level: ok=False from addon is surfaced as error
# ---------------------------------------------------------------------------

def test_create_object_error_result_surfaced():
    adapter = _make_adapter()
    addon_result = {
        "ok": False,
        "tool": "bma_create_object",
        "error": {"type": "CreateObjectFailed", "message": "unsupported type: MESH_TORUS"},
    }
    result = _mock_addon_response(adapter, addon_result=addon_result,
                                  params={"type": "MESH_TORUS", "name": "Torus"})
    assert result["ok"] is False
    assert result["error"]["type"] == "CreateObjectFailed"


# ---------------------------------------------------------------------------
# _tool_envelope: unknown params extra fields don't affect envelope shape
# ---------------------------------------------------------------------------

def test_tool_envelope_result_shape_stable():
    envelope = _tool_envelope(
        "bma_create_object",
        ok=True,
        result={"name": "Cube", "extra_field": "ignored"},
    )
    assert envelope["ok"] is True
    assert envelope["tool"] == "bma_create_object"
    assert envelope["error"] is None
    assert envelope["result"]["name"] == "Cube"


# ---------------------------------------------------------------------------
# Verify _call_create_object_with_if_exists strips if_exists before forwarding
# ---------------------------------------------------------------------------

def test_if_exists_stripped_from_create_params():
    """if_exists must not be forwarded to _socket_call_once as a param."""
    adapter = _make_adapter()
    captured_params: dict = {}

    def fake_socket_call(tool_name, params):
        if params:
            captured_params.update(params)
        return {"ok": True, "tool": tool_name, "result": {"name": "Cube"}, "error": None}

    fake_profile = MagicMock()

    # _call_create_object_with_if_exists calls _socket_call_once for the actual create.
    # The existence check uses a raw socket; patch that to report "not found".
    with patch.object(adapter, "_socket_call_once", side_effect=fake_socket_call) as mock_sc, \
         patch("benchmark.mcp.server_adapter.socket.create_connection") as mock_conn, \
         patch("benchmark.mcp.server_adapter._recv_all", return_value=b'{"status":"error","message":"not found"}'), \
         patch("benchmark.mcp.profiles.is_tool_allowed", return_value=True):
        mock_conn.return_value.__enter__ = lambda s: MagicMock()
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)

        result = adapter._call_create_object_with_if_exists(
            {
                "type": "MESH_CUBE",
                "name": "Cube",
                "if_exists": "update",
            },
            profile=fake_profile,
        )

    assert mock_sc.called
    _, forwarded_params = mock_sc.call_args[0]
    assert "if_exists" not in forwarded_params
