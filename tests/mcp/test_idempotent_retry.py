"""Tests for idempotent tool retry on transient socket failures.

Key invariants:
- bma_set_transform / bma_set_material / bma_assign_material retry once on
  EmptySocketResponse, SocketTimeout, SocketError.
- Non-idempotent tools (bma_create_light without if_exists=update) do NOT retry.
- bma_export_scene retries once on transient socket failures via watchdog policy.
- bma_create_object with if_exists=update retries once on transient failures.
- A second transient failure is returned as-is (no infinite retry).
- A non-transient error (ToolError, ExportFailed) is never retried.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from benchmark.mcp.server_adapter import (
    ExternalBlenderMcpServerAdapter,
    _IDEMPOTENT_TOOLS,
    _TRANSIENT_ERROR_TYPES,
    _tool_envelope,
)


def _make_adapter():
    from benchmark.mcp.config import McpServerConfig
    cfg = McpServerConfig(
        profile="minimal",
        blender_host="127.0.0.1",
        blender_port=9876,
    )
    return ExternalBlenderMcpServerAdapter(cfg)


def _transient_envelope(tool_name: str, error_type: str = "EmptySocketResponse") -> dict:
    return _tool_envelope(
        tool_name,
        ok=False,
        result={"raw_len": 0, "raw_preview": "", "failure_stage": "tool_response_parse"},
        error_type=error_type,
        error_message=f"transient: {error_type}",
    )


def _success_envelope(tool_name: str) -> dict:
    return _tool_envelope(tool_name, ok=True, result={"name": "Cube"})


# ---------------------------------------------------------------------------
# _TRANSIENT_ERROR_TYPES and _IDEMPOTENT_TOOLS membership
# ---------------------------------------------------------------------------

def test_transient_error_types_contains_expected():
    assert "EmptySocketResponse" in _TRANSIENT_ERROR_TYPES
    assert "SocketTimeout" in _TRANSIENT_ERROR_TYPES
    assert "SocketError" in _TRANSIENT_ERROR_TYPES
    assert "InvalidJsonResponse" not in _TRANSIENT_ERROR_TYPES


def test_idempotent_tools_contains_expected():
    assert "bma_set_transform" in _IDEMPOTENT_TOOLS
    assert "bma_set_material" in _IDEMPOTENT_TOOLS
    assert "bma_assign_material" in _IDEMPOTENT_TOOLS
    assert "bma_create_object" not in _IDEMPOTENT_TOOLS  # handled separately via if_exists
    assert "bma_export_scene" not in _IDEMPOTENT_TOOLS


# ---------------------------------------------------------------------------
# Idempotent tool retries on transient failure then succeeds
# ---------------------------------------------------------------------------

def test_set_transform_retries_on_empty_socket_and_succeeds():
    adapter = _make_adapter()
    call_results = [
        _transient_envelope("bma_set_transform", "EmptySocketResponse"),
        _success_envelope("bma_set_transform"),
    ]

    with patch.object(adapter, "_socket_call_once", side_effect=call_results) as mock_sc, \
         patch("benchmark.mcp.profiles.is_tool_allowed", return_value=True):
        result = adapter.call_tool("bma_set_transform", {"object_name": "Cube", "location": [1, 0, 0]})

    assert result["ok"] is True
    assert mock_sc.call_count == 2


def test_set_material_retries_on_socket_timeout():
    adapter = _make_adapter()
    call_results = [
        _transient_envelope("bma_set_material", "SocketTimeout"),
        _success_envelope("bma_set_material"),
    ]

    with patch.object(adapter, "_socket_call_once", side_effect=call_results) as mock_sc, \
         patch("benchmark.mcp.profiles.is_tool_allowed", return_value=True):
        result = adapter.call_tool("bma_set_material", {"object_name": "Cube"})

    assert result["ok"] is True
    assert mock_sc.call_count == 2


# ---------------------------------------------------------------------------
# Idempotent tool does NOT retry on second consecutive transient failure
# ---------------------------------------------------------------------------

def test_set_transform_returns_error_on_second_transient_failure():
    adapter = _make_adapter()
    call_results = [
        _transient_envelope("bma_set_transform", "EmptySocketResponse"),
        _transient_envelope("bma_set_transform", "EmptySocketResponse"),
    ]

    with patch.object(adapter, "_socket_call_once", side_effect=call_results) as mock_sc, \
         patch("benchmark.mcp.profiles.is_tool_allowed", return_value=True):
        result = adapter.call_tool("bma_set_transform", {"object_name": "Cube"})

    assert result["ok"] is False
    assert result["error"]["type"] == "EmptySocketResponse"
    assert mock_sc.call_count == 2  # retried exactly once, then gave up


# ---------------------------------------------------------------------------
# Non-idempotent tool does NOT retry
# ---------------------------------------------------------------------------

def test_create_light_does_not_retry_on_transient():
    adapter = _make_adapter()
    call_results = [
        _transient_envelope("bma_create_light", "EmptySocketResponse"),
        _success_envelope("bma_create_light"),
    ]

    with patch.object(adapter, "_socket_call_once", side_effect=call_results) as mock_sc, \
         patch("benchmark.mcp.profiles.is_tool_allowed", return_value=True):
        result = adapter.call_tool("bma_create_light", {"type": "POINT"})

    # Should NOT have retried — first (failed) result returned directly
    assert result["ok"] is False
    assert mock_sc.call_count == 1


def test_export_scene_retries_on_empty_socket_and_succeeds():
    adapter = _make_adapter()
    call_results = [
        _transient_envelope("bma_export_scene", "EmptySocketResponse"),
        _success_envelope("bma_export_scene"),
    ]

    with patch.object(adapter, "_socket_call_once", side_effect=call_results) as mock_sc, \
         patch("benchmark.mcp.profiles.is_tool_allowed", return_value=True):
        result = adapter.call_tool("bma_export_scene", {"filepath": "/tmp/scene.blend", "format": "BLEND"})

    assert result["ok"] is True
    assert mock_sc.call_count == 2


# ---------------------------------------------------------------------------
# Non-transient error is never retried
# ---------------------------------------------------------------------------

def test_idempotent_tool_does_not_retry_on_tool_error():
    adapter = _make_adapter()
    non_transient = _tool_envelope(
        "bma_set_transform",
        ok=False,
        result=None,
        error_type="ToolError",
        error_message="Object not found",
    )

    with patch.object(adapter, "_socket_call_once", return_value=non_transient) as mock_sc, \
         patch("benchmark.mcp.profiles.is_tool_allowed", return_value=True):
        result = adapter.call_tool("bma_set_transform", {"object_name": "Missing"})

    assert result["ok"] is False
    assert result["error"]["type"] == "ToolError"
    assert mock_sc.call_count == 1


# ---------------------------------------------------------------------------
# bma_create_object with if_exists=update retries on transient failure
# ---------------------------------------------------------------------------

def test_create_object_if_exists_update_retries_on_transient():
    adapter = _make_adapter()
    call_results = [
        _transient_envelope("bma_create_object", "EmptySocketResponse"),
        _success_envelope("bma_create_object"),
    ]

    with patch.object(adapter, "_socket_call_once", side_effect=call_results) as mock_sc, \
         patch("benchmark.mcp.profiles.is_tool_allowed", return_value=True), \
         patch("benchmark.mcp.server_adapter.socket.create_connection") as mock_conn, \
         patch("benchmark.mcp.server_adapter._recv_all", return_value=b'{"status":"error","message":"not found"}'):
        mock_conn.return_value.__enter__ = lambda s: MagicMock()
        mock_conn.return_value.__exit__ = MagicMock(return_value=False)
        result = adapter.call_tool("bma_create_object", {
            "type": "MESH_CUBE",
            "name": "Cube",
            "if_exists": "update",
        })

    assert result["ok"] is True
    assert mock_sc.call_count == 2
