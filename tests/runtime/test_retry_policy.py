from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from benchmark.agent.execution_backend import (
    _capture_snapshot_to_path,
    _envelope_failed,
    _prepare_blender_scene,
)
from benchmark.agent.tool_executor import McpToolExecutor
from benchmark.mcp.config import McpServerConfig
from benchmark.mcp.server_adapter import ExternalBlenderMcpServerAdapter


def test_reset_retried_once() -> None:
    adapter = MagicMock()
    adapter.reset_scene.side_effect = [
        {"ok": False, "error": {"type": "ResetSceneFailed", "message": "fail"}},
        {"ok": True, "result": {"reset": True}},
    ]
    adapter._watchdog = None
    executor = McpToolExecutor(adapter, profile="full")
    snap_path = Path("/tmp/snap.json")
    with patch("benchmark.agent.execution_backend._capture_snapshot_to_path", return_value=snap_path):
        with patch("benchmark.agent.execution_backend.SceneSnapshot.model_validate_json") as validate:
            validate.return_value = MagicMock(objects=[])
            with patch.object(Path, "read_text", return_value="{}"):
                error, _ = _prepare_blender_scene(executor, Path("/tmp"), "task-1")
    assert error is None
    assert adapter.reset_scene.call_count == 2


def test_snapshot_retried_once() -> None:
    adapter = MagicMock()
    adapter.collect_scene_snapshot.side_effect = [
        {"ok": False, "error": {"type": "SnapshotUnavailable", "message": "fail"}},
        {"ok": True, "result": {"path": "/tmp/snap.json"}},
    ]
    executor = McpToolExecutor(adapter, profile="full")
    path = Path("/tmp/snap.json")
    with patch.object(Path, "exists", return_value=True):
        captured = _capture_snapshot_to_path(executor, path)
    assert captured == path
    assert adapter.collect_scene_snapshot.call_count == 2


def test_idempotent_tool_retried_on_empty_response() -> None:
    cfg = McpServerConfig(profile="full", blender_host="localhost", blender_port=9876)
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    calls = {"count": 0}

    def _socket_once(tool_name, params, *, attempt=1):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "ok": False,
                "tool": tool_name,
                "error": {"type": "EmptySocketResponse", "message": "empty"},
            }
        return {"ok": True, "tool": tool_name, "result": {"ok": True}, "error": None}

    with patch.object(adapter, "_socket_call_once", side_effect=_socket_once):
        result = adapter.call_tool("bma_assign_material", {"object_name": "Cube", "material_name": "Red"})
    assert result.get("ok") is True
    assert calls["count"] == 2


def test_export_scene_retried_once_on_empty_socket() -> None:
    cfg = McpServerConfig(profile="full", blender_host="localhost", blender_port=9876)
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    calls = {"count": 0}

    def _socket_once(tool_name, params, *, attempt=1):
        calls["count"] += 1
        return {
            "ok": False,
            "tool": tool_name,
            "error": {"type": "EmptySocketResponse", "message": "empty"},
        }

    with patch.object(adapter, "_socket_call_once", side_effect=_socket_once):
        result = adapter.call_tool("bma_export_scene", {"filepath": "/tmp/out.glb"})
    assert result.get("ok") is False
    assert calls["count"] == 2


def test_execute_blender_code_not_retried_on_runtime_failure() -> None:
    cfg = McpServerConfig(profile="full", blender_host="localhost", blender_port=9876)
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    calls = {"count": 0}

    def _socket_once(tool_name, params, *, attempt=1):
        calls["count"] += 1
        return {
            "ok": False,
            "tool": tool_name,
            "error": {
                "type": "EmptySocketResponse",
                "message": "empty",
                "failure_stage": "blender_python_execution",
            },
        }

    with patch.object(adapter, "_socket_call_once", side_effect=_socket_once):
        result = adapter.call_tool("execute_blender_code", {"code": "import bpy"})
    assert result.get("ok") is False
    assert calls["count"] == 1


def test_envelope_failed_detects_warning_and_error_envelope() -> None:
    assert _envelope_failed({"warning": "legacy"}) is True
    assert _envelope_failed({"ok": False, "error": {"type": "ResetSceneFailed"}}) is True
    assert _envelope_failed({"ok": True, "result": {}}) is False
