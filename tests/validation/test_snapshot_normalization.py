"""Tests for scene snapshot normalization."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.validation.snapshot_normalization import (
    SnapshotSchemaError,
    normalize_scene_snapshot,
    unwrap_raw_snapshot,
    validate_from_tool_result,
)


def _minimal_snapshot() -> dict:
    fixture = Path("tests/fixtures/e2e/scene_snapshot.json")
    return json.loads(fixture.read_text(encoding="utf-8"))


def test_normalize_wrapped_mcp_snapshot() -> None:
    inner = _minimal_snapshot()
    wrapped = {"ok": True, "tool": "bma_get_scene_snapshot", "result": inner, "error": None}
    snapshot = normalize_scene_snapshot(wrapped)
    assert snapshot.scene_name == inner["scene_name"]
    assert len(snapshot.objects) == len(inner["objects"])


def test_normalize_plain_scene_snapshot() -> None:
    raw = _minimal_snapshot()
    snapshot = normalize_scene_snapshot(raw)
    assert snapshot.blender_version == raw["blender_version"]


def test_snapshot_schema_error_has_diagnostics() -> None:
    with pytest.raises(SnapshotSchemaError) as exc_info:
        normalize_scene_snapshot({"objects": []})
    err = exc_info.value
    diag = err.to_dict()
    assert diag["error_type"] == "SnapshotSchemaError"
    assert diag["failure_stage"] == "snapshot_normalization"
    assert "objects" in diag["raw_keys"]
    assert diag["expected_schema"]


def test_unwrap_nested_data_wrapper() -> None:
    inner = _minimal_snapshot()
    wrapped = {"data": {"snapshot": inner}}
    assert unwrap_raw_snapshot(wrapped) == inner


def test_validate_from_tool_result_unwraps_envelope() -> None:
    import yaml

    task_path = Path("tasks/geometry/geometry_001_basic_primitives.yaml")
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))

    class _Result:
        error = None
        result = {"ok": True, "tool": "bma_get_scene_snapshot", "result": _minimal_snapshot(), "error": None}

    val_result, reason = validate_from_tool_result(_Result(), task)
    assert reason is None
    assert val_result is not None


def test_validate_from_tool_result_partial_snapshot() -> None:
    class _Result:
        error = None
        result = {"ok": True, "tool": "bma_get_scene_snapshot", "result": {"objects": []}, "error": None}

    task = {"id": "geometry_001", "category": "geometry", "expected_scene": {"objects": []}}
    val_result, reason = validate_from_tool_result(_Result(), task)
    assert val_result is None
    assert reason == "snapshot_invalid_schema"
