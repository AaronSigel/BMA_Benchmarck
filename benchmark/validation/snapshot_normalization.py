"""Нормализация raw snapshot payloads в SceneSnapshot."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from benchmark.blender.models import SceneSnapshot

log = logging.getLogger(__name__)

_SCENE_SNAPSHOT_REQUIRED_KEYS = frozenset(
    {
        "scene_name",
        "objects",
        "materials",
        "lights",
        "cameras",
        "collections",
        "render_settings",
        "frame_current",
        "blender_version",
        "created_at",
    }
)

EXPECTED_SCHEMA = "SceneSnapshot (see benchmark/blender/models.py)"


class SnapshotSchemaError(Exception):
    """Structured error when raw snapshot cannot be normalized."""

    error_type = "SnapshotSchemaError"
    failure_stage = "snapshot_normalization"

    def __init__(
        self,
        message: str,
        *,
        raw_keys: list[str] | None = None,
        expected_schema: str = EXPECTED_SCHEMA,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.raw_keys = raw_keys or []
        self.expected_schema = expected_schema

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "raw_keys": self.raw_keys,
            "expected_schema": self.expected_schema,
            "failure_stage": self.failure_stage,
        }


def unwrap_raw_snapshot(raw: Any) -> dict[str, Any] | None:
    """Extract inner scene dict from MCP envelopes and wrapper keys."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None

    current: Any = raw
    for _ in range(6):
        if not isinstance(current, dict):
            return None
        if _looks_like_scene_snapshot(current):
            return current
        if {"ok", "tool", "result", "error"}.issubset(current.keys()):
            inner = current.get("result")
            if inner is None:
                return None
            current = inner
            continue
        for key in ("snapshot", "data", "scene", "scene_snapshot"):
            if key in current and isinstance(current[key], dict):
                current = current[key]
                break
        else:
            return current if isinstance(current, dict) else None
    return current if isinstance(current, dict) else None


def _looks_like_scene_snapshot(data: dict[str, Any]) -> bool:
    return _SCENE_SNAPSHOT_REQUIRED_KEYS.issubset(data.keys())


def is_full_scene_snapshot_payload(data: dict[str, Any]) -> bool:
    return _looks_like_scene_snapshot(data)


def normalize_scene_snapshot(raw: Any) -> SceneSnapshot:
    """Normalize raw MCP/tool/file payload into a validated SceneSnapshot."""
    unwrapped = unwrap_raw_snapshot(raw)
    if unwrapped is None:
        raw_keys = list(raw.keys()) if isinstance(raw, dict) else []
        raise SnapshotSchemaError(
            "snapshot payload is missing or not a dict",
            raw_keys=raw_keys,
        )
    if not is_full_scene_snapshot_payload(unwrapped):
        missing = sorted(_SCENE_SNAPSHOT_REQUIRED_KEYS - set(unwrapped.keys()))
        raise SnapshotSchemaError(
            f"partial snapshot payload; missing keys: {', '.join(missing)}",
            raw_keys=sorted(unwrapped.keys()),
        )
    try:
        return SceneSnapshot.model_validate(unwrapped)
    except ValidationError as exc:
        raise SnapshotSchemaError(
            f"SceneSnapshot validation failed: {exc}",
            raw_keys=sorted(unwrapped.keys()),
        ) from exc


def validate_task_scene_from_raw(
    raw: Any,
    task: dict[str, Any],
    *,
    artifacts_dir: Path | None = None,
) -> tuple[Any, SceneSnapshot]:
    """Normalize snapshot and run SceneValidator for a task dict."""
    from pathlib import Path as _Path

    from benchmark.tasks.models import BenchmarkTask
    from benchmark.validation.scene_validator import SceneValidator

    snapshot = normalize_scene_snapshot(raw)
    task_obj = BenchmarkTask.model_validate(task)
    kwargs: dict[str, Any] = {}
    if artifacts_dir is not None:
        kwargs["artifacts_dir"] = _Path(artifacts_dir)
    val_result = SceneValidator().validate(task_obj, snapshot, **kwargs)
    return val_result, snapshot


def validate_from_tool_result(
    tool_result: Any,
    task: dict[str, Any],
    snap_path: Path | None = None,
) -> tuple[Any | None, str | None]:
    """Validate scene from ToolCallResult; returns (validation_result, unavailable_reason)."""
    if getattr(tool_result, "error", None):
        return None, "snapshot_tool_failed"
    raw = getattr(tool_result, "result", None)
    if raw is None:
        return None, "no_snapshot"
    try:
        val_result, snapshot = validate_task_scene_from_raw(raw, task)
    except SnapshotSchemaError:
        return None, "snapshot_invalid_schema"
    except Exception:
        try:
            from benchmark.tasks.models import BenchmarkTask

            BenchmarkTask.model_validate(task)
        except Exception:
            return None, "task_parse_failed"
        return None, "validation_exception"
    if snap_path is not None:
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        snap_path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
    return val_result, None


def build_scene_validator_fn(
    tool_executor: Any,
    task: dict[str, Any],
) -> Callable[[Path], tuple[bool, float | None, Any]]:
    """Build scene_validator_fn preferring adapter collect_scene_snapshot, else tool path."""
    from benchmark.validation.models import ValidationStatus

    adapter = getattr(tool_executor, "adapter", None)

    def _fn(snap_path: Path) -> tuple[bool, float | None, Any]:
        try:
            if adapter is not None and hasattr(adapter, "collect_scene_snapshot"):
                raw = adapter.collect_scene_snapshot(snap_path)
                if isinstance(raw, dict) and raw.get("warning"):
                    log.debug("scene validator: collect_scene_snapshot warning: %s", raw.get("warning"))
                    return False, None, None
                if not snap_path.exists():
                    return False, None, None
                raw_payload = json.loads(snap_path.read_text(encoding="utf-8"))
                val_result, _ = validate_task_scene_from_raw(raw_payload, task)
            else:
                result = tool_executor.call_tool("bma_get_scene_snapshot", {})
                val_result, reason = validate_from_tool_result(result, task, snap_path)
                if val_result is None:
                    log.debug("scene validator unavailable: %s", reason)
                    return False, None, None
            scene_ok = val_result.overall_status in {ValidationStatus.PASSED, ValidationStatus.WARNING}
            return scene_ok, val_result.total_score, val_result
        except SnapshotSchemaError as exc:
            log.debug("scene validator snapshot schema error: %s", exc.message)
            return False, None, None
        except Exception as exc:
            log.debug("scene validator failed: %s", exc)
            return False, None, None

    return _fn
