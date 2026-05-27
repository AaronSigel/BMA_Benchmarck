"""Fixture: export_002 с одной плоскостью вместо low-poly сцены."""

from __future__ import annotations

from pathlib import Path

import yaml

from benchmark.blender.models import ObjectSnapshot, RenderSettingsSnapshot, SceneSnapshot, Vector3
from benchmark.tasks.models import BenchmarkTask
from benchmark.validation.models import CheckStatus, ValidationStatus
from benchmark.validation.scene_validator import SceneValidator
from benchmark.analysis.pass_type_rules import apply_export_pass_type_guard


def _plane_only_snapshot() -> SceneSnapshot:
    return SceneSnapshot(
        scene_name="Scene",
        objects=[
            ObjectSnapshot(
                name="Plane",
                type="MESH",
                primitive_hint="plane",
                location=Vector3(x=0, y=0, z=0),
                rotation_euler=Vector3(x=0, y=0, z=0),
                scale=Vector3(x=1, y=1, z=1),
                dimensions=Vector3(x=1, y=1, z=0),
                material_slots=[],
                parent=None,
                collection_names=["Collection"],
                vertex_count=4,
                polygon_count=1,
            )
        ],
        materials=[],
        lights=[],
        cameras=[],
        collections=["Collection"],
        render_settings=RenderSettingsSnapshot(
            engine="CYCLES",
            resolution_x=1920,
            resolution_y=1080,
            frame_start=1,
            frame_end=1,
            frame_current=1,
        ),
        frame_current=1,
        blender_version="4.0.0",
        created_at="2026-05-15T12:00:00Z",
    )


def _export_002_task() -> BenchmarkTask:
    raw = yaml.safe_load(
        Path("tasks/export/export_002_glb_file.yaml").read_text(encoding="utf-8")
    )
    return BenchmarkTask.model_validate(raw)


def test_plane_only_scene_fails_validation_with_export_checks(tmp_path: Path) -> None:
    task = _export_002_task()
    result = SceneValidator().validate(task, _plane_only_snapshot(), artifacts_dir=tmp_path)

    assert result.overall_status is ValidationStatus.FAILED
    assert result.total_score < 0.6
    issue_codes = {issue.code for issue in result.issues}
    assert "object_missing" in issue_codes

    exists_rows = [row for row in result.check_table if row.check_name == "object exists"]
    assert exists_rows
    assert any(row.actual == "not_found" for row in exists_rows)

    ground_exists = next(
        row for row in exists_rows if row.entity_ref == "Lowpoly_Ground"
    )
    assert ground_exists.expected == "exists"

    primitive_rows = [row for row in result.check_table if row.check_name == "primitive hint"]
    ground_primitive = next(
        (row for row in primitive_rows if row.entity_ref == "Lowpoly_Ground"),
        None,
    )
    if ground_primitive is not None:
        assert ground_primitive.actual == "plane"
        assert ground_primitive.passed is False

    skip_rows = [row for row in result.check_table if row.status is CheckStatus.SKIP]
    assert skip_rows

    export_rows = [
        row for row in result.check_table if row.validator_name == "export_validator"
    ]
    assert export_rows
    assert any(row.check_name == "file exists" for row in export_rows)

    scores = result.summary.get("scores") or {}
    assert "scene_score" in scores
    assert "export_score" in scores
    assert "final_score" in scores


def test_plane_only_classified_as_failed_validation_not_soft_pass() -> None:
    task = _export_002_task()
    result = SceneValidator().validate(task, _plane_only_snapshot(), artifacts_dir=Path("."))
    issues = [{"code": issue.code} for issue in result.issues]
    pass_type = apply_export_pass_type_guard(
        "soft_pass",
        task.id,
        issues,
        object_score=(result.summary.get("scores") or {}).get("scene_score"),
        export_score=(result.summary.get("scores") or {}).get("export_score"),
        import_back_score=(result.summary.get("scores") or {}).get("import_back_score"),
    )
    assert pass_type == "failed_validation"
