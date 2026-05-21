"""Unit tests for issue_action_mapper: issue code → RepairAction mapping."""
from __future__ import annotations

import pytest

from benchmark.blender.models import (
    ObjectSnapshot,
    RenderSettingsSnapshot,
    SceneSnapshot,
    Vector3 as SnapshotVector3,
)
from benchmark.agent.strategies.issue_action_mapper import (
    EXPORT_BLOCKING_CODES,
    RepairAction,
    build_task_checklist,
    has_export_blocking_issues,
    map_issue_to_repair,
    select_top_issue,
)
from benchmark.tasks.models import (
    BenchmarkTask,
    ColorRGBA,
    DifficultyLevel,
    ExpectedCamera,
    ExpectedExport,
    ExpectedLight,
    ExpectedMaterial,
    ExpectedObject,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
    Vector3,
)
from benchmark.validation.models import ValidationIssue, ValidationSeverity


def _issue(code: str, severity: ValidationSeverity = ValidationSeverity.ERROR) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=f"Issue: {code}",
        severity=severity,
        expected_path="expected_scene.objects[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )


def _task(**scene_kwargs) -> BenchmarkTask:
    return BenchmarkTask(
        id="test_task",
        title="Test",
        category=TaskCategory.GEOMETRY,
        difficulty=DifficultyLevel.EASY,
        prompt="Do stuff.",
        tags=["test"],
        allowed_tools=[],
        expected_scene=ExpectedScene(**scene_kwargs),
        success_criteria=[SuccessCriterion(metric="object_existence", weight=1.0)],
    )


def _snapshot_with_object(name: str, x: float, y: float, z: float) -> SceneSnapshot:
    vec = SnapshotVector3(x=x, y=y, z=z)
    return SceneSnapshot(
        scene_name="Scene",
        objects=[
            ObjectSnapshot(
                name=name,
                type="MESH",
                primitive_hint=None,
                location=vec,
                rotation_euler=SnapshotVector3(x=0, y=0, z=0),
                scale=SnapshotVector3(x=1, y=1, z=1),
                dimensions=SnapshotVector3(x=1, y=1, z=1),
                material_slots=[],
                parent=None,
                collection_names=["Collection"],
                vertex_count=None,
                polygon_count=None,
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


# --- RepairAction dataclass shape ---

def test_repair_action_has_required_fields() -> None:
    action = RepairAction(
        issue_code="object_missing",
        tool_name="bma_create_object",
        arguments_template={"name": "Cube"},
        description="Create Cube",
    )
    assert action.issue_code == "object_missing"
    assert action.tool_name == "bma_create_object"
    assert action.priority == 99
    assert action.blocking is False
    assert action.expected_value is None
    assert action.actual_value is None
    assert action.confidence == 1.0


def test_repair_action_blocking_flag_from_export_blocking_codes() -> None:
    assert "object_missing" in EXPORT_BLOCKING_CODES
    assert "export_missing" not in EXPORT_BLOCKING_CODES


# --- select_top_issue ---

def test_select_top_issue_prefers_error_severity() -> None:
    issues = [
        _issue("export_missing", ValidationSeverity.WARNING),
        _issue("object_missing", ValidationSeverity.ERROR),
        _issue("light_missing", ValidationSeverity.ERROR),
    ]
    top = select_top_issue(issues)
    assert top is not None
    assert top.code == "object_missing"


def test_select_top_issue_returns_none_for_empty_list() -> None:
    assert select_top_issue([]) is None


def test_select_top_issue_falls_back_to_warnings_when_no_errors() -> None:
    issues = [
        _issue("export_missing", ValidationSeverity.WARNING),
        _issue("light_missing", ValidationSeverity.WARNING),
    ]
    top = select_top_issue(issues)
    assert top is not None


# --- has_export_blocking_issues ---

def test_has_export_blocking_issues_detects_blocking() -> None:
    issues = [_issue("object_missing"), _issue("export_missing")]
    blocking = has_export_blocking_issues(issues)
    assert any(i.code == "object_missing" for i in blocking)
    assert all(i.code != "export_missing" for i in blocking)


# --- map_issue_to_repair ---

def test_map_object_missing_to_create_object() -> None:
    task = _task(objects=[ExpectedObject(name="Cube", type="MESH", primitive="cube")])
    issue = _issue("object_missing")
    issue = ValidationIssue(
        code="object_missing",
        message="Missing",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.objects[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )
    action = map_issue_to_repair(issue, task)
    assert action is not None
    assert action.tool_name == "bma_create_object"
    assert action.arguments_template.get("name") == "Cube"
    assert action.blocking is True


def test_map_material_missing_to_assign_material() -> None:
    task = _task(
        objects=[ExpectedObject(name="Cube", type="MESH", primitive="cube")],
        materials=[
            ExpectedMaterial(
                name="RedMat",
                base_color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            )
        ],
    )
    issue = ValidationIssue(
        code="material_missing",
        message="Missing material",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.materials[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )
    action = map_issue_to_repair(issue, task)
    assert action is not None
    assert action.tool_name == "bma_assign_material"


def test_map_light_missing_to_create_light() -> None:
    task = _task(
        lights=[ExpectedLight(name="Key", type="AREA", energy=500.0, location=Vector3(x=0, y=-3, z=4))]
    )
    issue = ValidationIssue(
        code="light_missing",
        message="Missing light",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.lights[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )
    action = map_issue_to_repair(issue, task)
    assert action is not None
    assert action.tool_name == "bma_create_light"
    assert action.arguments_template.get("name") == "Key"
    assert action.arguments_template.get("energy") == 500.0
    assert action.blocking is True


def test_map_light_location_mismatch() -> None:
    task = _task(lights=[ExpectedLight(name="Key", type="AREA", location=Vector3(x=0, y=-3, z=4))])
    issue = ValidationIssue(
        code="light_location_mismatch",
        message="Location off",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.lights[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )
    action = map_issue_to_repair(issue, task)
    assert action is not None
    assert action.tool_name == "bma_set_transform"
    assert action.arguments_template.get("object_name") == "Key"
    assert action.arguments_template.get("location") == [0.0, -3.0, 4.0]


def test_light_direction_mismatch_uses_set_transform() -> None:
    task = _task(
        lights=[
            ExpectedLight(
                name="Key_Light",
                type="AREA",
                location=Vector3(x=-3.0, y=-4.0, z=5.0),
                rotation=Vector3(x=60.0, y=0.0, z=-35.0),
                target="Center_Object",
            )
        ]
    )
    issue = ValidationIssue(
        code="light_direction_mismatch",
        message="Direction off",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.lights[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )
    action = map_issue_to_repair(issue, task)
    assert action is not None
    assert action.tool_name == "bma_create_light"
    assert action.arguments_template.get("name") == "Key_Light"
    assert action.arguments_template.get("if_exists") == "update"
    assert "target" in action.arguments_template or "target_object_name" in action.arguments_template


def test_light_missing_with_string_target_includes_rotation_radians() -> None:
    import math

    task = _task(
        lights=[
            ExpectedLight(
                name="Fill_Light",
                type="AREA",
                location=Vector3(x=4.0, y=-3.0, z=3.0),
                rotation=Vector3(x=55.0, y=0.0, z=45.0),
                target="Center_Object",
                energy=250.0,
            )
        ]
    )
    issue = ValidationIssue(
        code="light_missing",
        message="Missing light",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.lights[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )
    action = map_issue_to_repair(issue, task)
    assert action is not None
    assert action.tool_name == "bma_create_light"
    rotation = action.arguments_template.get("rotation")
    assert rotation is not None
    assert rotation[0] == math.radians(55.0)
    assert "target" not in action.arguments_template


def test_light_direction_mismatch_does_not_use_create_light() -> None:
    task = _task(
        lights=[ExpectedLight(name="Back_Light", type="SPOT", rotation=Vector3(x=55.0, y=0.0, z=180.0))]
    )
    issue = ValidationIssue(
        code="light_direction_mismatch",
        message="Direction off",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.lights[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )
    action = map_issue_to_repair(issue, task)
    assert action is not None
    assert action.tool_name != "bma_create_light"


def test_map_camera_target_to_look_at() -> None:
    task = _task(
        cameras=[
            ExpectedCamera(
                name="Camera",
                location=Vector3(x=4, y=-6, z=4),
                target=Vector3(x=0, y=0, z=1),
                focal_length=35.0,
            )
        ]
    )
    issue = ValidationIssue(
        code="camera_missing",
        message="Missing camera",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.cameras[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )
    action = map_issue_to_repair(issue, task)
    assert action is not None
    assert action.tool_name == "bma_create_camera_look_at"
    assert action.arguments_template.get("name") == "Camera"
    assert action.arguments_template.get("focal_length") == 35.0


def test_camera_target_string_does_not_crash_mapper() -> None:
    task = _task(
        cameras=[
            ExpectedCamera(
                name="Camera",
                location=Vector3(x=4, y=-6, z=4),
                target="Center_Sphere",
                focal_length=35.0,
            )
        ]
    )
    issue = ValidationIssue(
        code="camera_missing",
        message="Missing camera",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.cameras[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )

    action = map_issue_to_repair(issue, task)

    assert action is not None
    assert action.tool_name == "bma_create_camera_look_at"
    assert action.arguments_template["target_object_name"] == "Center_Sphere"


def test_camera_target_string_uses_snapshot_object_coords() -> None:
    task = _task(
        cameras=[
            ExpectedCamera(
                name="Camera",
                location=Vector3(x=4, y=-6, z=4),
                target="Center_Sphere",
                focal_length=35.0,
            )
        ]
    )
    issue = ValidationIssue(
        code="camera_missing",
        message="Missing camera",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.cameras[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )

    action = map_issue_to_repair(issue, task, _snapshot_with_object("Center_Sphere", 0.0, 0.0, 0.75))

    assert action is not None
    assert action.arguments_template["target"] == [0.0, 0.0, 0.75]
    assert "target_object_name" not in action.arguments_template


def test_map_camera_location_mismatch() -> None:
    task = _task(cameras=[ExpectedCamera(name="Camera", location=Vector3(x=4, y=-6, z=4))])
    issue = ValidationIssue(
        code="camera_location_mismatch",
        message="Camera location off",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.cameras[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )
    action = map_issue_to_repair(issue, task)
    assert action is not None
    assert action.tool_name in {"bma_create_camera", "bma_create_camera_look_at"}


def test_map_camera_focal_length_mismatch() -> None:
    task = _task(cameras=[ExpectedCamera(name="Camera", focal_length=50.0)])
    issue = ValidationIssue(
        code="camera_focal_length_mismatch",
        message="Lens mismatch",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.cameras[0]",
        actual_path=None,
        expected_value=50.0,
        actual_value=35.0,
    )
    action = map_issue_to_repair(issue, task)
    assert action is not None


def test_map_export_missing_to_export_scene() -> None:
    task = _task(exports=[ExpectedExport(format="glb", filename="result.glb")])
    issue = ValidationIssue(
        code="export_missing",
        message="Missing export",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.exports[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )
    action = map_issue_to_repair(issue, task)
    assert action is not None
    assert action.tool_name == "bma_export_scene"
    assert action.arguments_template.get("format") == "glb"
    assert action.blocking is False


def test_map_export_empty_file() -> None:
    task = _task(exports=[ExpectedExport(format="glb")])
    issue = ValidationIssue(
        code="export_empty_file",
        message="Export file empty",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.exports[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )
    action = map_issue_to_repair(issue, task)
    assert action is not None
    assert action.tool_name == "bma_export_scene"


def test_map_glb_import_back_failed() -> None:
    task = _task(exports=[ExpectedExport(format="glb")])
    issue = ValidationIssue(
        code="export_import_failed",
        message="GLB import failed",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.exports[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )
    action = map_issue_to_repair(issue, task)
    assert action is not None
    assert action.tool_name == "bma_export_scene"


def test_map_unknown_issue_returns_none() -> None:
    task = _task()
    issue = ValidationIssue(
        code="completely_unknown_issue_xyz",
        message="Unknown",
        severity=ValidationSeverity.ERROR,
        expected_path=None,
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )
    assert map_issue_to_repair(issue, task) is None


def test_repair_action_priority_populated() -> None:
    task = _task(objects=[ExpectedObject(name="Cube", type="MESH", primitive="cube")])
    issue = ValidationIssue(
        code="object_missing",
        message="Missing",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.objects[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )
    action = map_issue_to_repair(issue, task)
    assert action is not None
    assert action.priority == 0  # highest priority


# --- build_task_checklist ---

def test_build_task_checklist_includes_all_sections() -> None:
    task = _task(
        objects=[ExpectedObject(name="Cube", type="MESH", primitive="cube")],
        lights=[ExpectedLight(name="Key", type="AREA")],
        cameras=[ExpectedCamera(name="Camera")],
        exports=[ExpectedExport(format="glb")],
    )
    checklist = build_task_checklist(task)
    assert len(checklist["required_objects"]) == 1
    assert len(checklist["required_lights"]) == 1
    assert len(checklist["required_cameras"]) == 1
    assert len(checklist["required_exports"]) == 1
