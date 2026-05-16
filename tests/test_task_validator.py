"""Tests for benchmark/tasks/validator.py — structural consistency checks."""

import pytest

from benchmark.tasks.models import (
    BenchmarkTask,
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
from benchmark.tasks.validator import validate_task, validate_task_set


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vec(x=0.0, y=0.0, z=0.0) -> Vector3:
    return Vector3(x=x, y=y, z=z)


def _make(
    task_id: str = "geometry_001",
    category: TaskCategory = TaskCategory.GEOMETRY,
    scene: ExpectedScene | None = None,
    tools: list[str] | None = None,
    criteria: list[SuccessCriterion] | None = None,
    prompt: str = "Do the thing.",
) -> BenchmarkTask:
    return BenchmarkTask(
        id=task_id,
        title="Test task",
        category=category,
        difficulty=DifficultyLevel.EASY,
        prompt=prompt,
        tags=["test"],
        allowed_tools=tools if tools is not None else ["create_object"],
        expected_scene=scene
        if scene is not None
        else ExpectedScene(objects=[ExpectedObject(type="MESH")]),
        success_criteria=criteria
        if criteria is not None
        else [SuccessCriterion(metric="object_existence", weight=1.0)],
    )


# ---------------------------------------------------------------------------
# Tool consistency — set_transform
# ---------------------------------------------------------------------------


def test_warns_when_object_has_rotation_but_no_set_transform() -> None:
    scene = ExpectedScene(
        objects=[ExpectedObject(type="MESH", rotation=_vec(0, 0, 45))]
    )
    task = _make(scene=scene, tools=["create_object"])

    warnings = validate_task(task)

    assert any("set_transform" in w for w in warnings)


def test_warns_when_object_has_dimensions_but_no_set_transform() -> None:
    scene = ExpectedScene(
        objects=[ExpectedObject(type="MESH", dimensions=_vec(2, 2, 2))]
    )
    task = _make(scene=scene, tools=["create_object"])

    warnings = validate_task(task)

    assert any("set_transform" in w for w in warnings)


def test_warns_when_object_has_location_but_no_set_transform() -> None:
    scene = ExpectedScene(
        objects=[ExpectedObject(type="MESH", location=_vec(1, 0, 0))]
    )
    task = _make(scene=scene, tools=["create_object"])

    warnings = validate_task(task)

    assert any("set_transform" in w for w in warnings)


def test_no_set_transform_warning_when_tool_present() -> None:
    scene = ExpectedScene(
        objects=[ExpectedObject(type="MESH", rotation=_vec(0, 0, 45))]
    )
    task = _make(scene=scene, tools=["create_object", "set_transform"])

    warnings = validate_task(task)

    assert not any("set_transform" in w for w in warnings)


# ---------------------------------------------------------------------------
# Tool consistency — materials
# ---------------------------------------------------------------------------


def test_warns_when_materials_present_but_no_assign_material() -> None:
    scene = ExpectedScene(
        objects=[ExpectedObject(type="MESH", material="Red")],
        materials=[ExpectedMaterial(name="Red")],
    )
    task = _make(
        task_id="materials_001",
        category=TaskCategory.MATERIALS,
        scene=scene,
        tools=["create_object"],
        criteria=[SuccessCriterion(metric="material_accuracy", weight=1.0)],
    )

    warnings = validate_task(task)

    assert any("assign_material" in w for w in warnings)


def test_warns_when_material_has_params_but_no_set_material_properties() -> None:
    scene = ExpectedScene(
        objects=[ExpectedObject(type="MESH", material="Red")],
        materials=[ExpectedMaterial(name="Red", roughness=0.5)],
    )
    task = _make(
        task_id="materials_001",
        category=TaskCategory.MATERIALS,
        scene=scene,
        tools=["create_object", "assign_material"],
        criteria=[SuccessCriterion(metric="material_accuracy", weight=1.0)],
    )

    warnings = validate_task(task)

    assert any("set_material_properties" in w for w in warnings)


# ---------------------------------------------------------------------------
# Tool consistency — lights
# ---------------------------------------------------------------------------


def test_warns_when_lights_present_but_no_create_light() -> None:
    scene = ExpectedScene(
        objects=[],
        lights=[ExpectedLight(type="AREA")],
    )
    task = _make(
        task_id="lighting_001",
        category=TaskCategory.LIGHTING,
        scene=scene,
        tools=["create_object", "inspect_scene"],
        criteria=[SuccessCriterion(metric="light_existence", weight=1.0)],
    )

    warnings = validate_task(task)

    assert any("create_light" in w for w in warnings)


def test_warns_when_light_has_energy_but_no_set_light_properties() -> None:
    scene = ExpectedScene(
        lights=[ExpectedLight(type="AREA", energy=500.0)],
    )
    task = _make(
        task_id="lighting_001",
        category=TaskCategory.LIGHTING,
        scene=scene,
        tools=["create_object", "create_light", "set_transform"],
        criteria=[SuccessCriterion(metric="light_existence", weight=1.0)],
    )

    warnings = validate_task(task)

    assert any("set_light_properties" in w for w in warnings)


# ---------------------------------------------------------------------------
# Tool consistency — cameras
# ---------------------------------------------------------------------------


def test_warns_when_cameras_present_but_no_camera_tool() -> None:
    scene = ExpectedScene(cameras=[ExpectedCamera()])
    task = _make(
        task_id="camera_001",
        category=TaskCategory.CAMERA,
        scene=scene,
        tools=["create_object"],
        criteria=[SuccessCriterion(metric="camera_existence", weight=1.0)],
    )

    warnings = validate_task(task)

    assert any("create_camera" in w or "set_camera" in w for w in warnings)


# ---------------------------------------------------------------------------
# Tool consistency — exports
# ---------------------------------------------------------------------------


def test_warns_when_exports_present_but_no_export_scene() -> None:
    scene = ExpectedScene(exports=[ExpectedExport(format="blend", filename="result.blend")])
    task = _make(
        task_id="export_001",
        category=TaskCategory.EXPORT,
        scene=scene,
        tools=["create_object"],
        criteria=[SuccessCriterion(metric="export_validity", weight=1.0)],
    )

    warnings = validate_task(task)

    assert any("export_scene" in w for w in warnings)


# ---------------------------------------------------------------------------
# Criteria coverage
# ---------------------------------------------------------------------------


def test_warns_when_geometry_task_has_objects_but_no_object_metric_in_criteria() -> None:
    scene = ExpectedScene(objects=[ExpectedObject(type="MESH")])
    task = _make(
        task_id="geometry_001",
        category=TaskCategory.GEOMETRY,
        scene=scene,
        tools=["create_object"],
        criteria=[SuccessCriterion(metric="export_validity", weight=1.0)],
    )

    warnings = validate_task(task)

    assert any("object_existence" in w or "geometry_accuracy" in w or "object_placement" in w for w in warnings)


def test_no_object_metric_warning_for_lighting_task_with_objects() -> None:
    scene = ExpectedScene(
        objects=[ExpectedObject(type="MESH")],
        lights=[ExpectedLight(type="AREA")],
    )
    task = _make(
        task_id="lighting_001",
        category=TaskCategory.LIGHTING,
        scene=scene,
        tools=["create_object", "create_light", "set_transform", "set_light_properties", "inspect_scene"],
        criteria=[SuccessCriterion(metric="light_existence", weight=1.0)],
    )

    warnings = validate_task(task)

    object_coverage_warnings = [w for w in warnings if "object metric" in w]
    assert not object_coverage_warnings


def test_warns_when_materials_present_but_no_material_metric_in_criteria() -> None:
    scene = ExpectedScene(
        objects=[ExpectedObject(type="MESH", material="Red")],
        materials=[ExpectedMaterial(name="Red")],
    )
    task = _make(
        task_id="materials_001",
        category=TaskCategory.MATERIALS,
        scene=scene,
        tools=["create_object", "assign_material"],
        criteria=[SuccessCriterion(metric="object_existence", weight=1.0)],
    )

    warnings = validate_task(task)

    assert any("material_accuracy" in w or "parameter_correctness" in w for w in warnings)


def test_warns_when_lights_present_but_no_light_metric_in_criteria() -> None:
    scene = ExpectedScene(lights=[ExpectedLight(type="AREA")])
    task = _make(
        task_id="lighting_001",
        category=TaskCategory.LIGHTING,
        scene=scene,
        tools=["create_object", "create_light", "set_transform", "set_light_properties", "inspect_scene"],
        criteria=[SuccessCriterion(metric="object_existence", weight=1.0)],
    )

    warnings = validate_task(task)

    assert any("light_existence" in w or "lighting_correctness" in w for w in warnings)


def test_warns_when_exports_present_but_no_export_validity_in_criteria() -> None:
    scene = ExpectedScene(exports=[ExpectedExport(format="blend", filename="result.blend")])
    task = _make(
        task_id="export_001",
        category=TaskCategory.EXPORT,
        scene=scene,
        tools=["create_object", "export_scene", "inspect_scene"],
        criteria=[SuccessCriterion(metric="object_existence", weight=1.0)],
    )

    warnings = validate_task(task)

    assert any("export_validity" in w for w in warnings)


# ---------------------------------------------------------------------------
# All real task files pass validation with zero warnings
# ---------------------------------------------------------------------------


def test_all_task_files_pass_validate_task_set_with_no_warnings() -> None:
    from pathlib import Path
    from benchmark.tasks.loader import load_tasks_from_dir

    tasks_dir = Path(__file__).resolve().parents[1] / "tasks"
    tasks = load_tasks_from_dir(tasks_dir)
    warnings = validate_task_set(tasks)

    assert warnings == [], f"Unexpected warnings:\n" + "\n".join(warnings)
