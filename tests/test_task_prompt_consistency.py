"""Tests that verify task YAML prompt/expected_scene consistency contracts.

Rules checked here (subset that can be verified structurally):
  1. Geometry tasks specify exact names for all expected objects.
  2. Lighting tasks specify exact names for all expected lights.
  3. Camera tasks specify exact names for all expected cameras.
  4. Material tasks include numeric parameter values for every expected material.
  5. Rotation tasks specify rotation values in degrees inside prompt.
  6. Export tasks name the export file explicitly in the prompt.
  7. No task uses `target_visibility` as a success metric (not implemented).
  8. All core task ids from the final matrix are present in the task set.
"""

import re
from pathlib import Path

import pytest

from benchmark.tasks.loader import load_tasks_from_dir
from benchmark.tasks.models import BenchmarkTask, TaskCategory

TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"


@pytest.fixture(scope="module")
def all_tasks() -> list[BenchmarkTask]:
    return load_tasks_from_dir(TASKS_DIR)


def _task_by_id(all_tasks: list[BenchmarkTask], task_id: str) -> BenchmarkTask:
    for task in all_tasks:
        if task.id == task_id:
            return task
    raise KeyError(f"Task not found: {task_id}")


# ---------------------------------------------------------------------------
# 1. Geometry — expected object names appear in prompt
# ---------------------------------------------------------------------------


def test_geometry_object_names_in_prompt(all_tasks: list[BenchmarkTask]) -> None:
    for task in all_tasks:
        if task.category is not TaskCategory.GEOMETRY:
            continue
        for obj in task.expected_scene.objects:
            if obj.name:
                assert obj.name in task.prompt, (
                    f"{task.id}: expected object name '{obj.name}' is not mentioned in the prompt"
                )


# ---------------------------------------------------------------------------
# 2. Lighting — expected light names appear in prompt
# ---------------------------------------------------------------------------


def test_lighting_light_names_in_prompt(all_tasks: list[BenchmarkTask]) -> None:
    for task in all_tasks:
        if task.category is not TaskCategory.LIGHTING:
            continue
        for light in task.expected_scene.lights:
            if light.name:
                assert light.name in task.prompt, (
                    f"{task.id}: expected light name '{light.name}' is not mentioned in the prompt"
                )


# ---------------------------------------------------------------------------
# 3. Camera — expected camera names appear in prompt
# ---------------------------------------------------------------------------


def test_camera_names_in_prompt(all_tasks: list[BenchmarkTask]) -> None:
    for task in all_tasks:
        if task.category is not TaskCategory.CAMERA:
            continue
        for cam in task.expected_scene.cameras:
            if cam.name:
                assert cam.name in task.prompt, (
                    f"{task.id}: expected camera name '{cam.name}' is not mentioned in the prompt"
                )


# ---------------------------------------------------------------------------
# 4. Materials — expected material names appear in prompt and have numeric params
# ---------------------------------------------------------------------------


def test_material_names_in_prompt(all_tasks: list[BenchmarkTask]) -> None:
    for task in all_tasks:
        if task.category is not TaskCategory.MATERIALS:
            continue
        for mat in task.expected_scene.materials:
            assert mat.name in task.prompt, (
                f"{task.id}: expected material name '{mat.name}' is not mentioned in the prompt"
            )


def test_material_numeric_params_present(all_tasks: list[BenchmarkTask]) -> None:
    for task in all_tasks:
        if task.category is not TaskCategory.MATERIALS:
            continue
        for mat in task.expected_scene.materials:
            assert mat.roughness is not None, (
                f"{task.id}: material '{mat.name}' has no roughness value in expected_scene"
            )
            assert mat.metallic is not None, (
                f"{task.id}: material '{mat.name}' has no metallic value in expected_scene"
            )
            assert mat.base_color is not None, (
                f"{task.id}: material '{mat.name}' has no base_color in expected_scene"
            )


# ---------------------------------------------------------------------------
# 5. Rotation tasks — prompt contains degree values when rotation is checked
# ---------------------------------------------------------------------------


_DEGREE_PATTERN = re.compile(r"\b\d+(\.\d+)?\s*(degree|deg|°)\b", re.IGNORECASE)
_NUMERIC_ROTATION = re.compile(r"\brotation\b.*?\(\s*\d", re.IGNORECASE | re.DOTALL)


def _has_numeric_rotation_spec(prompt: str) -> bool:
    return bool(_DEGREE_PATTERN.search(prompt) or _NUMERIC_ROTATION.search(prompt))


def test_rotation_tasks_mention_numeric_rotation_in_prompt(all_tasks: list[BenchmarkTask]) -> None:
    for task in all_tasks:
        objects_with_rotation = [o for o in task.expected_scene.objects if o.rotation is not None]
        lights_with_rotation = [l for l in task.expected_scene.lights if l.rotation is not None]
        cameras_with_rotation = [c for c in task.expected_scene.cameras if c.rotation is not None]

        has_rotation = objects_with_rotation or lights_with_rotation or cameras_with_rotation
        if not has_rotation:
            continue

        assert _has_numeric_rotation_spec(task.prompt), (
            f"{task.id}: expected_scene has rotation values but the prompt does not contain "
            "a numeric rotation specification (e.g. '45 degrees' or 'rotation (x, y, z)')"
        )


# ---------------------------------------------------------------------------
# 6. Export — prompt names the export file explicitly
# ---------------------------------------------------------------------------


def test_export_filename_in_prompt(all_tasks: list[BenchmarkTask]) -> None:
    for task in all_tasks:
        if task.category is not TaskCategory.EXPORT:
            continue
        for export in task.expected_scene.exports:
            if export.filename:
                assert export.filename in task.prompt, (
                    f"{task.id}: expected export filename '{export.filename}' "
                    "is not mentioned in the prompt"
                )


# ---------------------------------------------------------------------------
# 7. No task uses target_visibility (validator not implemented)
# ---------------------------------------------------------------------------


def test_no_task_uses_target_visibility_metric(all_tasks: list[BenchmarkTask]) -> None:
    for task in all_tasks:
        metrics = [c.metric for c in task.success_criteria]
        assert "target_visibility" not in metrics, (
            f"{task.id}: uses target_visibility metric but view-frustum validation "
            "is not implemented in CameraValidator"
        )


# ---------------------------------------------------------------------------
# 8. Final core task set — all recommended task ids exist
# ---------------------------------------------------------------------------


FINAL_CORE_TASK_IDS = [
    "geometry_001_basic_primitives",
    "geometry_002_positions",
    "geometry_003_dimensions",
    "geometry_005_composition",
    "materials_001_basic_colors",
    "materials_002_roughness",
    "materials_004_multiple_objects",
    "lighting_001_area_light",
    "lighting_003_three_point_lighting",
    "camera_001_front_view",
    "camera_003_composition_view",
    "export_001_blend_file",
    "export_002_glb_file",
]


def test_final_core_task_set_ids_present(all_tasks: list[BenchmarkTask]) -> None:
    existing_ids = {task.id for task in all_tasks}
    missing = [tid for tid in FINAL_CORE_TASK_IDS if tid not in existing_ids]
    assert not missing, f"Core task ids missing from task set: {missing}"
