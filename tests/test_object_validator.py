import pytest

from benchmark.blender.models import (
    ObjectSnapshot,
    RenderSettingsSnapshot,
    SceneSnapshot,
    Vector3,
)
from benchmark.tasks.models import (
    BenchmarkTask,
    DifficultyLevel,
    ExpectedObject,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
)
from benchmark.validation.models import ValidationStatus
from benchmark.validation.validators.object_validator import ObjectValidator


def vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector3:
    return Vector3(x=x, y=y, z=z)


def object_snapshot(
    name: str,
    type_: str = "MESH",
    primitive_hint: str | None = "cube",
) -> ObjectSnapshot:
    return ObjectSnapshot(
        name=name,
        type=type_,
        primitive_hint=primitive_hint,
        location=vector(),
        rotation_euler=vector(),
        scale=vector(1.0, 1.0, 1.0),
        dimensions=vector(2.0, 2.0, 2.0),
        material_slots=[],
        parent=None,
        collection_names=["Collection"],
        vertex_count=None,
        polygon_count=None,
    )


def scene_snapshot(objects: list[ObjectSnapshot]) -> SceneSnapshot:
    return SceneSnapshot(
        scene_name="Scene",
        objects=objects,
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


def task_with_objects(expected_objects: list[ExpectedObject]) -> BenchmarkTask:
    return BenchmarkTask(
        id="geometry_001_basic_primitives",
        title="Create primitives",
        category=TaskCategory.GEOMETRY,
        difficulty=DifficultyLevel.EASY,
        prompt="Create the expected primitives.",
        tags=["geometry"],
        allowed_tools=[],
        expected_scene=ExpectedScene(objects=expected_objects),
        success_criteria=[SuccessCriterion(metric="objects", weight=1.0)],
    )


def metric_score(result, name: str) -> float:
    return next(metric.score for metric in result.metrics if metric.name == name)


def test_object_validator_passes_when_all_objects_match() -> None:
    task = task_with_objects(
        [
            ExpectedObject(name="Cube", type="MESH", primitive="cube"),
            ExpectedObject(name="Sphere", type="MESH", primitive="sphere"),
        ]
    )
    snapshot = scene_snapshot(
        [
            object_snapshot("Cube.001", primitive_hint="cube"),
            object_snapshot("Sphere", primitive_hint="sphere"),
        ]
    )

    result = ObjectValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
    assert result.issues == []
    assert metric_score(result, "object_existence_score") == 1.0
    assert metric_score(result, "type_score") == 1.0
    assert metric_score(result, "primitive_score") == 1.0


def test_object_validator_score_decreases_when_one_of_two_objects_is_missing() -> None:
    task = task_with_objects(
        [
            ExpectedObject(name="Cube", type="MESH", primitive="cube"),
            ExpectedObject(name="Sphere", type="MESH", primitive="sphere"),
        ]
    )
    snapshot = scene_snapshot([object_snapshot("Cube", primitive_hint="cube")])

    result = ObjectValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert result.score < 1.0
    assert result.score == pytest.approx(0.7)
    assert result.issues[0].code == "object_missing"
    assert result.issues[0].severity.value == "error"
    assert result.issues[0].expected_path == "expected_scene.objects[1]"
    assert result.issues[0].message


def test_object_validator_skips_empty_expected_objects() -> None:
    result = ObjectValidator().validate(task_with_objects([]), scene_snapshot([]))

    assert result.status is ValidationStatus.SKIPPED
    assert result.score == 0.0
    assert result.issues == []
    assert result.metrics == []


def test_object_validator_reports_type_mismatch() -> None:
    task = task_with_objects([ExpectedObject(name="Key", type="LIGHT")])
    snapshot = scene_snapshot([object_snapshot("Key", type_="MESH", primitive_hint=None)])

    result = ObjectValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert result.score == pytest.approx(0.8)
    assert result.issues[0].code == "object_type_mismatch"
    assert result.issues[0].expected_path == "expected_scene.objects[0].type"
    assert result.issues[0].actual_path == "snapshot.objects[0].type"
    assert result.issues[0].message


def test_object_validator_reports_primitive_mismatch() -> None:
    task = task_with_objects([ExpectedObject(name="Cube", type="MESH", primitive="cube")])
    snapshot = scene_snapshot([object_snapshot("Cube", primitive_hint="sphere")])

    result = ObjectValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert result.score == pytest.approx(0.8)
    assert result.issues[0].code == "primitive_mismatch"
    assert result.issues[0].expected_path == "expected_scene.objects[0].primitive"
    assert result.issues[0].actual_path == "snapshot.objects[0].primitive_hint"
    assert result.issues[0].message


def test_object_validator_does_not_penalize_absent_primitive_expectations() -> None:
    task = task_with_objects([ExpectedObject(name="Generated Mesh", type="MESH")])
    snapshot = scene_snapshot([object_snapshot("Generated Mesh", type_="MESH", primitive_hint=None)])

    result = ObjectValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
    assert metric_score(result, "primitive_score") == 1.0
