import pytest

from benchmark.blender.models import (
    ObjectSnapshot,
    RenderSettingsSnapshot,
    SceneSnapshot,
    Vector3 as SnapshotVector3,
)
from benchmark.tasks.models import (
    BenchmarkTask,
    DifficultyLevel,
    ExpectedObject,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
    Vector3,
)
from benchmark.validation.models import ValidationStatus
from benchmark.validation.validators.transform_validator import TransformValidator


def vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector3:
    return Vector3(x=x, y=y, z=z)


def snapshot_vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> SnapshotVector3:
    return SnapshotVector3(x=x, y=y, z=z)


def object_snapshot(
    name: str,
    location: SnapshotVector3 | None = None,
    rotation_euler: SnapshotVector3 | None = None,
    scale: SnapshotVector3 | None = None,
    dimensions: SnapshotVector3 | None = None,
) -> ObjectSnapshot:
    return ObjectSnapshot(
        name=name,
        type="MESH",
        primitive_hint="cube",
        location=location or snapshot_vector(),
        rotation_euler=rotation_euler or snapshot_vector(),
        scale=scale or snapshot_vector(1.0, 1.0, 1.0),
        dimensions=dimensions or snapshot_vector(2.0, 2.0, 2.0),
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
        id="geometry_002_positions",
        title="Create positioned objects",
        category=TaskCategory.GEOMETRY,
        difficulty=DifficultyLevel.EASY,
        prompt="Create the expected objects.",
        tags=["geometry"],
        allowed_tools=[],
        expected_scene=ExpectedScene(objects=expected_objects),
        success_criteria=[SuccessCriterion(metric="transforms", weight=1.0)],
    )


def test_transform_validator_scores_one_inside_tolerance() -> None:
    task = task_with_objects(
        [
            ExpectedObject(
                name="Cube",
                type="MESH",
                location=vector(1.0, 2.0, 3.0),
                tolerance=0.1,
            )
        ]
    )
    snapshot = scene_snapshot([object_snapshot("Cube.001", location=snapshot_vector(1.05, 2.0, 3.0))])

    result = TransformValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
    assert result.issues == []


def test_transform_validator_score_decreases_for_large_distance() -> None:
    task = task_with_objects(
        [
            ExpectedObject(
                name="Cube",
                type="MESH",
                location=vector(0.0, 0.0, 0.0),
                tolerance=1.0,
            )
        ]
    )
    snapshot = scene_snapshot([object_snapshot("Cube", location=snapshot_vector(1.5, 0.0, 0.0))])

    result = TransformValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert result.score == pytest.approx(0.5)
    assert result.issues[0].code == "location_mismatch"
    assert result.issues[0].expected_path == "expected_scene.objects[0].location"
    assert result.issues[0].actual_path == "snapshot.objects[0].location"


def test_transform_validator_reports_missing_object() -> None:
    task = task_with_objects(
        [
            ExpectedObject(
                name="Missing",
                type="MESH",
                location=vector(0.0, 0.0, 0.0),
            )
        ]
    )

    result = TransformValidator().validate(task, scene_snapshot([]))

    assert result.status is ValidationStatus.FAILED
    assert result.score == 0.0
    assert result.issues[0].code == "object_missing_for_transform"
    assert result.issues[0].expected_path == "expected_scene.objects[0]"
    assert result.issues[0].message


def test_transform_validator_skips_when_no_transform_fields_are_expected() -> None:
    task = task_with_objects([ExpectedObject(name="Cube", type="MESH")])
    snapshot = scene_snapshot([object_snapshot("Cube", location=snapshot_vector(99.0, 0.0, 0.0))])

    result = TransformValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.SKIPPED
    assert result.score == 0.0
    assert result.issues == []
    assert result.metrics == []


def test_unchecked_fields_do_not_affect_score() -> None:
    task = task_with_objects(
        [
            ExpectedObject(
                name="Cube",
                type="MESH",
                location=vector(0.0, 0.0, 0.0),
                tolerance=0.1,
            )
        ]
    )
    snapshot = scene_snapshot(
        [
            object_snapshot(
                "Cube",
                location=snapshot_vector(0.0, 0.0, 0.0),
                rotation_euler=snapshot_vector(99.0, 99.0, 99.0),
                scale=snapshot_vector(99.0, 99.0, 99.0),
                dimensions=snapshot_vector(99.0, 99.0, 99.0),
            )
        ]
    )

    result = TransformValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
    assert [metric.name for metric in result.metrics] == ["location_score"]


def test_transform_validator_checks_rotation_scale_and_dimensions() -> None:
    task = task_with_objects(
        [
            ExpectedObject(
                name="Cube",
                type="MESH",
                rotation=vector(0.0, 0.0, 0.0),
                scale=vector(1.0, 1.0, 1.0),
                dimensions=vector(2.0, 2.0, 2.0),
                tolerance=0.1,
            )
        ]
    )
    snapshot = scene_snapshot(
        [
            object_snapshot(
                "Cube",
                rotation_euler=snapshot_vector(0.0, 0.0, 0.0),
                scale=snapshot_vector(1.0, 1.0, 1.0),
                dimensions=snapshot_vector(2.0, 2.0, 2.0),
            )
        ]
    )

    result = TransformValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
    assert [metric.name for metric in result.metrics] == [
        "rotation_score",
        "scale_score",
        "dimensions_score",
    ]
