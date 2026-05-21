import math

import pytest

from benchmark.blender.models import (
    CameraSnapshot,
    ObjectSnapshot,
    RenderSettingsSnapshot,
    SceneSnapshot,
    Vector3 as SnapshotVector3,
)
from benchmark.tasks.models import (
    BenchmarkTask,
    DifficultyLevel,
    ExpectedCamera,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
    Vector3,
)
from benchmark.validation.models import ValidationSeverity, ValidationStatus
from benchmark.validation.validators.camera_validator import CameraValidator


def vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector3:
    return Vector3(x=x, y=y, z=z)


def snapshot_vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> SnapshotVector3:
    return SnapshotVector3(x=x, y=y, z=z)


def camera_snapshot(
    name: str,
    location: SnapshotVector3 | None = None,
    rotation_euler: SnapshotVector3 | None = None,
    lens: float | None = 50.0,
    is_active: bool = True,
) -> CameraSnapshot:
    return CameraSnapshot(
        name=name,
        location=location or snapshot_vector(),
        rotation_euler=rotation_euler or snapshot_vector(),
        lens=lens,
        sensor_width=36.0,
        is_active=is_active,
    )


def object_snapshot(name: str, location: SnapshotVector3) -> ObjectSnapshot:
    return ObjectSnapshot(
        name=name,
        type="MESH",
        primitive_hint=None,
        location=location,
        rotation_euler=snapshot_vector(),
        scale=snapshot_vector(1.0, 1.0, 1.0),
        dimensions=snapshot_vector(1.0, 1.0, 1.0),
        material_slots=[],
        parent=None,
        collection_names=["Collection"],
        vertex_count=None,
        polygon_count=None,
    )


def scene_snapshot(cameras: list[CameraSnapshot], objects: list[ObjectSnapshot] | None = None) -> SceneSnapshot:
    return SceneSnapshot(
        scene_name="Scene",
        objects=objects or [],
        materials=[],
        lights=[],
        cameras=cameras,
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


def task_with_cameras(expected_cameras: list[ExpectedCamera]) -> BenchmarkTask:
    return BenchmarkTask(
        id="camera_001_front_view",
        title="Create camera",
        category=TaskCategory.CAMERA,
        difficulty=DifficultyLevel.EASY,
        prompt="Create the expected camera.",
        tags=["camera"],
        allowed_tools=[],
        expected_scene=ExpectedScene(cameras=expected_cameras),
        success_criteria=[SuccessCriterion(metric="camera", weight=1.0)],
    )


def metric_score(result, name: str) -> float:
    return next(metric.score for metric in result.metrics if metric.name == name)


def test_camera_validator_skips_empty_expected_cameras() -> None:
    result = CameraValidator().validate(task_with_cameras([]), scene_snapshot([]))

    assert result.status is ValidationStatus.SKIPPED
    assert result.score == 0.0
    assert result.issues == []
    assert result.metrics == []


def test_camera_validator_passes_matching_active_camera() -> None:
    task = task_with_cameras(
        [
            ExpectedCamera(
                name="Render Camera",
                location=vector(0.0, -5.0, 3.0),
                rotation=vector(math.degrees(1.0), 0.0, 0.0),
                focal_length=50.0,
                tolerance=0.1,
            )
        ]
    )
    snapshot = scene_snapshot(
        [
            camera_snapshot(
                "Render_Camera.001",
                location=snapshot_vector(0.0, -5.0, 3.0),
                rotation_euler=snapshot_vector(1.0, 0.0, 0.0),
                lens=50.0,
                is_active=True,
            )
        ]
    )

    result = CameraValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
    assert result.issues == []


def test_camera_validator_uses_any_candidate_when_name_is_absent() -> None:
    task = task_with_cameras([ExpectedCamera(focal_length=35.0, tolerance=0.1)])
    snapshot = scene_snapshot(
        [
            camera_snapshot("Camera A", lens=50.0, is_active=False),
            camera_snapshot("Camera B", lens=35.0, is_active=True),
        ]
    )

    result = CameraValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.PASSED
    assert metric_score(result, "camera_focal_length_score") == 1.0


def test_camera_validator_compares_focal_length_to_lens() -> None:
    task = task_with_cameras([ExpectedCamera(name="Camera", focal_length=50.0, tolerance=10.0)])
    snapshot = scene_snapshot([camera_snapshot("Camera", lens=65.0, is_active=True)])

    result = CameraValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert metric_score(result, "camera_focal_length_score") == pytest.approx(0.5)
    assert result.issues[0].code == "camera_focal_length_mismatch"
    assert result.issues[0].actual_path == "snapshot.cameras[0].lens"


def test_camera_validator_checks_location_with_tolerance() -> None:
    task = task_with_cameras(
        [ExpectedCamera(name="Camera", location=vector(0.0, 0.0, 0.0), tolerance=1.0)]
    )
    snapshot = scene_snapshot(
        [camera_snapshot("Camera", location=snapshot_vector(1.5, 0.0, 0.0), is_active=True)]
    )

    result = CameraValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert metric_score(result, "camera_transform_score") == pytest.approx(0.5)
    assert result.issues[0].code == "camera_location_mismatch"


def test_camera_validator_checks_rotation_with_tolerance() -> None:
    task = task_with_cameras(
        [ExpectedCamera(name="Camera", rotation=vector(0.0, 0.0, 0.0), tolerance=1.0)]
    )
    snapshot = scene_snapshot(
        [camera_snapshot("Camera", rotation_euler=snapshot_vector(0.0, 1.5, 0.0), is_active=True)]
    )

    result = CameraValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert metric_score(result, "camera_transform_score") == pytest.approx(0.5)
    assert result.issues[0].code == "camera_rotation_mismatch"


def test_camera_validator_checks_look_at_target_instead_of_euler_when_target_is_set() -> None:
    task = task_with_cameras(
        [
            ExpectedCamera(
                name="Front_Camera",
                location=vector(0.0, -6.0, 2.0),
                rotation=vector(0.0, 0.0, 0.0),
                target=vector(0.0, 0.0, 1.0),
                focal_length=35.0,
                require_active=True,
                direction_tolerance_deg=5.0,
                tolerance=0.1,
            )
        ]
    )
    snapshot = scene_snapshot(
        [
            camera_snapshot(
                "Front_Camera",
                location=snapshot_vector(0.0, -6.0, 2.0),
                rotation_euler=snapshot_vector(math.atan2(6.0, 1.0), 0.0, 0.0),
                lens=35.0,
                is_active=True,
            )
        ]
    )

    result = CameraValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.PASSED
    assert metric_score(result, "camera_direction_score") == 1.0
    assert not any(issue.code == "camera_rotation_mismatch" for issue in result.issues)


def test_camera_validator_reports_target_direction_mismatch() -> None:
    task = task_with_cameras(
        [ExpectedCamera(name="Camera", target=vector(0.0, 0.0, 1.0), direction_tolerance_deg=5.0)]
    )
    snapshot = scene_snapshot(
        [
            camera_snapshot(
                "Camera",
                location=snapshot_vector(0.0, -6.0, 2.0),
                rotation_euler=snapshot_vector(0.0, 0.0, 0.0),
                is_active=True,
            )
        ]
    )

    result = CameraValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert metric_score(result, "camera_direction_score") < 1.0
    assert any(issue.code == "camera_direction_mismatch" for issue in result.issues)


def test_camera_validator_resolves_string_target_from_snapshot_object() -> None:
    task = task_with_cameras(
        [ExpectedCamera(name="Camera", target="Center_Sphere", direction_tolerance_deg=5.0)]
    )
    snapshot = scene_snapshot(
        [
            camera_snapshot(
                "Camera",
                location=snapshot_vector(0.0, -6.0, 2.0),
                rotation_euler=snapshot_vector(math.atan2(6.0, 1.0), 0.0, 0.0),
                is_active=True,
            )
        ],
        objects=[object_snapshot("Center_Sphere", snapshot_vector(0.0, 0.0, 1.0))],
    )

    result = CameraValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.PASSED
    assert metric_score(result, "camera_direction_score") == 1.0


def test_single_expected_camera_must_be_active() -> None:
    task = task_with_cameras([ExpectedCamera(name="Camera")])
    snapshot = scene_snapshot([camera_snapshot("Camera", is_active=False)])

    result = CameraValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert metric_score(result, "active_camera_score") == 0.0
    assert result.issues[0].code == "active_camera_mismatch"
    assert result.issues[0].severity is ValidationSeverity.ERROR


def test_camera_validator_reports_missing_camera() -> None:
    task = task_with_cameras([ExpectedCamera(name="Missing", focal_length=50.0)])

    result = CameraValidator().validate(task, scene_snapshot([]))

    assert result.status is ValidationStatus.FAILED
    assert metric_score(result, "camera_existence_score") == 0.0
    assert result.issues[0].code == "camera_missing"
    assert result.issues[0].expected_path == "expected_scene.cameras[0]"
    assert result.issues[0].message


def test_unchecked_camera_fields_do_not_affect_score() -> None:
    task = task_with_cameras([ExpectedCamera(name="Camera")])
    snapshot = scene_snapshot(
        [
            camera_snapshot(
                "Camera",
                location=snapshot_vector(99.0, 99.0, 99.0),
                rotation_euler=snapshot_vector(99.0, 99.0, 99.0),
                lens=999.0,
                is_active=True,
            )
        ]
    )

    result = CameraValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
    assert metric_score(result, "camera_transform_score") == 1.0
    assert metric_score(result, "camera_focal_length_score") == 1.0
