import pytest

from benchmark.blender.models import (
    ColorRGBA,
    LightSnapshot,
    RenderSettingsSnapshot,
    SceneSnapshot,
    Vector3 as SnapshotVector3,
)
from benchmark.tasks.models import (
    BenchmarkTask,
    DifficultyLevel,
    ExpectedLight,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
    Vector3,
)
from benchmark.validation.models import ValidationStatus
from benchmark.validation.validators.light_validator import LightValidator


def vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector3:
    return Vector3(x=x, y=y, z=z)


def snapshot_vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> SnapshotVector3:
    return SnapshotVector3(x=x, y=y, z=z)


def light_snapshot(
    name: str,
    type_: str = "AREA",
    location: SnapshotVector3 | None = None,
    rotation_euler: SnapshotVector3 | None = None,
    energy: float | None = 500.0,
) -> LightSnapshot:
    return LightSnapshot(
        name=name,
        type=type_,
        location=location or snapshot_vector(),
        rotation_euler=rotation_euler or snapshot_vector(),
        energy=energy,
        color=ColorRGBA(r=1.0, g=1.0, b=1.0),
    )


def scene_snapshot(lights: list[LightSnapshot]) -> SceneSnapshot:
    return SceneSnapshot(
        scene_name="Scene",
        objects=[],
        materials=[],
        lights=lights,
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


def task_with_lights(expected_lights: list[ExpectedLight]) -> BenchmarkTask:
    return BenchmarkTask(
        id="lighting_001_area_light",
        title="Create lights",
        category=TaskCategory.LIGHTING,
        difficulty=DifficultyLevel.EASY,
        prompt="Create the expected lights.",
        tags=["lighting"],
        allowed_tools=[],
        expected_scene=ExpectedScene(lights=expected_lights),
        success_criteria=[SuccessCriterion(metric="lights", weight=1.0)],
    )


def metric_score(result, name: str) -> float:
    return next(metric.score for metric in result.metrics if metric.name == name)


def test_light_validator_skips_empty_expected_lights() -> None:
    result = LightValidator().validate(task_with_lights([]), scene_snapshot([]))

    assert result.status is ValidationStatus.SKIPPED
    assert result.score == 0.0
    assert result.issues == []
    assert result.metrics == []


def test_light_validator_passes_matching_area_light() -> None:
    task = task_with_lights(
        [
            ExpectedLight(
                name="Key Light",
                type="AREA",
                location=vector(0.0, -3.0, 4.0),
                rotation=vector(1.0, 0.0, 0.0),
                energy=500.0,
                tolerance=0.1,
            )
        ]
    )
    snapshot = scene_snapshot(
        [
            light_snapshot(
                "Key_Light.001",
                "AREA",
                location=snapshot_vector(0.0, -3.0, 4.0),
                rotation_euler=snapshot_vector(1.0, 0.0, 0.0),
                energy=500.0,
            )
        ]
    )

    result = LightValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
    assert result.issues == []
    assert metric_score(result, "light_type_score") == 1.0


def test_light_validator_reports_point_area_type_mismatch() -> None:
    task = task_with_lights([ExpectedLight(name="Key", type="POINT")])
    snapshot = scene_snapshot([light_snapshot("Key", "AREA")])

    result = LightValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert metric_score(result, "light_type_score") == 0.0
    assert result.issues[0].code == "light_type_mismatch"
    assert result.issues[0].expected_path == "expected_scene.lights[0].type"
    assert result.issues[0].actual_path == "snapshot.lights[0].type"


def test_light_validator_checks_energy_with_tolerance() -> None:
    task = task_with_lights([ExpectedLight(name="Key", type="AREA", energy=500.0, tolerance=100.0)])
    snapshot = scene_snapshot([light_snapshot("Key", "AREA", energy=650.0)])

    result = LightValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert metric_score(result, "light_energy_score") == pytest.approx(0.5)
    assert result.issues[0].code == "light_energy_mismatch"


def test_light_validator_checks_location_with_tolerance() -> None:
    task = task_with_lights(
        [ExpectedLight(name="Key", type="AREA", location=vector(0.0, 0.0, 0.0), tolerance=1.0)]
    )
    snapshot = scene_snapshot([light_snapshot("Key", "AREA", location=snapshot_vector(1.5, 0.0, 0.0))])

    result = LightValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert metric_score(result, "light_transform_score") == pytest.approx(0.5)
    assert result.issues[0].code == "light_location_mismatch"


def test_light_validator_checks_rotation_with_tolerance() -> None:
    task = task_with_lights(
        [ExpectedLight(name="Key", type="AREA", rotation=vector(0.0, 0.0, 0.0), tolerance=1.0)]
    )
    snapshot = scene_snapshot(
        [light_snapshot("Key", "AREA", rotation_euler=snapshot_vector(0.0, 1.5, 0.0))]
    )

    result = LightValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert metric_score(result, "light_transform_score") == pytest.approx(0.5)
    assert result.issues[0].code == "light_rotation_mismatch"


def test_light_validator_reports_missing_light() -> None:
    task = task_with_lights([ExpectedLight(name="Missing", type="AREA", energy=500.0)])

    result = LightValidator().validate(task, scene_snapshot([]))

    assert result.status is ValidationStatus.FAILED
    assert metric_score(result, "light_existence_score") == 0.0
    assert result.issues[0].code == "light_missing"
    assert result.issues[0].expected_path == "expected_scene.lights[0]"
    assert result.issues[0].message


def test_unchecked_light_fields_do_not_affect_score() -> None:
    task = task_with_lights([ExpectedLight(name="Key", type="AREA")])
    snapshot = scene_snapshot(
        [
            light_snapshot(
                "Key",
                "AREA",
                location=snapshot_vector(99.0, 99.0, 99.0),
                rotation_euler=snapshot_vector(99.0, 99.0, 99.0),
                energy=9999.0,
            )
        ]
    )

    result = LightValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
    assert metric_score(result, "light_transform_score") == 1.0
    assert metric_score(result, "light_energy_score") == 1.0
