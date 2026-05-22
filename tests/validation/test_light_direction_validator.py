"""Tests for light direction validation.

For SUN/AREA/SPOT lights with a target, direction deviation must be checked.
When direction is correct, Euler mismatch alone must not be fatal.
"""
from __future__ import annotations

import math

import pytest

from benchmark.blender.models import LightSnapshot, ObjectSnapshot, SceneSnapshot, Vector3 as BVec3
from benchmark.tasks.models import BenchmarkTask, ExpectedLight, ExpectedScene, Vector3
from benchmark.validation.validators.light_validator import LightValidator


def _make_scene_snapshot(**kwargs) -> SceneSnapshot:
    from benchmark.blender.models import RenderSettingsSnapshot
    import datetime
    defaults = dict(
        scene_name="Scene",
        objects=[],
        materials=[],
        lights=[],
        cameras=[],
        collections=[],
        render_settings=RenderSettingsSnapshot(
            engine="CYCLES",
            resolution_x=1920,
            resolution_y=1080,
            frame_start=1,
            frame_end=250,
            frame_current=1,
        ),
        frame_current=1,
        blender_version="4.0.0",
        created_at=datetime.datetime.now().isoformat(),
    )
    defaults.update(kwargs)
    return SceneSnapshot(**defaults)


def _snapshot(
    light_type: str,
    location: tuple,
    rotation_euler: tuple,
    name: str = "Light",
    energy: float = 100.0,
) -> SceneSnapshot:
    light = LightSnapshot(
        name=name,
        type=light_type.upper(),
        location=BVec3(x=location[0], y=location[1], z=location[2]),
        rotation_euler=BVec3(x=rotation_euler[0], y=rotation_euler[1], z=rotation_euler[2]),
        energy=energy,
        color=None,
    )
    return _make_scene_snapshot(lights=[light])


def _task_with_light(
    light_type: str,
    location: tuple | None = None,
    target: tuple | None = None,
    rotation: tuple | None = None,
    direction_tolerance_deg: float = 15.0,
    energy: float | None = None,
    name: str = "Light",
) -> BenchmarkTask:
    kwargs: dict = {
        "name": name,
        "type": light_type.upper(),
        "direction_tolerance_deg": direction_tolerance_deg,
    }
    if location is not None:
        kwargs["location"] = Vector3(x=location[0], y=location[1], z=location[2])
    if target is not None:
        kwargs["target"] = Vector3(x=target[0], y=target[1], z=target[2])
    if rotation is not None:
        kwargs["rotation"] = Vector3(x=rotation[0], y=rotation[1], z=rotation[2])
    if energy is not None:
        kwargs["energy"] = energy
    light = ExpectedLight(**kwargs)
    task = BenchmarkTask(
        id="lighting_001",
        title="Test light direction",
        category="lighting",
        difficulty="easy",
        prompt="Create a light pointing at origin",
        tags=["lighting"],
        allowed_tools=["bma_create_light"],
        expected_scene=ExpectedScene(lights=[light]),
        success_criteria=[],
    )
    return task


# ---------------------------------------------------------------------------
# AREA light direction tests
# ---------------------------------------------------------------------------

def test_area_light_direction_passes_when_euler_differs() -> None:
    """Correct direction passes even when raw Euler differs from expected rotation."""
    snapshot = _snapshot("AREA", (0.0, -3.0, 5.0), (math.radians(45.0), 0.0, 0.0), name="Key_Area_Light", energy=500.0)
    task = _task_with_light(
        "AREA",
        location=(0.0, -3.0, 5.0),
        target=(0.0, 0.0, 0.0),
        rotation=(60.0, 0.0, 0.0),
        direction_tolerance_deg=15.0,
        energy=500.0,
        name="Key_Area_Light",
    )
    result = LightValidator().validate(task, snapshot)
    fatal = [i for i in result.issues if i.severity.value == "error"]
    assert not any(i.code == "light_direction_mismatch" for i in fatal)
    assert result.status.value == "passed"


def test_area_light_direction_passes_with_different_euler() -> None:
    """AREA light with correct direction to target passes even if Euler differs slightly.

    Blender lights default to pointing in the local -Z direction.
    rotation=(0,0,0) → world direction (0,0,-1) → pointing toward origin from (0,0,5).
    A slight rx deviation of 0.1 rad (~5.7°) is within 20° tolerance.
    """
    rx = 0.1  # ~5.7° deviation from straight-down
    snapshot = _snapshot("AREA", (0.0, 0.0, 5.0), (rx, 0.0, 0.0))
    task = _task_with_light(
        "AREA",
        location=(0.0, 0.0, 5.0),
        target=(0.0, 0.0, 0.0),
        direction_tolerance_deg=20.0,
    )
    validator = LightValidator()
    result = validator.validate(task, snapshot)
    direction_issues = [i for i in result.issues if "direction" in i.code]
    assert not direction_issues, f"Unexpected direction issues: {[i.message for i in direction_issues]}"


def test_area_light_direction_within_grace_is_warning_not_blocking() -> None:
    """Deviation slightly above tolerance but within grace band is WARNING, not ERROR."""
    rx = math.radians(17.0)
    snapshot = _snapshot("AREA", (0.0, 0.0, 5.0), (rx, 0.0, 0.0))
    task = _task_with_light(
        "AREA",
        location=(0.0, 0.0, 5.0),
        target=(0.0, 0.0, 0.0),
        direction_tolerance_deg=15.0,
    )
    result = LightValidator().validate(task, snapshot)
    direction_issues = [i for i in result.issues if i.code == "light_direction_mismatch"]
    assert direction_issues, "Expected light_direction_mismatch within grace band"
    assert direction_issues[0].severity.value == "warning"
    assert result.status.value == "passed"


def test_area_light_wrong_direction_fails() -> None:
    """AREA light pointing sideways (not toward target) must fail direction check.

    rotation=(pi/2, 0, 0) rotates the default -Z direction to +Y (90° away from -Z).
    The expected target direction is (0,0,-1). Deviation = 90° >> 5° tolerance.
    """
    snapshot = _snapshot("AREA", (0.0, 0.0, 5.0), (math.pi / 2.0, 0.0, 0.0))
    task = _task_with_light(
        "AREA",
        location=(0.0, 0.0, 5.0),
        target=(0.0, 0.0, 0.0),
        direction_tolerance_deg=5.0,
    )
    validator = LightValidator()
    result = validator.validate(task, snapshot)
    direction_issues = [i for i in result.issues if "direction" in i.code]
    assert direction_issues, "Expected light_direction_mismatch but got none"


# ---------------------------------------------------------------------------
# SUN light direction tests
# ---------------------------------------------------------------------------

def test_sun_direction_mismatch_fails() -> None:
    """Alias for SUN wrong-direction failure."""
    test_sun_wrong_direction_fails()


def test_sun_wrong_direction_fails() -> None:
    """SUN light pointing in wrong direction must produce direction_mismatch issue.

    rotation=(pi, 0, 0) rotates -Z by 180° around X → (0, 0, +1) (pointing up).
    From (0,0,10) toward (0,0,0) the expected direction is (0,0,-1).
    Deviation = 180° >> 5° tolerance → FAIL.
    """
    snapshot = _snapshot("SUN", (0.0, 0.0, 10.0), (math.pi, 0.0, 0.0))
    task = _task_with_light(
        "SUN",
        location=(0.0, 0.0, 10.0),
        target=(0.0, 0.0, 0.0),
        direction_tolerance_deg=5.0,
    )
    validator = LightValidator()
    result = validator.validate(task, snapshot)
    direction_issues = [i for i in result.issues if "direction" in i.code]
    assert direction_issues, f"Expected direction_mismatch for SUN, got none"


def test_sun_correct_direction_passes() -> None:
    """SUN light pointing correctly at target must pass direction check.

    Default rotation=(0,0,0) gives world direction (0,0,-1).
    From (0,0,5) toward (0,0,0) is also (0,0,-1). Deviation = 0° → PASS.
    """
    snapshot = _snapshot("SUN", (0.0, 0.0, 5.0), (0.0, 0.0, 0.0))
    task = _task_with_light(
        "SUN",
        location=(0.0, 0.0, 5.0),
        target=(0.0, 0.0, 0.0),
        direction_tolerance_deg=20.0,
    )
    validator = LightValidator()
    result = validator.validate(task, snapshot)
    direction_issues = [i for i in result.issues if "direction" in i.code]
    assert not direction_issues, f"Direction should pass: {[i.message for i in direction_issues]}"


def test_three_point_lighting_matches_named_lights() -> None:
    """Named Key/Fill/Back lights are matched individually; direction/energy checked."""
    import yaml
    from pathlib import Path

    task = BenchmarkTask.model_validate(
        yaml.safe_load(Path("tasks/lighting/lighting_003_three_point_lighting.yaml").read_text(encoding="utf-8"))
    )
    snapshot = _make_scene_snapshot(
        objects=[
            ObjectSnapshot(
                name="Center_Object",
                type="MESH",
                primitive_hint="sphere",
                location=BVec3(x=0, y=0, z=1),
                rotation_euler=BVec3(x=0, y=0, z=0),
                scale=BVec3(x=1, y=1, z=1),
                dimensions=BVec3(x=1.5, y=1.5, z=1.5),
                material_slots=[],
                parent=None,
                collection_names=["Collection"],
                vertex_count=100,
                polygon_count=50,
            )
        ],
        lights=[
            LightSnapshot(
                name="Key_Light",
                type="AREA",
                location=BVec3(x=-3, y=-4, z=5),
                rotation_euler=BVec3(x=math.radians(60), y=0, z=math.radians(-35)),
                energy=700.0,
                color=None,
            ),
            LightSnapshot(
                name="Fill_Light",
                type="AREA",
                location=BVec3(x=4, y=-3, z=3),
                rotation_euler=BVec3(x=math.radians(55), y=0, z=math.radians(45)),
                energy=250.0,
                color=None,
            ),
            LightSnapshot(
                name="Back_Light",
                type="SPOT",
                location=BVec3(x=0, y=4, z=4),
                rotation_euler=BVec3(x=math.radians(55), y=0, z=math.radians(180)),
                energy=450.0,
                color=None,
            ),
        ],
    )
    result = LightValidator().validate(task, snapshot)
    missing = [i for i in result.issues if i.code == "light_missing"]
    assert not missing, f"Unexpected missing lights: {missing}"
    assert result.status.value == "passed"


def test_three_point_lighting_fallback_matches_by_location_type() -> None:
    """When names differ, lights are matched by type + nearest location."""
    snapshot = _make_scene_snapshot(
        lights=[
            LightSnapshot(
                name="RenamedKey",
                type="AREA",
                location=BVec3(x=-3, y=-4, z=5),
                rotation_euler=BVec3(x=math.radians(60), y=0, z=math.radians(-35)),
                energy=700.0,
                color=None,
            ),
            LightSnapshot(
                name="RenamedFill",
                type="AREA",
                location=BVec3(x=4, y=-3, z=3),
                rotation_euler=BVec3(x=math.radians(55), y=0, z=math.radians(45)),
                energy=250.0,
                color=None,
            ),
            LightSnapshot(
                name="RenamedBack",
                type="SPOT",
                location=BVec3(x=0, y=4, z=4),
                rotation_euler=BVec3(x=math.radians(55), y=0, z=math.radians(180)),
                energy=450.0,
                color=None,
            ),
        ]
    )
    task = BenchmarkTask(
        id="lighting_003",
        title="Three point",
        category="lighting",
        difficulty="medium",
        prompt="three point lighting",
        tags=["lighting"],
        allowed_tools=["bma_create_light"],
        expected_scene=ExpectedScene(
            lights=[
                ExpectedLight(
                    name="Key_Light",
                    type="AREA",
                    location=Vector3(x=-3, y=-4, z=5),
                    rotation=Vector3(x=60, y=0, z=-35),
                    energy=700.0,
                    tolerance=0.2,
                    direction_tolerance_deg=15.0,
                ),
                ExpectedLight(
                    name="Fill_Light",
                    type="AREA",
                    location=Vector3(x=4, y=-3, z=3),
                    rotation=Vector3(x=55, y=0, z=45),
                    energy=250.0,
                    tolerance=0.2,
                    direction_tolerance_deg=15.0,
                ),
                ExpectedLight(
                    name="Back_Light",
                    type="SPOT",
                    location=Vector3(x=0, y=4, z=4),
                    rotation=Vector3(x=55, y=0, z=180),
                    energy=450.0,
                    tolerance=0.2,
                    direction_tolerance_deg=15.0,
                ),
            ]
        ),
        success_criteria=[],
    )
    result = LightValidator().validate(task, snapshot)
    assert not [i for i in result.issues if i.code == "light_missing"]
    assert result.status.value == "passed"
