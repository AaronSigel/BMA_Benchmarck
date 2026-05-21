"""Tests for camera direction/target validation.

When expected camera has a target, the validator must use direction-based scoring
and must NOT fail for Euler rotation mismatches that are consistent with the correct
pointing direction.
"""
from __future__ import annotations

import math

import pytest

from benchmark.blender.models import CameraSnapshot, SceneSnapshot, Vector3 as BVec3
from benchmark.tasks.models import BenchmarkTask, ExpectedCamera, ExpectedScene, Vector3
from benchmark.validation.validators.camera_validator import CameraValidator


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
    camera_location: tuple,
    camera_rotation_euler: tuple,
    is_active: bool = True,
) -> SceneSnapshot:
    cam = CameraSnapshot(
        name="Camera",
        location=BVec3(x=camera_location[0], y=camera_location[1], z=camera_location[2]),
        rotation_euler=BVec3(
            x=camera_rotation_euler[0],
            y=camera_rotation_euler[1],
            z=camera_rotation_euler[2],
        ),
        lens=None,
        sensor_width=None,
        is_active=is_active,
    )
    return _make_scene_snapshot(cameras=[cam])


def _task_with_camera(
    location: tuple | None = None,
    target: tuple | None = None,
    rotation: tuple | None = None,
    direction_tolerance_deg: float = 15.0,
    name: str = "Camera",
) -> BenchmarkTask:
    cam_kwargs: dict = {"name": name, "direction_tolerance_deg": direction_tolerance_deg}
    if location is not None:
        cam_kwargs["location"] = Vector3(x=location[0], y=location[1], z=location[2])
    if target is not None:
        cam_kwargs["target"] = Vector3(x=target[0], y=target[1], z=target[2])
    if rotation is not None:
        cam_kwargs["rotation"] = Vector3(x=rotation[0], y=rotation[1], z=rotation[2])
    cam = ExpectedCamera(**cam_kwargs)
    task = BenchmarkTask(
        id="camera_003",
        title="Test camera direction",
        category="camera",
        difficulty="easy",
        prompt="Place camera pointing at origin",
        tags=["camera"],
        allowed_tools=["bma_create_camera"],
        expected_scene=ExpectedScene(cameras=[cam]),
        success_criteria=[],
    )
    return task


def test_camera_target_direction_passes_with_correct_pointing() -> None:
    """Camera at (0,5,5) pointing toward origin (0,0,0) should pass direction check."""
    import math
    # Camera at (0, 5, 5), should point toward (0,0,0)
    # For a 45° downward angle looking from +Y: rotation approx (-45°, 0, 0) in degrees
    rx = -math.radians(45.0)
    snapshot = _snapshot((0.0, 5.0, 5.0), (rx, 0.0, 0.0))
    task = _task_with_camera(
        location=(0.0, 5.0, 5.0),
        target=(0.0, 0.0, 0.0),
        direction_tolerance_deg=20.0,
    )
    validator = CameraValidator()
    result = validator.validate(task, snapshot)
    direction_issues = [i for i in result.issues if "direction" in i.code]
    assert not direction_issues, f"Unexpected direction issues: {direction_issues}"


def test_camera_target_direction_passes_with_different_euler() -> None:
    """Camera pointing correctly at target must pass even if Euler differs from expected.

    The camera_validator skips rotation check when target is set; only direction matters.
    """
    import math
    # Camera at (7, 0, 0) pointing roughly toward origin.
    # The 'correct' Euler would be ~(0, pi/2, 0) but we give it (0.1, 1.57, 0.1) - slight deviation.
    # As long as the angular deviation to the target is within tolerance, it should pass.
    snapshot = _snapshot((7.0, 0.0, 0.0), (0.05, math.pi / 2.0 + 0.05, 0.05))
    task = _task_with_camera(
        location=(7.0, 0.0, 0.0),
        target=(0.0, 0.0, 0.0),
        direction_tolerance_deg=20.0,  # generous tolerance
    )
    validator = CameraValidator()
    result = validator.validate(task, snapshot)
    rotation_issues = [i for i in result.issues if "rotation" in i.code]
    # When target is present, rotation_mismatch must NOT be raised
    assert not rotation_issues, f"rotation_mismatch must not fire when target is set: {rotation_issues}"


def test_camera_wrong_direction_fails() -> None:
    """Camera pointing in completely the wrong direction from the target must fail."""
    import math
    # Camera at (0, 5, 5) but rotated so it points away from origin (upward, e.g.)
    rx = math.radians(90.0)  # pointing upward, not toward origin
    snapshot = _snapshot((0.0, 5.0, 5.0), (rx, 0.0, 0.0))
    task = _task_with_camera(
        location=(0.0, 5.0, 5.0),
        target=(0.0, 0.0, 0.0),
        direction_tolerance_deg=5.0,  # tight tolerance
    )
    validator = CameraValidator()
    result = validator.validate(task, snapshot)
    direction_issues = [i for i in result.issues if "direction" in i.code]
    assert direction_issues, "Expected camera_direction_mismatch issue, got none"


def test_camera_without_target_checks_rotation() -> None:
    """When no target is set, rotation mismatch should still be checked."""
    snapshot = _snapshot((0.0, 0.0, 5.0), (0.1, 0.2, 0.3))
    task = _task_with_camera(
        location=(0.0, 0.0, 5.0),
        rotation=(0.0, 0.0, 0.0),  # expected rotation is 0,0,0
        target=None,
        direction_tolerance_deg=5.0,
    )
    validator = CameraValidator()
    result = validator.validate(task, snapshot)
    rotation_issues = [i for i in result.issues if "rotation" in i.code]
    assert rotation_issues, "Rotation mismatch should be detected when no target is set"
