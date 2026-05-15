import sys

import pytest
from pydantic import ValidationError

from benchmark.blender.models import (
    BlenderCommandResult,
    CameraSnapshot,
    ColorRGBA,
    LightSnapshot,
    MaterialSnapshot,
    ObjectSnapshot,
    RenderSettingsSnapshot,
    SceneSnapshot,
    Vector3,
)


def vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector3:
    return Vector3(x=x, y=y, z=z)


def make_scene_snapshot() -> SceneSnapshot:
    return SceneSnapshot(
        scene_name="Scene",
        objects=[
            ObjectSnapshot(
                name="Cube",
                type="MESH",
                primitive_hint="cube",
                location=vector(),
                rotation_euler=vector(),
                scale=vector(1.0, 1.0, 1.0),
                dimensions=vector(2.0, 2.0, 2.0),
                material_slots=["Red"],
                parent=None,
                collection_names=["Collection"],
                vertex_count=8,
                polygon_count=6,
            )
        ],
        materials=[
            MaterialSnapshot(
                name="Red",
                base_color=ColorRGBA(r=1.0, g=0.0, b=0.0),
                roughness=0.5,
                metallic=0.0,
                use_nodes=True,
            )
        ],
        lights=[
            LightSnapshot(
                name="Key",
                type="AREA",
                location=vector(0.0, -3.0, 4.0),
                rotation_euler=vector(),
                energy=500.0,
                color=ColorRGBA(r=1.0, g=1.0, b=1.0),
            )
        ],
        cameras=[
            CameraSnapshot(
                name="Camera",
                location=vector(0.0, -5.0, 3.0),
                rotation_euler=vector(1.1, 0.0, 0.0),
                lens=50.0,
                sensor_width=36.0,
                is_active=True,
            )
        ],
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


def test_blender_models_import_without_bpy() -> None:
    assert "bpy" not in sys.modules
    assert Vector3(x=1.0, y=2.0, z=3.0).z == 3.0


def test_scene_snapshot_json_round_trip() -> None:
    snapshot = make_scene_snapshot()

    raw_json = snapshot.model_dump_json()
    restored = SceneSnapshot.model_validate_json(raw_json)

    assert restored == snapshot


@pytest.mark.parametrize("channel", ["r", "g", "b", "a"])
def test_color_rgba_channels_must_be_between_zero_and_one(channel: str) -> None:
    data = {"r": 0.5, "g": 0.5, "b": 0.5, "a": 0.5}
    data[channel] = 1.1

    with pytest.raises(ValidationError):
        ColorRGBA(**data)


@pytest.mark.parametrize("field", ["resolution_x", "resolution_y"])
def test_render_resolution_must_be_positive(field: str) -> None:
    data = {
        "engine": "CYCLES",
        "resolution_x": 1920,
        "resolution_y": 1080,
        "frame_start": 1,
        "frame_end": 1,
        "frame_current": 1,
    }
    data[field] = 0

    with pytest.raises(ValidationError):
        RenderSettingsSnapshot(**data)


def test_blender_command_result_can_be_created() -> None:
    result = BlenderCommandResult(
        ok=True,
        command="collect_snapshot",
        output_files=["artifacts/blender_smoke/scene_snapshot.json"],
        stdout=None,
        stderr=None,
        error=None,
        duration_sec=0.2,
    )

    assert result.ok is True

