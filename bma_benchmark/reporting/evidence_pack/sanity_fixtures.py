from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from benchmark.blender.models import (
    CameraSnapshot,
    ColorRGBA,
    LightSnapshot,
    MaterialSnapshot,
    ObjectSnapshot,
    RenderSettingsSnapshot,
    SceneSnapshot,
    Vector3,
)
from benchmark.tasks.loader import load_task
from benchmark.tasks.models import BenchmarkTask, ExpectedObject
from benchmark.validation.validators.camera_validator import _angular_deviation_deg, _camera_forward
from benchmark.validation.validators.light_validator import (
    _angle_between_deg,
    _direction_to_target,
    _euler_to_direction,
)


@dataclass(frozen=True)
class SanityCaseSpec:
    case_id: str
    task_id: str
    validator: str
    positive_or_negative: str
    expected_outcome: str
    mutation: str | None = None
    copy_export_glb: bool = False
    notes: str = ""


SANITY_CASES: tuple[SanityCaseSpec, ...] = (
    SanityCaseSpec("geometry_positive", "geometry_002_positions", "Transform/Object", "positive", "pass"),
    SanityCaseSpec(
        "geometry_negative_position",
        "geometry_002_positions",
        "Transform",
        "negative",
        "fail/partial",
        mutation="shift_cube_right",
    ),
    SanityCaseSpec(
        "material_positive",
        "materials_004_multiple_objects",
        "Material",
        "positive",
        "pass",
    ),
    SanityCaseSpec(
        "material_negative_swap",
        "materials_004_multiple_objects",
        "Material",
        "negative",
        "fail/partial",
        mutation="swap_materials",
    ),
    SanityCaseSpec(
        "lighting_positive",
        "lighting_003_three_point_lighting",
        "Light",
        "positive",
        "pass",
    ),
    SanityCaseSpec(
        "lighting_negative_missing_light",
        "lighting_003_three_point_lighting",
        "Light",
        "negative",
        "fail",
        mutation="remove_fill_light",
    ),
    SanityCaseSpec(
        "camera_positive",
        "camera_003_composition_view",
        "Camera",
        "positive",
        "pass",
    ),
    SanityCaseSpec(
        "camera_negative_wrong_direction",
        "camera_003_composition_view",
        "Camera",
        "negative",
        "fail/partial",
        mutation="camera_wrong_rotation",
    ),
    SanityCaseSpec(
        "export_positive_glb",
        "export_002_glb_file",
        "Export",
        "positive",
        "export_pass",
        copy_export_glb=True,
    ),
    SanityCaseSpec(
        "export_negative_missing_file",
        "export_002_glb_file",
        "Export",
        "negative",
        "fail",
    ),
)


def find_task_path(task_id: str, tasks_root: Path) -> Path | None:
    for path in tasks_root.rglob("*.yaml"):
        if path.stem == task_id:
            return path
    for path in tasks_root.rglob("*.yml"):
        if path.stem == task_id:
            return path
    return None


def build_sanity_snapshot(spec: SanityCaseSpec) -> SceneSnapshot:
    task_path = find_task_path(spec.task_id, Path("tasks"))
    if task_path is None:
        raise FileNotFoundError(f"task not found: {spec.task_id}")
    task = load_task(task_path)
    snapshot = _snapshot_from_task(task)
    if spec.mutation == "shift_cube_right":
        for obj in snapshot.objects:
            if obj.name == "Cube_Right":
                obj.location = Vector3(x=0.0, y=0.0, z=0.0)
    elif spec.mutation == "swap_materials":
        for obj in snapshot.objects:
            if obj.name == "Yellow_Cube":
                obj.material_slots = ["Cyan"]
            elif obj.name == "Cyan_Sphere":
                obj.material_slots = ["Yellow"]
    elif spec.mutation == "remove_fill_light":
        snapshot.lights = [light for light in snapshot.lights if light.name != "Fill_Light"]
    elif spec.mutation == "camera_wrong_rotation":
        for cam in snapshot.cameras:
            if cam.name == "Composition_Camera":
                cam.rotation_euler = Vector3(x=10.0, y=0.0, z=10.0)
    return snapshot


def _snapshot_from_task(task: BenchmarkTask) -> SceneSnapshot:
    expected = task.expected_scene
    objects: list[ObjectSnapshot] = []
    for exp in expected.objects:
        objects.append(_object_from_expected(exp))
    materials = [
        MaterialSnapshot(
            name=mat.name,
            base_color=ColorRGBA(
                r=mat.base_color.r,
                g=mat.base_color.g,
                b=mat.base_color.b,
                a=mat.base_color.a,
            )
            if mat.base_color
            else None,
            roughness=mat.roughness,
            metallic=mat.metallic,
            use_nodes=True,
        )
        for mat in expected.materials
    ]
    cameras = [
        CameraSnapshot(
            name=cam.name or "Camera",
            location=_vec(cam.location),
            rotation_euler=_fit_camera_rotation(_vec(cam.location), _resolve_target_point(cam.target, objects)),
            lens=cam.focal_length,
            sensor_width=36.0,
            is_active=True,
        )
        for cam in expected.cameras
    ]
    lights = [
        LightSnapshot(
            name=light.name or f"Light_{idx}",
            type=light.type,
            location=loc,
            rotation_euler=_fit_light_rotation(loc, _resolve_target_point(light.target, objects)),
            energy=light.energy,
            color=None,
        )
        for idx, light in enumerate(expected.lights)
        for loc in [_vec(light.location)]
    ]
    return SceneSnapshot(
        scene_name=f"Sanity_{task.id}",
        objects=objects,
        materials=materials,
        lights=lights,
        cameras=cameras,
        collections=["Collection"],
        render_settings=RenderSettingsSnapshot(
            engine="BLENDER_EEVEE",
            resolution_x=1920,
            resolution_y=1080,
            frame_start=1,
            frame_end=1,
            frame_current=1,
        ),
        frame_current=1,
        blender_version="4.2.0",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _object_from_expected(exp: ExpectedObject) -> ObjectSnapshot:
    dims = exp.dimensions
    return ObjectSnapshot(
        name=exp.name or "Object",
        type=exp.type,
        primitive_hint=exp.primitive,
        location=_vec(exp.location),
        rotation_euler=_vec(exp.rotation),
        scale=_vec(exp.scale) if exp.scale else Vector3(x=1.0, y=1.0, z=1.0),
        dimensions=_vec(dims) if dims else Vector3(x=1.0, y=1.0, z=1.0),
        material_slots=[exp.material] if exp.material else [],
        parent=None,
        collection_names=["Collection"],
        vertex_count=8,
        polygon_count=6,
    )


def _vec(value) -> Vector3:
    if value is None:
        return Vector3(x=0.0, y=0.0, z=0.0)
    return Vector3(x=float(value.x), y=float(value.y), z=float(value.z))


def _resolve_target_point(target, objects: list[ObjectSnapshot]) -> Vector3:
    if target is None:
        return Vector3(x=0.0, y=0.0, z=0.0)
    if isinstance(target, str):
        for obj in objects:
            if obj.name == target:
                return obj.location
        return Vector3(x=0.0, y=0.0, z=0.0)
    return _vec(target)


def _fit_light_rotation(location: Vector3, target: Vector3) -> Vector3:
    desired = _direction_to_target(location, target)
    return _search_euler(desired, _euler_to_direction)


def _fit_camera_rotation(location: Vector3, target: Vector3) -> Vector3:
    desired = _normalize_tuple((
        target.x - location.x,
        target.y - location.y,
        target.z - location.z,
    ))
    return _search_euler(desired, _camera_forward)


def _search_euler(
    desired: tuple[float, float, float],
    forward_fn,
) -> Vector3:
    best = Vector3(x=0.0, y=0.0, z=0.0)
    best_angle = 999.0
    for rx in _frange(-math.pi, math.pi, math.pi / 18):
        for ry in _frange(-math.pi, math.pi, math.pi / 18):
            for rz in _frange(-math.pi, math.pi, math.pi / 18):
                euler = Vector3(x=rx, y=ry, z=rz)
                angle = _angle_between_deg(forward_fn(euler), desired)
                if angle < best_angle:
                    best_angle = angle
                    best = euler
    return best


def _frange(start: float, stop: float, step: float):
    value = start
    while value <= stop:
        yield value
        value += step


def _normalize_tuple(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(part * part for part in v))
    if length <= 0.0:
        return (0.0, 0.0, -1.0)
    return tuple(part / length for part in v)
