import math
from pathlib import Path
from typing import Any

from benchmark.blender.scripts.reset_scene import reset_scene


def _create_material(bpy: Any, name: str, color: tuple[float, float, float, float]) -> Any:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    return material


def _assign_material(obj: Any, material: Any) -> None:
    obj.data.materials.append(material)


def _active_object(bpy: Any) -> Any:
    return bpy.context.object


def _look_at(obj: Any, target: tuple[float, float, float]) -> None:
    direction = (
        target[0] - obj.location[0],
        target[1] - obj.location[1],
        target[2] - obj.location[2],
    )

    if hasattr(direction, "to_track_quat"):
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        return

    dx, dy, dz = direction
    horizontal = math.sqrt(dx * dx + dy * dy)
    obj.rotation_euler = (
        math.atan2(horizontal, dz),
        0.0,
        math.atan2(dx, dy),
    )


def create_fixture_scene(payload: dict) -> dict:
    import bpy

    scene_name = str(payload.get("scene_name") or "BMA Fixture Scene")
    add_camera = bool(payload.get("add_camera", True))
    add_light = bool(payload.get("add_light", True))
    save_path = payload.get("save_path")

    reset_scene({"scene_name": scene_name})

    red = _create_material(bpy, "RedMaterial", (1.0, 0.05, 0.02, 1.0))
    blue = _create_material(bpy, "BlueMaterial", (0.05, 0.2, 1.0, 1.0))
    gray = _create_material(bpy, "GrayMaterial", (0.55, 0.55, 0.55, 1.0))

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 1.0))
    cube = _active_object(bpy)
    cube.name = "FixtureCube"
    _assign_material(cube, red)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, location=(3.0, 0.0, 1.0))
    sphere = _active_object(bpy)
    sphere.name = "FixtureSphere"
    _assign_material(sphere, blue)

    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=1.0, depth=2.0, location=(-3.0, 0.0, 1.0))
    cylinder = _active_object(bpy)
    cylinder.name = "FixtureCylinder"
    _assign_material(cylinder, red)

    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 0.0))
    plane = _active_object(bpy)
    plane.name = "FixtureFloor"
    _assign_material(plane, gray)

    light_name = None
    if add_light:
        bpy.ops.object.light_add(type="AREA", location=(0.0, -4.0, 6.0))
        light = _active_object(bpy)
        light.name = "FixtureAreaLight"
        light.data.energy = 500.0
        light.data.size = 5.0
        light_name = light.name

    camera_name = None
    if add_camera:
        bpy.ops.object.camera_add(location=(6.0, -6.0, 4.0))
        camera = _active_object(bpy)
        camera.name = "FixtureCamera"
        _look_at(camera, (0.0, 0.0, 1.0))
        bpy.context.scene.camera = camera
        camera_name = camera.name

    saved = False
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        saved = path.exists()

    return {
        "scene_name": bpy.context.scene.name,
        "objects": [cube.name, sphere.name, cylinder.name, plane.name],
        "materials": [red.name, blue.name, gray.name],
        "light": light_name,
        "camera": camera_name,
        "save_path": str(save_path) if save_path else None,
        "saved": saved,
    }

