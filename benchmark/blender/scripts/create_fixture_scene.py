from pathlib import Path
from typing import Any

from benchmark.blender.scripts.reset_scene import reset_scene


FLOOR_Z = 0.0
CLEARANCE = 0.03


def _find_principled_bsdf(material: Any) -> Any | None:
    nodes = material.node_tree.nodes

    bsdf = nodes.get("Principled BSDF")
    if bsdf is not None:
        return bsdf

    for node in nodes:
        if getattr(node, "type", None) == "BSDF_PRINCIPLED":
            return node

    return None


def _create_material(
    bpy: Any,
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.5,
    metallic: float = 0.0,
) -> Any:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = color

    bsdf = _find_principled_bsdf(material)

    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = color

        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness

        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic

        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = color[3]

    return material


def _assign_material(obj: Any, material: Any) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)


def _active_object(bpy: Any) -> Any:
    return bpy.context.object


def _place_on_floor(
    bpy: Any,
    obj: Any,
    floor_z: float = FLOOR_Z,
    clearance: float = CLEARANCE,
) -> None:
    from mathutils import Vector

    bpy.context.view_layer.update()

    world_corners = [
        obj.matrix_world @ Vector(corner)
        for corner in obj.bound_box
    ]

    min_z = min(corner.z for corner in world_corners)
    target_min_z = floor_z + clearance

    obj.location.z += target_min_z - min_z

    bpy.context.view_layer.update()


def _look_at(obj: Any, target: tuple[float, float, float]) -> None:
    from mathutils import Vector

    target_vector = Vector(target)
    direction = target_vector - obj.location

    if direction.length == 0:
        return

    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _shade_smooth(obj: Any) -> None:
    mesh = getattr(obj, "data", None)
    polygons = getattr(mesh, "polygons", None)

    if polygons is None:
        return

    for polygon in polygons:
        polygon.use_smooth = True


def create_fixture_scene(payload: dict) -> dict:
    import bpy

    scene_name = str(payload.get("scene_name") or "BMA Fixture Scene")
    add_camera = bool(payload.get("add_camera", True))
    add_light = bool(payload.get("add_light", True))
    save_path = payload.get("save_path")

    reset_scene({"scene_name": scene_name})

    red = _create_material(
        bpy,
        "RedMaterial",
        (1.0, 0.05, 0.02, 1.0),
        roughness=0.45,
        metallic=0.0,
    )

    blue = _create_material(
        bpy,
        "BlueMaterial",
        (0.05, 0.2, 1.0, 1.0),
        roughness=0.35,
        metallic=0.0,
    )

    gray = _create_material(
        bpy,
        "GrayMaterial",
        (0.55, 0.55, 0.55, 1.0),
        roughness=0.75,
        metallic=0.0,
    )

    # Floor
    bpy.ops.mesh.primitive_plane_add(
        size=8.0,
        location=(0.0, 0.0, FLOOR_Z),
    )
    plane = _active_object(bpy)
    plane.name = "FixtureFloor"
    _assign_material(plane, gray)

    # Cube
    bpy.ops.mesh.primitive_cube_add(
        size=2.0,
        location=(0.0, 0.0, 0.0),
    )
    cube = _active_object(bpy)
    cube.name = "FixtureCube"
    _place_on_floor(bpy, cube)
    _assign_material(cube, red)

    # Sphere
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=32,
        ring_count=16,
        radius=1.0,
        location=(3.0, 0.0, 0.0),
    )
    sphere = _active_object(bpy)
    sphere.name = "FixtureSphere"
    _place_on_floor(bpy, sphere)
    _shade_smooth(sphere)
    _assign_material(sphere, blue)

    # Cylinder
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=32,
        radius=1.0,
        depth=2.0,
        location=(-3.0, 0.0, 0.0),
    )
    cylinder = _active_object(bpy)
    cylinder.name = "FixtureCylinder"
    _place_on_floor(bpy, cylinder)
    _shade_smooth(cylinder)
    _assign_material(cylinder, red)

    light_name = None
    if add_light:
        bpy.ops.object.light_add(
            type="AREA",
            location=(0.0, -4.0, 6.0),
        )
        light = _active_object(bpy)
        light.name = "FixtureAreaLight"
        light.data.energy = 500.0
        light.data.size = 5.0
        light_name = light.name

    camera_name = None
    if add_camera:
        bpy.ops.object.camera_add(
            location=(11.72, -12.4, 7.93),
        )
        camera = _active_object(bpy)
        camera.name = "FixtureCamera"

        _look_at(camera, (0.0, 0.0, 1.0))

        camera.data.lens = 35.0
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
        "objects": [
            cube.name,
            sphere.name,
            cylinder.name,
            plane.name,
        ],
        "materials": [
            red.name,
            blue.name,
            gray.name,
        ],
        "light": light_name,
        "camera": camera_name,
        "floor_z": FLOOR_Z,
        "clearance": CLEARANCE,
        "save_path": str(save_path) if save_path else None,
        "saved": saved,
    }