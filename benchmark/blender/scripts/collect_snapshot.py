import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _vector3(value: Any) -> dict[str, float]:
    return {
        "x": float(value[0]),
        "y": float(value[1]),
        "z": float(value[2]),
    }


def _color_rgba(value: Any) -> dict[str, float] | None:
    if value is None:
        return None

    values = list(value)
    if len(values) < 3:
        return None
    if len(values) == 3:
        values.append(1.0)

    return {
        "r": float(values[0]),
        "g": float(values[1]),
        "b": float(values[2]),
        "a": float(values[3]),
    }


def _node_input_value(material: Any, input_name: str) -> Any:
    node_tree = getattr(material, "node_tree", None)
    nodes = getattr(node_tree, "nodes", []) if node_tree is not None else []
    for node in nodes:
        if getattr(node, "type", None) != "BSDF_PRINCIPLED":
            continue
        inputs = getattr(node, "inputs", {})
        socket = inputs.get(input_name) if hasattr(inputs, "get") else None
        if socket is not None:
            return getattr(socket, "default_value", None)
    return None


def _material_base_color(material: Any) -> dict[str, float] | None:
    node_color = _node_input_value(material, "Base Color")
    if node_color is not None:
        return _color_rgba(node_color)
    return _color_rgba(getattr(material, "diffuse_color", None))


def _material_float(material: Any, input_name: str, fallback_attr: str) -> float | None:
    node_value = _node_input_value(material, input_name)
    if node_value is not None:
        return float(node_value)

    fallback = getattr(material, fallback_attr, None)
    return float(fallback) if fallback is not None else None


def _material_names(obj: Any) -> list[str]:
    names = []
    for slot in getattr(obj, "material_slots", []) or []:
        material = getattr(slot, "material", None)
        if material is not None:
            names.append(material.name)

    if names:
        return names

    data = getattr(obj, "data", None)
    for material in getattr(data, "materials", []) or []:
        if material is not None:
            names.append(material.name)
    return names


def _primitive_hint(obj: Any) -> str | None:
    custom_hint = _custom_property(obj, "bma_primitive_hint")
    if custom_hint:
        return str(custom_hint).lower()

    text = f"{getattr(obj, 'name', '')} {getattr(getattr(obj, 'data', None), 'name', '')}".lower()
    for primitive in ("cube", "sphere", "cylinder", "plane", "cone", "torus"):
        if primitive in text:
            return primitive
    return None


def _custom_property(obj: Any, key: str) -> Any:
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            return None
    try:
        return obj[key]
    except Exception:
        return None


def _collection_names(obj: Any) -> list[str]:
    return [collection.name for collection in getattr(obj, "users_collection", [])]


def _mesh_count(obj: Any, attr: str) -> int | None:
    if getattr(obj, "type", None) != "MESH":
        return None

    data = getattr(obj, "data", None)
    values = getattr(data, attr, None)
    return len(values) if values is not None else None


def _object_snapshot(obj: Any) -> dict[str, Any]:
    return {
        "name": obj.name,
        "type": obj.type,
        "primitive_hint": _primitive_hint(obj),
        "location": _vector3(obj.location),
        "rotation_euler": _vector3(obj.rotation_euler),
        "scale": _vector3(obj.scale),
        "dimensions": _vector3(obj.dimensions),
        "material_slots": _material_names(obj),
        "parent": getattr(getattr(obj, "parent", None), "name", None),
        "collection_names": _collection_names(obj),
        "vertex_count": _mesh_count(obj, "vertices"),
        "polygon_count": _mesh_count(obj, "polygons"),
    }


def _material_snapshot(material: Any) -> dict[str, Any]:
    return {
        "name": material.name,
        "base_color": _material_base_color(material),
        "roughness": _material_float(material, "Roughness", "roughness"),
        "metallic": _material_float(material, "Metallic", "metallic"),
        "use_nodes": bool(getattr(material, "use_nodes", False)),
    }


def _light_snapshot(obj: Any) -> dict[str, Any]:
    data = getattr(obj, "data", None)
    return {
        "name": obj.name,
        "type": getattr(data, "type", obj.type),
        "location": _vector3(obj.location),
        "rotation_euler": _vector3(obj.rotation_euler),
        "energy": float(data.energy) if getattr(data, "energy", None) is not None else None,
        "color": _color_rgba(getattr(data, "color", None)),
    }


def _camera_snapshot(scene: Any, obj: Any) -> dict[str, Any]:
    data = getattr(obj, "data", None)
    return {
        "name": obj.name,
        "location": _vector3(obj.location),
        "rotation_euler": _vector3(obj.rotation_euler),
        "lens": float(data.lens) if getattr(data, "lens", None) is not None else None,
        "sensor_width": (
            float(data.sensor_width) if getattr(data, "sensor_width", None) is not None else None
        ),
        "is_active": getattr(scene, "camera", None) is obj,
    }


def _render_settings(scene: Any) -> dict[str, Any]:
    render = scene.render
    return {
        "engine": str(scene.render.engine),
        "resolution_x": int(render.resolution_x),
        "resolution_y": int(render.resolution_y),
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "frame_current": int(scene.frame_current),
    }


def _write_snapshot(payload: dict, snapshot: dict[str, Any]) -> None:
    output_path = payload.get("output_path")
    output_dir = payload.get("output_dir")
    if not output_path and output_dir:
        output_path = str(Path(output_dir) / "scene_snapshot.json")
    if not output_path:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")


def collect_snapshot(payload: dict) -> dict:
    import bpy

    scene = bpy.context.scene
    objects = list(getattr(scene, "objects", bpy.data.objects))

    snapshot = {
        "scene_name": scene.name,
        "objects": [_object_snapshot(obj) for obj in objects],
        "materials": [_material_snapshot(material) for material in bpy.data.materials],
        "lights": [_light_snapshot(obj) for obj in objects if getattr(obj, "type", None) == "LIGHT"],
        "cameras": [_camera_snapshot(scene, obj) for obj in objects if getattr(obj, "type", None) == "CAMERA"],
        "collections": [collection.name for collection in bpy.data.collections],
        "render_settings": _render_settings(scene),
        "frame_current": int(scene.frame_current),
        "blender_version": ".".join(str(part) for part in bpy.app.version),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    _write_snapshot(payload, snapshot)
    return snapshot
