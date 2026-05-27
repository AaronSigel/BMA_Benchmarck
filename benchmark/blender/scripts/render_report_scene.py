from __future__ import annotations

from pathlib import Path
from typing import Any


def _error(message: str, **extra: Any) -> dict:
    return {"ok": False, "error": message, **extra}


def _mesh_bounds(bpy: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    import mathutils

    min_co = [float("inf")] * 3
    max_co = [float("-inf")] * 3
    found = False
    for obj in bpy.data.objects:
        if getattr(obj, "type", None) != "MESH":
            continue
        found = True
        for corner in obj.bound_box:
            world = obj.matrix_world @ mathutils.Vector(corner)
            for idx in range(3):
                min_co[idx] = min(min_co[idx], world[idx])
                max_co[idx] = max(max_co[idx], world[idx])
    if not found:
        return None
    return tuple(min_co), tuple(max_co)


def _ensure_camera(bpy: Any, scene: Any) -> Any:
    camera = scene.camera
    if camera is not None and getattr(camera, "type", None) == "CAMERA":
        return camera

    bounds = _mesh_bounds(bpy)
    if bounds is None:
        center = (0.0, 0.0, 0.0)
        distance = 5.0
        clip_end = 1000.0
    else:
        (min_co, max_co) = bounds
        center = tuple((min_co[i] + max_co[i]) / 2 for i in range(3))
        span = max(max_co[i] - min_co[i] for i in range(3))
        distance = max(span * 1.8, 2.0)
        clip_end = max(distance * 10, 100.0)

    cam_data = bpy.data.cameras.new(name="ReportFallbackCamera")
    cam_obj = bpy.data.objects.new("ReportFallbackCamera", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = (
        center[0] + distance,
        center[1] - distance,
        center[2] + distance * 0.7,
    )
    direction = tuple(center[i] - cam_obj.location[i] for i in range(3))
    cam_obj.rotation_euler = _look_at_rotation(direction)
    cam_data.lens = 35.0
    cam_data.clip_start = 0.01
    cam_data.clip_end = clip_end
    scene.camera = cam_obj
    return cam_obj


def _look_at_rotation(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    import mathutils

    vec = mathutils.Vector(direction)
    if vec.length == 0:
        return (0.0, 0.0, 0.0)
    rot_quat = vec.to_track_quat("-Z", "Y")
    return rot_quat.to_euler()


def _ensure_light(bpy: Any, scene: Any) -> None:
    for obj in bpy.data.objects:
        if getattr(obj, "type", None) == "LIGHT":
            return

    bounds = _mesh_bounds(bpy)
    if bounds is None:
        center = (0.0, 0.0, 2.0)
        z_offset = 2.0
    else:
        min_co, max_co = bounds
        center = tuple((min_co[i] + max_co[i]) / 2 for i in range(3))
        z_offset = max(2.0, max_co[2] - min_co[2])

    light_data = bpy.data.lights.new(name="ReportFallbackLight", type="AREA")
    light_data.energy = 500.0
    light_obj = bpy.data.objects.new("ReportFallbackLight", light_data)
    scene.collection.objects.link(light_obj)
    light_obj.location = (center[0], center[1] - 2.0, center[2] + z_offset)


def _set_engine(bpy: Any, scene: Any, requested: str) -> str:
    available = {
        item.identifier
        for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    }
    if requested == "viewport":
        candidates = ["BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"]
    else:
        candidates = ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES", "BLENDER_WORKBENCH"]
    for candidate in candidates:
        if candidate in available:
            scene.render.engine = candidate
            return candidate
    raise RuntimeError(f"No supported render engine found: {sorted(available)}")


def _configure_output(scene: Any, path: Path, width: int, height: int) -> None:
    render = scene.render
    render.resolution_x = width
    render.resolution_y = height
    render.resolution_percentage = 100
    render.filepath = str(path)
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.image_settings.color_depth = "8"


def _load_source(bpy: Any, source_path: Path) -> str | None:
    suffix = source_path.suffix.lower()
    if suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(source_path))
        return None
    if suffix == ".glb":
        bpy.ops.wm.read_homefile(use_empty=True)
        try:
            bpy.ops.import_scene.gltf(filepath=str(source_path))
        except Exception as exc:
            return f"GLB import failed: {exc}"
        return None
    return f"unsupported source format: {suffix}"


def _render_to_file(bpy: Any, scene: Any, path: Path, *, engine_mode: str) -> str | None:
    _configure_output(scene, path, scene.render.resolution_x, scene.render.resolution_y)
    try:
        engine = _set_engine(bpy, scene, engine_mode)
    except Exception as exc:
        return str(exc)

    if engine == "BLENDER_WORKBENCH":
        try:
            shading = scene.display.shading
            shading.color_type = "MATERIAL"
            shading.light = "STUDIO"
        except Exception:
            pass
    elif engine in {"BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"}:
        try:
            scene.eevee.taa_render_samples = 16 if engine_mode == "viewport" else 64
        except Exception:
            pass

    _ensure_camera(bpy, scene)
    _ensure_light(bpy, scene)

    try:
        bpy.ops.render.render(write_still=True)
    except Exception as exc:
        return f"render failed: {exc}"

    if not path.is_file() or path.stat().st_size == 0:
        return "render produced no output file"
    return None


def render_report_scene(payload: dict) -> dict:
    import bpy

    source_path = payload.get("source_path")
    output_dir = payload.get("output_dir")
    mode = str(payload.get("mode", "viewport")).lower()
    width = int(payload.get("width", 1280))
    height = int(payload.get("height", 720))

    if not source_path:
        return _error("source_path is required")
    if not output_dir:
        return _error("output_dir is required")

    source = Path(str(source_path))
    out_dir = Path(str(output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source.is_file():
        return _error(f"source file not found: {source}")

    load_error = _load_source(bpy, source)
    if load_error:
        return _error(load_error, source=str(source))

    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height

    viewport_path = out_dir / "viewport.png"
    final_render_path = out_dir / "final_render.png"
    outputs: dict[str, str | None] = {"viewport_path": None, "final_render_path": None}
    errors: list[str] = []

    if mode in {"viewport", "both"}:
        err = _render_to_file(bpy, scene, viewport_path, engine_mode="viewport")
        if err:
            errors.append(f"viewport: {err}")
        elif viewport_path.is_file():
            outputs["viewport_path"] = str(viewport_path)

    if mode in {"render", "both"}:
        err = _render_to_file(bpy, scene, final_render_path, engine_mode="render")
        if err:
            errors.append(f"render: {err}")
        elif final_render_path.is_file():
            outputs["final_render_path"] = str(final_render_path)

    if not outputs["viewport_path"] and not outputs["final_render_path"]:
        return _error("; ".join(errors) or "no render output produced", source=str(source))

    return {
        "ok": True,
        "source": str(source),
        "mode": mode,
        "viewport_path": outputs["viewport_path"],
        "final_render_path": outputs["final_render_path"],
        "errors": errors or None,
    }
