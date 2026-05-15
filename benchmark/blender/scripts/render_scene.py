from pathlib import Path
from typing import Any


def _file_result(path: Path, resolution_x: int, resolution_y: int, engine: str) -> dict:
    return {
        "ok": path.exists() and path.stat().st_size > 0,
        "output_path": str(path),
        "exists": path.exists(),
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "resolution_x": resolution_x,
        "resolution_y": resolution_y,
        "engine": engine,
        "error": None,
    }


def _error_result(
    message: str,
    output_path: str | None = None,
    resolution_x: int = 512,
    resolution_y: int = 512,
    engine: str | None = None,
) -> dict:
    path = Path(output_path) if output_path else None
    return {
        "ok": False,
        "output_path": str(path) if path else None,
        "exists": path.exists() if path else False,
        "file_size_bytes": path.stat().st_size if path and path.exists() else 0,
        "resolution_x": resolution_x,
        "resolution_y": resolution_y,
        "engine": engine,
        "error": message,
    }


def _set_engine(scene: Any, requested_engine: str) -> str:
    try:
        scene.render.engine = requested_engine
        return requested_engine
    except TypeError:
        fallback = "BLENDER_EEVEE"
        scene.render.engine = fallback
        return fallback


def _find_camera(bpy: Any, camera_name: str) -> Any | None:
    for obj in bpy.data.objects:
        if getattr(obj, "type", None) == "CAMERA" and getattr(obj, "name", None) == camera_name:
            return obj
    return None


def render_scene(payload: dict) -> dict:
    import bpy

    output_path = payload.get("output_path")
    resolution_x = int(payload.get("resolution_x", 512))
    resolution_y = int(payload.get("resolution_y", 512))
    requested_engine = str(payload.get("engine", "BLENDER_EEVEE_NEXT"))
    camera_name = payload.get("camera_name")
    transparent = bool(payload.get("transparent", False))

    if not output_path:
        return _error_result(
            "output_path is required",
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            engine=requested_engine,
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    render = scene.render
    render.resolution_x = resolution_x
    render.resolution_y = resolution_y
    engine = _set_engine(scene, requested_engine)

    if hasattr(render, "film_transparent"):
        render.film_transparent = transparent

    if camera_name:
        camera = _find_camera(bpy, str(camera_name))
        if camera is None:
            return _error_result(
                f"camera not found: {camera_name}",
                output_path=str(path),
                resolution_x=resolution_x,
                resolution_y=resolution_y,
                engine=engine,
            )
        scene.camera = camera
    elif getattr(scene, "camera", None) is None:
        return _error_result(
            "active camera is required",
            output_path=str(path),
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            engine=engine,
        )

    render.filepath = str(path)
    render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)

    return _file_result(path, resolution_x, resolution_y, engine)

