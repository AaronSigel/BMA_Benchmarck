from pathlib import Path
from typing import Callable


SUPPORTED_FORMATS = {"blend", "glb", "gltf", "fbx"}


def _result(path: Path | None, export_format: str | None, error: str | None = None) -> dict:
    exists = path.exists() if path else False
    file_size_bytes = path.stat().st_size if path and exists else 0
    return {
        "ok": error is None and exists and file_size_bytes > 0,
        "output_path": str(path) if path else None,
        "format": export_format,
        "exists": exists,
        "file_size_bytes": file_size_bytes,
        "error": error,
    }


def _operator_available(operator: object) -> tuple[bool, str | None]:
    poll = getattr(operator, "poll", None)
    if callable(poll):
        try:
            return bool(poll()), None
        except Exception as exc:
            return False, str(exc)
    return callable(operator), None


def _enable_addon(bpy: object, module_name: str) -> str | None:
    try:
        import addon_utils

        addon_utils.enable(module_name, default_set=False, persistent=False)
    except Exception as exc:
        return str(exc)
    return None


def _call_export_operator(operator: Callable, **kwargs) -> None:
    operator(**kwargs)


def export_scene(payload: dict) -> dict:
    import bpy

    output_path = payload.get("output_path")
    export_format = str(payload.get("format", "")).lower()

    if not output_path:
        return _result(None, export_format or None, "output_path is required")
    if export_format not in SUPPORTED_FORMATS:
        return _result(Path(output_path), export_format or None, f"unsupported format: {export_format}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if export_format == "blend":
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        return _result(path, export_format)

    if export_format in {"glb", "gltf"}:
        addon_error = _enable_addon(bpy, "io_scene_gltf2")
        operator = getattr(bpy.ops.export_scene, "gltf", None)
        if operator is None:
            error = "Blender glTF export operator is not available"
            if addon_error:
                error = f"{error}: {addon_error}"
            return _result(path, export_format, error)

        available, poll_error = _operator_available(operator)
        if not available:
            error = "Blender glTF export operator is not available"
            if poll_error:
                error = f"{error}: {poll_error}"
            elif addon_error:
                error = f"{error}: {addon_error}"
            return _result(path, export_format, error)

        export_format_arg = "GLB" if export_format == "glb" else "GLTF_SEPARATE"
        _call_export_operator(operator, filepath=str(path), export_format=export_format_arg)
        return _result(path, export_format)

    _enable_addon(bpy, "io_scene_fbx")
    operator = getattr(bpy.ops.export_scene, "fbx", None)
    if operator is None:
        return _result(path, export_format, "Blender FBX export operator is not available")

    available, poll_error = _operator_available(operator)
    if not available:
        error = "Blender FBX export operator is not available"
        if poll_error:
            error = f"{error}: {poll_error}"
        return _result(path, export_format, error)

    _call_export_operator(operator, filepath=str(path))
    return _result(path, export_format)
