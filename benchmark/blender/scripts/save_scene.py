from pathlib import Path


def save_scene(payload: dict) -> dict:
    import bpy

    raw_path = payload.get("path")
    if not raw_path:
        return {
            "ok": False,
            "path": None,
            "exists": False,
            "file_size_bytes": 0,
            "error": "path is required",
        }

    path = Path(raw_path)
    if path.suffix.lower() != ".blend":
        return {
            "ok": False,
            "path": str(path),
            "exists": path.exists(),
            "file_size_bytes": path.stat().st_size if path.exists() else 0,
            "error": "path must end with .blend",
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))

    return {
        "ok": True,
        "path": str(path),
        "exists": path.exists(),
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
        "error": None,
    }

