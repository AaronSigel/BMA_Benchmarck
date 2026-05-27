from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class RenderedSceneArtifacts(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    run_dir: Path
    source_scene_path: Path | None = None
    viewport_path: Path | None = None
    final_render_path: Path | None = None
    status: Literal["rendered", "skipped", "failed"]
    reason: str | None = None


def resolve_source_scene(run_dir: Path, *, task_id: str | None = None) -> tuple[Path | None, str | None]:
    run_dir = Path(run_dir)
    final_blend = run_dir / "final_scene.blend"
    if final_blend.is_file() and final_blend.stat().st_size > 0:
        return final_blend, None

    manifest = _read_manifest(run_dir)
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if isinstance(files, list):
        for name in files:
            path = run_dir / str(name)
            if path.suffix.lower() == ".blend" and path.is_file():
                return path, None

    blends = sorted(run_dir.rglob("*.blend"))
    for path in blends:
        if path.is_file() and path.stat().st_size > 0:
            return path, None

    if task_id and "export" in task_id:
        glb = _find_glb(run_dir, manifest)
        if glb is not None:
            return glb, None

    glb = _find_glb(run_dir, manifest)
    if glb is not None:
        return glb, None

    return None, "no .blend or .glb source for visual rendering"


def _find_glb(run_dir: Path, manifest: dict) -> Path | None:
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    if isinstance(artifacts, dict):
        glb_entry = artifacts.get("glb_export")
        if isinstance(glb_entry, dict):
            rel = glb_entry.get("path")
            if rel:
                path = run_dir / str(rel)
                if path.is_file():
                    return path

    exports = run_dir / "exports"
    if exports.exists():
        matches = sorted(exports.rglob("*.glb"))
        if matches:
            return matches[0]

    fallback = run_dir / "result.glb"
    if fallback.is_file():
        return fallback
    return None


def _read_manifest(run_dir: Path) -> dict:
    path = run_dir / "artifact_manifest.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
