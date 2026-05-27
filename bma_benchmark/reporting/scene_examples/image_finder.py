from __future__ import annotations

import json
from pathlib import Path

IMAGE_PATTERNS = [
    "final_render.png",
    "viewport.png",
    "render.png",
    "screenshot.png",
    "artifacts/final_render.png",
    "artifacts/viewport.png",
    "artifacts/render.png",
    "renders/*.png",
    "figures/*.png",
]

EXCLUDED_SUFFIXES = (
    "_card.png",
    "_examples.png",
)


def _is_excluded_png(name: str) -> bool:
    lowered = name.lower()
    if any(lowered.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return True
    if lowered in {
        "clean_pass_examples.png",
        "failed_validation_examples.png",
        "soft_pass_examples.png",
        "soft_pass_export_example.png",
        "validator_expected_actual_example.png",
        "mixed_scene_examples.png",
    }:
        return True
    return False


def _valid_png(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _manifest_image_paths(run_dir: Path) -> list[Path]:
    manifest_path = run_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    if not isinstance(artifacts, dict):
        return []

    keys = ("final_render", "viewport", "render")
    found: list[Path] = []
    for key in keys:
        entry = artifacts.get(key)
        if not isinstance(entry, dict):
            continue
        rel = entry.get("path")
        if not rel:
            continue
        path = run_dir / str(rel)
        if _valid_png(path):
            found.append(path)
    return found


def find_scene_images(run_dir: Path) -> tuple[Path | None, Path | None, str | None]:
    """Return (render_path, viewport_path, missing_reason)."""
    render_path: Path | None = None
    viewport_path: Path | None = None

    manifest_paths = _manifest_image_paths(run_dir)
    for path in manifest_paths:
        name = path.name.lower()
        if "viewport" in name:
            viewport_path = viewport_path or path
        else:
            render_path = render_path or path

    for pattern in IMAGE_PATTERNS:
        for path in sorted(run_dir.glob(pattern)):
            if not _valid_png(path) or _is_excluded_png(path.name):
                continue
            name = path.name.lower()
            if "viewport" in name:
                viewport_path = viewport_path or path
            elif render_path is None:
                render_path = path

    if render_path is None and viewport_path is None:
        for path in sorted(run_dir.rglob("*.png")):
            if not _valid_png(path) or _is_excluded_png(path.name):
                continue
            name = path.name.lower()
            if "viewport" in name:
                viewport_path = viewport_path or path
            elif render_path is None:
                render_path = path

    if render_path or viewport_path:
        return render_path, viewport_path, None
    return None, None, "no render/viewport image found"


def find_scene_image(run_dir: Path) -> tuple[Path | None, str | None]:
    render_path, viewport_path, reason = find_scene_images(run_dir)
    if viewport_path is not None:
        return viewport_path, None
    if render_path is not None:
        return render_path, None
    return None, reason
