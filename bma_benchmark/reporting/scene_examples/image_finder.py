from __future__ import annotations

from pathlib import Path

IMAGE_PATTERNS = [
    "render.png",
    "viewport.png",
    "screenshot.png",
    "final_render.png",
    "artifacts/render.png",
    "artifacts/viewport.png",
    "renders/*.png",
    "figures/*.png",
]


def find_scene_image(run_dir: Path) -> tuple[Path | None, str | None]:
    for pattern in IMAGE_PATTERNS:
        for path in sorted(run_dir.glob(pattern)):
            if path.is_file() and path.stat().st_size > 0:
                return path, None
    return None, "no render/viewport image found"
