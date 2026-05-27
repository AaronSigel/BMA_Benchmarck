from __future__ import annotations

import argparse
from pathlib import Path

from bma_benchmark.reporting.rendering.blender_renderer import render_scene_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render report scene artifacts for a benchmark run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--blender-bin", type=str, default="blender")
    parser.add_argument("--mode", choices=["viewport", "render", "both"], default="viewport")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--timeout-sec", type=int, default=120)
    args = parser.parse_args(argv)

    result = render_scene_artifacts(
        args.run_dir,
        blender_bin=args.blender_bin,
        width=args.width,
        height=args.height,
        mode=args.mode,
        timeout_sec=args.timeout_sec,
    )
    print(f"status: {result.status}")
    if result.viewport_path:
        print(f"viewport: {result.viewport_path}")
    if result.final_render_path:
        print(f"final_render: {result.final_render_path}")
    if result.reason:
        print(f"reason: {result.reason}")
    return 0 if result.status in {"rendered", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
