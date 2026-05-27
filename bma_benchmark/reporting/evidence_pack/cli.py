from __future__ import annotations

import argparse
from pathlib import Path

from bma_benchmark.reporting.evidence_pack.builder import build_evidence_pack


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build report evidence pack from demo slice experiment.")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/report_evidence_pack"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"))
    parser.add_argument("--render-missing-with-blender", action="store_true")
    parser.add_argument("--blender-bin", type=str, default="blender")
    parser.add_argument(
        "--render-mode",
        choices=["viewport", "render", "both"],
        default="viewport",
    )
    parser.add_argument("--render-timeout-sec", type=int, default=120)
    args = parser.parse_args(argv)

    result = build_evidence_pack(
        args.experiment,
        args.out,
        config_path=args.config,
        tasks_root=args.tasks_dir,
        render_missing_with_blender=args.render_missing_with_blender,
        blender_bin=args.blender_bin,
        render_mode=args.render_mode,
        render_timeout_sec=args.render_timeout_sec,
    )
    print(f"evidence_pack: {args.out}")
    print(f"demo_runs_found: {result.get('demo_runs_found')}")
    print(f"visual_evidence_complete: {result.get('visual_evidence_complete')}")
    print(f"completeness: {args.out / 'completeness_check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
