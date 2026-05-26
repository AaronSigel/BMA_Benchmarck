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
    args = parser.parse_args(argv)

    result = build_evidence_pack(
        args.experiment,
        args.out,
        config_path=args.config,
        tasks_root=args.tasks_dir,
    )
    print(f"evidence_pack: {args.out}")
    print(f"demo_runs_found: {result.get('demo_runs_found')}")
    print(f"completeness: {args.out / 'completeness_check.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
