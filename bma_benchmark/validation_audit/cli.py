from __future__ import annotations

import argparse
from pathlib import Path

from bma_benchmark.validation_audit.collector import collect_validator_audit
from bma_benchmark.validation_audit.writers import write_validator_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build validator audit artifacts.")
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = collect_validator_audit(args.tasks_dir)
    write_validator_audit(report, args.out)
    print(f"validator_audit: {args.out}")
    print(f"rows: {len(report.rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
