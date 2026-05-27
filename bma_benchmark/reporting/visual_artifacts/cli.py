from __future__ import annotations

import argparse
from pathlib import Path

from bma_benchmark.reporting.visual_artifacts.check import write_visual_artifacts_check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check visual artifacts for experiment runs.")
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args(argv)

    csv_path, json_path = write_visual_artifacts_check(args.experiment)
    print(f"visual_artifacts_check_csv: {csv_path}")
    print(f"visual_artifacts_check_json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
