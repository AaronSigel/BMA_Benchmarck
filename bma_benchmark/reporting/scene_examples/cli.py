from __future__ import annotations

import argparse
from pathlib import Path

from bma_benchmark.reporting.scene_examples.discovery import discover_runs
from bma_benchmark.reporting.scene_examples.models import SceneExampleSelectionConfig
from bma_benchmark.reporting.scene_examples.selection import select_scene_examples
from bma_benchmark.reporting.scene_examples.writers import write_scene_examples


def build_scene_gallery(input_dir: Path, out_dir: Path, examples_per_status: int = 4) -> int:
    runs = discover_runs(input_dir)
    bundle = select_scene_examples(
        runs,
        SceneExampleSelectionConfig(examples_per_status=examples_per_status),
    )
    write_scene_examples(bundle, out_dir)
    print(f"scene_examples: {out_dir}")
    print(f"examples: {len(bundle.examples)}")
    for warning in bundle.warnings:
        print(f"WARNING: {warning}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build representative scene example metadata and images.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--examples-per-status", type=int, default=4)
    args = parser.parse_args(argv)
    return build_scene_gallery(args.input, args.out, args.examples_per_status)


if __name__ == "__main__":
    raise SystemExit(main())
