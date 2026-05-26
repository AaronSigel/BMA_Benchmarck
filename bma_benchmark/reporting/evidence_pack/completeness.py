from __future__ import annotations

import json
from pathlib import Path

from bma_benchmark.reporting.evidence_pack.sanity import SanitySuiteResult
from bma_benchmark.reporting.scene_examples.models import RunArtifactRef, SceneExample


def write_completeness_check(
    out_dir: Path,
    *,
    runs: list[RunArtifactRef],
    examples: list[SceneExample],
    sanity_result: SanitySuiteResult,
    figure_paths: dict[str, Path],
    table_paths: dict[str, Path],
    expected_runs: int = 100,
) -> dict:
    clean = sum(1 for e in examples if e.pass_type == "clean_pass")
    failed = sum(1 for e in examples if e.pass_type == "failed_validation")
    soft = sum(1 for e in examples if e.pass_type == "soft_pass")

    required_figures = {
        "clean_pass_examples.png": (out_dir / "figures" / "clean_pass_examples.png").is_file(),
        "failed_validation_examples.png": (out_dir / "figures" / "failed_validation_examples.png").is_file(),
        "soft_pass_export_example.png": (out_dir / "figures" / "soft_pass_export_example.png").is_file(),
        "validator_expected_actual_example.png": (
            out_dir / "figures" / "validator_expected_actual_example.png"
        ).is_file(),
    }
    for name, path in figure_paths.items():
        required_figures[name] = path.is_file()

    required_tables = {
        "selected_scene_examples.csv": (out_dir / "tables" / "selected_scene_examples.csv").is_file(),
        "validator_expected_actual_examples.csv": (
            out_dir / "tables" / "validator_expected_actual_examples.csv"
        ).is_file(),
        "validator_sanity_results.csv": (out_dir / "tables" / "validator_sanity_results.csv").is_file(),
        "demo_slice_results.csv": (out_dir / "tables" / "demo_slice_results.csv").is_file(),
        "demo_slice_summary.csv": (out_dir / "tables" / "demo_slice_summary.csv").is_file(),
    }
    for name, path in table_paths.items():
        required_tables[name] = path.is_file()

    payload = {
        "demo_runs_expected": expected_runs,
        "demo_runs_found": len(runs),
        "clean_pass_examples": clean,
        "failed_validation_examples": failed,
        "soft_pass_examples": soft,
        "validator_sanity_cases_expected": 10,
        "validator_sanity_cases_found": len(sanity_result.cases),
        "validator_sanity_all_passed_as_expected": sanity_result.all_passed_as_expected,
        "required_figures": required_figures,
        "required_tables": required_tables,
    }

    path = out_dir / "completeness_check.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if len(runs) < expected_runs:
        _write_incomplete_runs(out_dir / "INCOMPLETE_RUNS.md", runs, expected_runs)

    return payload


def _write_incomplete_runs(path: Path, runs: list, expected: int) -> None:
    lines = [
        "# Incomplete Demo Runs",
        "",
        f"Expected: {expected}",
        f"Found: {len(runs)}",
        "",
        "Possible causes:",
        "- Matrix run interrupted before completion (use `--resume`)",
        "- Preflight or runtime errors blocked some runs",
        "- Environment issues (OpenRouter, Blender MCP)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
