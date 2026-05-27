from __future__ import annotations

import json
from pathlib import Path

from bma_benchmark.reporting.evidence_pack.sanity import SanitySuiteResult
from bma_benchmark.reporting.scene_examples.contact_sheet import _scene_image
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
    visual_warnings: list[str] | None = None,
) -> dict:
    clean = sum(1 for e in examples if e.pass_type == "clean_pass")
    failed = sum(1 for e in examples if e.pass_type == "failed_validation")
    soft = sum(1 for e in examples if e.pass_type == "soft_pass")

    with_images = [e for e in examples if _scene_image(e) is not None]
    without_images = len(examples) - len(with_images)

    figure_names = {
        "clean_pass_examples.png",
        "failed_validation_examples.png",
        "soft_pass_export_example.png",
        "validator_expected_actual_example.png",
    }
    figures_with_real_scene_images = {
        name: (out_dir / "figures" / name).is_file() for name in sorted(figure_names)
    }
    for name, path in figure_paths.items():
        figures_with_real_scene_images[name] = path.is_file()

    required_figures = dict(figures_with_real_scene_images)

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

    clean_with_images = sum(1 for e in examples if e.pass_type == "clean_pass" and _scene_image(e))
    failed_with_images = sum(
        1 for e in examples if e.pass_type == "failed_validation" and _scene_image(e)
    )
    soft_with_images = sum(1 for e in examples if e.pass_type == "soft_pass" and _scene_image(e))
    validator_has_image = figures_with_real_scene_images.get("validator_expected_actual_example.png", False)

    visual_evidence_complete = (
        clean_with_images >= 4
        and failed_with_images >= 4
        and soft_with_images >= 1
        and validator_has_image
    )

    warnings = list(visual_warnings or [])

    payload = {
        "demo_runs_expected": expected_runs,
        "demo_runs_found": len(runs),
        "clean_pass_examples": clean,
        "failed_validation_examples": failed,
        "soft_pass_examples": soft,
        "validator_sanity_cases_expected": 10,
        "validator_sanity_cases_found": len(sanity_result.cases),
        "validator_sanity_all_passed_as_expected": sanity_result.all_passed_as_expected,
        "visual_evidence_complete": visual_evidence_complete,
        "visual_evidence_status": "complete" if visual_evidence_complete else "incomplete",
        "selected_examples_total": len(examples),
        "selected_examples_with_images": len(with_images),
        "selected_examples_without_images": without_images,
        "figures_with_real_scene_images": figures_with_real_scene_images,
        "required_figures": required_figures,
        "required_tables": required_tables,
        "warnings": warnings,
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
