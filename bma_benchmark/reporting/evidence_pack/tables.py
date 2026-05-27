from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.models import ExperimentMatrix

from bma_benchmark.reporting.evidence_pack.sanity import SanitySuiteResult, write_validator_sanity_results_csv
from bma_benchmark.reporting.scene_examples.discovery import discover_runs
from bma_benchmark.reporting.scene_examples.models import RunArtifactRef, SceneExample


def write_evidence_tables(
    experiment_dir: Path,
    out_dir: Path,
    runs: list[RunArtifactRef],
    examples: list[SceneExample],
    matrix: ExperimentMatrix | None,
    sanity_result: SanitySuiteResult,
) -> dict[str, Path]:
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["demo_slice_matrix.csv"] = _write_demo_slice_matrix(tables_dir, matrix)
    paths["demo_slice_results.csv"] = _write_demo_slice_results(tables_dir, experiment_dir, runs)
    paths["demo_slice_summary.csv"] = _write_demo_slice_summary(tables_dir, experiment_dir)
    paths["selected_scene_examples.csv"] = _write_selected_scene_examples(tables_dir, examples, out_dir)
    paths["validator_expected_actual_examples.csv"] = _write_validator_expected_actual(
        tables_dir, examples, out_dir
    )
    paths["validator_sanity_results.csv"] = write_validator_sanity_results_csv(
        tables_dir / "validator_sanity_results.csv", sanity_result
    )
    paths["missing_artifacts.csv"] = _write_missing_artifacts(tables_dir, runs, out_dir)
    return paths


def _write_demo_slice_matrix(tables_dir: Path, matrix: ExperimentMatrix | None) -> Path:
    path = tables_dir / "demo_slice_matrix.csv"
    fields = ["run_id", "task_id", "model", "strategy", "mcp_profile", "repetition"]
    rows: list[dict[str, Any]] = []
    if matrix is not None:
        config = generate_experiment_config(matrix)
        for run in config.runs:
            rows.append({
                "run_id": run.run_id,
                "task_id": run.task_id,
                "model": run.metadata.get("model_id") or run.metadata.get("model"),
                "strategy": run.metadata.get("agent_strategy"),
                "mcp_profile": run.mcp_profile,
                "repetition": run.metadata.get("repetition", 1),
            })
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_demo_slice_results(tables_dir: Path, experiment_dir: Path, runs: list[RunArtifactRef]) -> Path:
    path = tables_dir / "demo_slice_results.csv"
    summary_rows = _read_summary_csv(experiment_dir / "summary.csv")
    summary_by_id = {row.get("run_id", ""): row for row in summary_rows}
    fields = [
        "task_id",
        "category",
        "model",
        "strategy",
        "mcp_profile",
        "pass_type",
        "scene_score",
        "strict_success",
        "valid_success",
        "duration_sec",
        "tool_calls",
        "error_type",
        "top_issue_codes",
        "render_available",
        "validation_result_available",
        "snapshot_available",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            summary = summary_by_id.get(run.run_id, {})
            validation = run.validation_result or {}
            issues = validation.get("issues") or []
            codes = [
                str(i.get("code"))
                for i in issues
                if isinstance(i, dict) and i.get("code")
            ][:5]
            pass_type = summary.get("pass_type") or run.pass_type or ""
            writer.writerow({
                "task_id": run.task_id or summary.get("task_id"),
                "category": run.category or _category_from_task(run.task_id),
                "model": run.model or summary.get("model"),
                "strategy": run.strategy or summary.get("strategy"),
                "mcp_profile": run.mcp_profile or summary.get("mcp_profile"),
                "pass_type": pass_type,
                "scene_score": _first_float(summary, run, "score", "total_score", "scene_score"),
                "strict_success": pass_type == "clean_pass",
                "valid_success": pass_type in {"clean_pass", "soft_pass"},
                "duration_sec": summary.get("duration_sec"),
                "tool_calls": summary.get("tool_call_count"),
                "error_type": summary.get("error_type"),
                "top_issue_codes": ";".join(codes),
                "render_available": bool(run.render_path or run.viewport_path),
                "validation_result_available": bool(run.validation_result_path),
                "snapshot_available": bool(run.snapshot_path),
            })
    return path


def _write_demo_slice_summary(tables_dir: Path, experiment_dir: Path) -> Path:
    path = tables_dir / "demo_slice_summary.csv"
    rows = _read_summary_csv(experiment_dir / "summary.csv")
    fields = [
        "slice",
        "group",
        "n_runs",
        "clean_pass_count",
        "soft_pass_count",
        "failed_validation_count",
        "runtime_error_count",
        "valid_rate",
        "strict_rate",
        "mean_scene_score",
    ]
    output: list[dict[str, Any]] = []
    for slice_name, key_fn in (
        ("by_task", lambda r: r.get("task_id", "")),
        ("by_strategy", lambda r: r.get("strategy", "")),
        ("by_model", lambda r: r.get("model", "")),
        ("by_pass_type", lambda r: r.get("pass_type", "")),
    ):
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[key_fn(row)].append(row)
        for group, items in sorted(groups.items()):
            output.append(_aggregate_row(slice_name, group, items))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    return path


def _aggregate_row(slice_name: str, group: str, items: list[dict]) -> dict[str, Any]:
    n = len(items)
    clean = sum(1 for r in items if r.get("pass_type") == "clean_pass")
    soft = sum(1 for r in items if r.get("pass_type") == "soft_pass")
    failed = sum(1 for r in items if r.get("pass_type") == "failed_validation")
    runtime = sum(1 for r in items if r.get("pass_type") == "runtime_error")
    scores = [_parse_float(r.get("score") or r.get("total_score")) for r in items]
    scores = [s for s in scores if s is not None]
    valid = clean + soft
    return {
        "slice": slice_name,
        "group": group,
        "n_runs": n,
        "clean_pass_count": clean,
        "soft_pass_count": soft,
        "failed_validation_count": failed,
        "runtime_error_count": runtime,
        "valid_rate": f"{valid / n:.4f}" if n else "0.0000",
        "strict_rate": f"{clean / n:.4f}" if n else "0.0000",
        "mean_scene_score": f"{sum(scores) / len(scores):.4f}" if scores else "",
    }


def _write_selected_scene_examples(
    tables_dir: Path,
    examples: list[SceneExample],
    out_dir: Path,
) -> Path:
    path = tables_dir / "selected_scene_examples.csv"
    fields = [
        "example_id",
        "run_id",
        "task_id",
        "category",
        "model",
        "strategy",
        "mcp_profile",
        "pass_type",
        "scene_score",
        "strict_success",
        "render_path",
        "viewport_path",
        "blend_path",
        "glb_path",
        "validation_result_path",
        "scene_snapshot_path",
        "top_issue_codes",
        "selection_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for idx, ex in enumerate(examples, start=1):
            writer.writerow({
                "example_id": f"ex_{idx:02d}",
                "run_id": ex.run_id,
                "task_id": ex.task_id,
                "category": ex.category,
                "model": ex.model,
                "strategy": ex.strategy,
                "mcp_profile": ex.mcp_profile,
                "pass_type": ex.pass_type,
                "scene_score": ex.scene_score,
                "strict_success": ex.strict_success,
                "render_path": _rel(out_dir, ex.render_path),
                "viewport_path": _rel(out_dir, ex.viewport_path),
                "blend_path": _rel(out_dir, ex.blend_path),
                "glb_path": _rel(out_dir, ex.glb_path),
                "validation_result_path": _rel(out_dir, ex.validation_result_path),
                "scene_snapshot_path": _rel(out_dir, ex.snapshot_path),
                "top_issue_codes": ";".join(ex.top_issues),
                "selection_reason": ex.selection_reason,
            })
    return path


def _write_validator_expected_actual(
    tables_dir: Path,
    examples: list[SceneExample],
    out_dir: Path,
) -> Path:
    path = tables_dir / "validator_expected_actual_examples.csv"
    fields = [
        "example_id",
        "run_id",
        "task_id",
        "validator_name",
        "check_name",
        "object",
        "entity_ref",
        "field",
        "expected",
        "actual",
        "status",
        "tolerance",
        "passed",
        "score",
        "issue_code",
        "message",
        "severity",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for idx, ex in enumerate(examples, start=1):
            example_id = f"ex_{idx:02d}"
            validation = _read_json(ex.validation_result_path) or {}
            rows = validation.get("check_table") or ex.check_table_excerpt or []
            if not rows and ex.snapshot_path:
                rows = _posthoc_checks(ex)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                status = row.get("status")
                if status is None:
                    status = "pass" if row.get("passed") else "fail"
                writer.writerow({
                    "example_id": example_id,
                    "run_id": ex.run_id,
                    "task_id": ex.task_id,
                    "validator_name": row.get("validator_name"),
                    "check_name": row.get("check_name"),
                    "object": row.get("entity_ref"),
                    "entity_ref": row.get("entity_ref"),
                    "field": row.get("field"),
                    "expected": row.get("expected"),
                    "actual": row.get("actual"),
                    "status": status,
                    "tolerance": row.get("tolerance"),
                    "passed": row.get("passed"),
                    "score": row.get("score"),
                    "issue_code": row.get("issue_code"),
                    "message": row.get("message"),
                    "severity": row.get("severity"),
                })
    return path


def _write_missing_artifacts(tables_dir: Path, runs: list[RunArtifactRef], out_dir: Path) -> Path:
    path = tables_dir / "missing_artifacts.csv"
    fields = ["run_id", "task_id", "missing", "notes"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            missing = []
            if not (run.render_path or run.viewport_path):
                if run.blend_path:
                    missing.append("render_or_viewport")
                elif run.snapshot_path:
                    missing.append("render_or_viewport_no_blend")
                else:
                    missing.append("scene_image_and_blend")
            if not run.validation_result_path:
                missing.append("validation_result")
            if not run.snapshot_path:
                missing.append("scene_snapshot")
            if missing:
                writer.writerow({
                    "run_id": run.run_id,
                    "task_id": run.task_id,
                    "missing": ";".join(missing),
                    "notes": run.render_missing_reason or "",
                })
    return path


def _posthoc_checks(example: SceneExample) -> list[dict]:
    """Fallback: excerpt из issues с expected_value/actual_value."""
    validation = _read_json(example.validation_result_path) or {}
    rows = []
    for issue in validation.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        rows.append({
            "validator_name": issue.get("expected_path", "").split(".")[0] if issue.get("expected_path") else "",
            "check_name": issue.get("code"),
            "entity_ref": None,
            "field": issue.get("actual_path"),
            "expected": issue.get("expected_value"),
            "actual": issue.get("actual_value"),
            "passed": False,
            "issue_code": issue.get("code"),
        })
    return rows


def _read_summary_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _rel(base: Path, path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(base.resolve().parent))
    except ValueError:
        return str(path)


def _category_from_task(task_id: str | None) -> str:
    if not task_id:
        return ""
    return task_id.split("_", 1)[0]


def _first_float(summary: dict, run: RunArtifactRef, *keys: str) -> str:
    for key in keys:
        val = summary.get(key)
        if val not in (None, ""):
            return str(val)
    if run.scene_score is not None:
        return str(run.scene_score)
    return ""


def _parse_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
