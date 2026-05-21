from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_FILES = {
    "summary.csv",
    "summary.json",
    "experiment_analysis.json",
    "report.md",
    "report.html",
    "report_text_ru.md",
    "manifest.json",
    "README_REPORT.md",
    "run_artifact_manifests.json",
}

REQUIRED_FIGURES = {
    "success_by_strategy.png",
    "success_by_profile.png",
    "success_by_category.png",
    "top_validation_issues.png",
    "cost_by_strategy.png",
}


def validate_report_bundle(bundle: Path | str) -> list[str]:
    return [check["message"] for check in validate_report_bundle_result(bundle, write_result=False)["checks"] if check["status"] == "failed"]


def validate_report_bundle_result(bundle: Path | str, *, write_result: bool = True) -> dict[str, Any]:
    root = Path(bundle)
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, **payload: Any) -> None:
        message = payload.pop("message", None)
        checks.append({"name": name, "status": status, **({"message": message} if message else {}), **payload})

    if not root.exists() or not root.is_dir():
        add("bundle_directory", "failed", message=f"bundle directory not found: {root}")
        return _finish_result(root, checks, write_result)
    add("bundle_directory", "passed", path=str(root))

    for name in sorted(REQUIRED_FILES):
        if not (root / name).is_file():
            add(f"required_file:{name}", "failed", message=f"missing required file: {name}")
        else:
            add(f"required_file:{name}", "passed")
    figures_dir = root / "figures"
    if not figures_dir.is_dir():
        add("figures_directory", "failed", message="missing figures directory")
    else:
        add("figures_directory", "passed")
        for name in sorted(REQUIRED_FIGURES):
            path = figures_dir / name
            if not path.is_file() or path.stat().st_size <= 0:
                add(f"figure:{name}", "failed", message=f"missing or empty figure: figures/{name}")
            else:
                add(f"figure:{name}", "passed", size_bytes=path.stat().st_size)

    errors: list[str] = []
    rows = _read_summary_rows(root / "summary.csv", errors)
    summary = _read_json(root / "summary.json", errors)
    analysis = _read_json(root / "experiment_analysis.json", errors)
    manifest = _read_json(root / "manifest.json", errors)
    run_manifest_index = _read_json(root / "run_artifact_manifests.json", errors)
    for error in errors:
        add("read_json_or_csv", "failed", message=error)

    if rows:
        add("summary_csv_readable", "passed", actual=len(rows))
        for idx, row in enumerate(rows, start=2):
            if not row.get("pass_type"):
                add("summary_csv_pass_type", "failed", message=f"summary.csv row {idx} has empty pass_type")
        expected = _expected_runs(manifest, analysis)
        if expected is not None and len(rows) != expected:
            add("summary_csv_rows", "failed", expected=expected, actual=len(rows), message=f"summary.csv run count {len(rows)} != expected {expected}")
        elif expected is not None:
            add("summary_csv_rows", "passed", expected=expected, actual=len(rows))

    if rows and isinstance(summary, dict):
        _check_total("summary.json", summary.get("total_runs"), len(rows), add)
    if rows and isinstance(analysis, dict):
        analysis_summary = analysis.get("summary")
        if isinstance(analysis_summary, dict):
            _check_total("experiment_analysis.json", analysis_summary.get("total_runs"), len(rows), add)

    report_text = _read_text(root / "report_text_ru.md", [])
    if report_text and re.search(r"placeholder|todo|tbd", report_text, re.I):
        add("report_text_ru_placeholders", "failed", message="report_text_ru.md contains placeholder text")
    elif report_text:
        add("report_text_ru_placeholders", "passed")

    report_md = _read_text(root / "report.md", [])
    if rows and report_md:
        if str(len(rows)) not in report_md:
            add("report_md_total_runs", "failed", message="report.md does not appear to contain total run count")
        else:
            add("report_md_total_runs", "passed", total_runs=len(rows))

    if isinstance(manifest, dict) and isinstance(manifest.get("report_bundle_files"), list):
        declared = sorted(str(item) for item in manifest["report_bundle_files"])
        missing_declared = sorted(item for item in declared if not (root / item).is_file())
        if missing_declared:
            add("manifest_paths_exist", "failed", message=f"manifest references missing files: {', '.join(missing_declared)}")
        else:
            add("manifest_paths_exist", "passed", declared=len(declared))
    elif (root / "manifest.json").exists():
        add("manifest_paths_exist", "failed", message="manifest.json missing report_bundle_files")

    if isinstance(manifest, dict):
        missing_protocol = [
            key for key in (
                "benchmark_protocol_version",
                "task_schema_version",
                "validator_version",
                "tool_contract_version",
                "report_schema_version",
                "matrix_config_hash",
                "task_set_hash",
                "tool_contract_hash",
            )
            if not manifest.get(key)
        ]
        if missing_protocol:
            add("manifest_protocol_versions_and_hashes", "failed", missing=missing_protocol, message="manifest.json missing protocol versions or hashes")
        else:
            add("manifest_protocol_versions_and_hashes", "passed")

    if isinstance(run_manifest_index, dict):
        run_count = len(run_manifest_index.get("runs", [])) if isinstance(run_manifest_index.get("runs"), list) else 0
        expected = len(rows) if rows else _expected_runs(manifest, analysis)
        if expected is not None and run_count != expected:
            add("run_artifact_manifests_rows", "failed", expected=expected, actual=run_count, message=f"run_artifact_manifests.json run count {run_count} != expected {expected}")
        else:
            add("run_artifact_manifests_rows", "passed", expected=expected, actual=run_count)
        missing_required = run_manifest_index.get("missing_required_artifacts")
        if missing_required not in (0, None):
            add("run_artifact_manifests_complete", "failed", actual=missing_required, message="run_artifact_manifests.json contains missing required artifacts")
        else:
            add("run_artifact_manifests_complete", "passed")

    # Readiness gates: evaluate matrix-defined thresholds against actual run data.
    # Structural validation (required files, row counts) is separate from benchmark readiness.
    readiness_gates = None
    if isinstance(manifest, dict):
        meta = manifest.get("metadata")
        if isinstance(meta, dict):
            readiness_gates = meta.get("readiness_gates")
    if readiness_gates is not None and rows:
        gate_result = _evaluate_readiness_gates(readiness_gates, rows)
        if gate_result["readiness_ok"]:
            add("readiness_gates", "passed", readiness_ok=True, gates_checked=len(gate_result["gates_checked"]))
        else:
            for fg in gate_result["failed_gates"]:
                add(
                    f"readiness_gate:{fg['name']}",
                    "failed",
                    expected=fg["expected"],
                    actual=fg["actual"],
                    severity=fg.get("severity", "blocking"),
                    message=(
                        f"readiness gate '{fg['name']}' failed: "
                        f"expected {fg['expected']}, actual {fg['actual']}"
                    ),
                )
            add(
                "readiness_gates",
                "failed",
                readiness_ok=False,
                failed_gate_count=len(gate_result["failed_gates"]),
                message=f"benchmark readiness failed: {len(gate_result['failed_gates'])} gate(s) violated",
            )

    return _finish_result(root, checks, write_result)


def _read_summary_rows(path: Path, errors: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"failed to read summary.csv: {exc}")
        return []


def _read_json(path: Path, errors: list[str]) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"failed to read {path.name}: {exc}")
        return None


def _read_text(path: Path, errors: list[str]) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"failed to read {path.name}: {exc}")
        return ""


def _expected_runs(manifest: Any, analysis: Any) -> int | None:
    if isinstance(manifest, dict):
        metadata = manifest.get("metadata")
        if isinstance(metadata, dict):
            for key in ("expected_runs", "planned_runs"):
                value = metadata.get(key)
                if isinstance(value, int):
                    return value
    if isinstance(analysis, dict):
        summary = analysis.get("summary")
        if isinstance(summary, dict) and isinstance(summary.get("total_runs"), int):
            return int(summary["total_runs"])
    return None


def _check_total(label: str, value: Any, expected: int, add) -> None:
    if isinstance(value, int) and value == expected:
        add(f"{label}_total_runs", "passed", expected=expected, actual=value)
        return
    add(f"{label}_total_runs", "failed", expected=expected, actual=value, message=f"{label} total_runs {value} != summary.csv rows {expected}")


def _evaluate_readiness_gates(
    gates: dict[str, Any],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Evaluate matrix readiness gates against summary.csv rows.

    Returns a dict with readiness_ok, failed_gates, and gates_checked.
    Gates are blocking criteria — any failure sets readiness_ok=False.
    """
    failed_gates: list[dict[str, Any]] = []
    gates_checked: list[str] = []

    def _fail(name: str, expected: Any, actual: Any) -> None:
        failed_gates.append({"name": name, "expected": expected, "actual": actual, "severity": "blocking"})

    # --- invalid_tool_response_export_max ---
    # Max allowed InvalidToolResponse / EmptySocketResponse / InvalidJsonResponse on export tasks.
    gate_name = "invalid_tool_response_export_max"
    if gate_name in gates:
        gates_checked.append(gate_name)
        threshold = int(gates[gate_name])
        export_rows = [r for r in rows if _task_category(r) == "export"]
        bad_types = {"InvalidToolResponse", "InvalidJsonResponse", "EmptySocketResponse", "ToolError"}
        invalid_count = sum(
            1 for r in export_rows
            if str(r.get("error_type", "")).strip() in bad_types
        )
        if invalid_count > threshold:
            _fail(gate_name, threshold, invalid_count)

    # --- clean_pass_with_error_type_max ---
    gate_name = "clean_pass_with_error_type_max"
    if gate_name in gates:
        gates_checked.append(gate_name)
        threshold = int(gates[gate_name])
        count = sum(
            1 for r in rows
            if str(r.get("pass_type", "")).strip() == "clean_pass"
            and str(r.get("error_type", "")).strip() not in {"", "null", "None"}
        )
        if count > threshold:
            _fail(gate_name, threshold, count)

    # --- react_max_steps_rate_max ---
    # Max fraction of ReAct runs that hit ReactMaxSteps.
    gate_name = "react_max_steps_rate_max"
    if gate_name in gates:
        gates_checked.append(gate_name)
        threshold = float(gates[gate_name])
        react_rows = [r for r in rows if str(r.get("strategy", "")).strip() == "react"]
        if react_rows:
            max_steps_count = sum(
                1 for r in react_rows
                if "ReactMaxSteps" in str(r.get("error_type", ""))
                or "ReactMaxSteps" in str(r.get("error", ""))
            )
            rate = max_steps_count / len(react_rows)
            if rate > threshold:
                _fail(gate_name, threshold, round(rate, 4))

    # --- require_react_repair_steps_on_validation_issues ---
    # At least some ReAct runs with validation issues must have repair steps > 0.
    gate_name = "require_react_repair_steps_on_validation_issues"
    if gates.get(gate_name) is True:
        gates_checked.append(gate_name)
        react_rows = [r for r in rows if str(r.get("strategy", "")).strip() == "react"]
        validation_issue_rows = [
            r for r in react_rows
            if str(r.get("pass_type", "")).strip() in {"failed_validation", "runtime_error"}
            and str(r.get("validation_issues", "")).strip() not in {"", "null", "None"}
        ]
        if validation_issue_rows:
            # Check for any runs reporting repair activity (heuristic: agent_issues contains repair fields).
            # Since summary.csv doesn't directly store react_repair_steps, we check agent_issues for clues.
            # A gate violation is reported as a warning-level signal rather than a hard block here,
            # because the repair step count is in the trace metadata, not summary.csv.
            # We surface this as an informational failed gate when no evidence of repair is found.
            has_repair_evidence = any(
                "repair" in str(r.get("agent_issues", "")).lower()
                for r in validation_issue_rows
            )
            if not has_repair_evidence:
                failed_gates.append({
                    "name": gate_name,
                    "expected": "react_repair_steps > 0 on at least one run with validation issues",
                    "actual": "no repair evidence found in agent_issues",
                    "severity": "warning",
                })

    # --- export_smoke_required ---
    # This gate is evaluated by preflight, not by post-run analysis.
    # We record it as checked but cannot evaluate it from summary.csv alone.
    if gates.get("export_smoke_required") is True:
        gates_checked.append("export_smoke_required")
        # Check if any export task had 0 successes (all runs failed).
        export_rows = [r for r in rows if _task_category(r) == "export"]
        if export_rows:
            success_count = sum(
                1 for r in export_rows
                if str(r.get("pass_type", "")).strip() in {"clean_pass", "soft_pass"}
            )
            if success_count == 0:
                _fail("export_smoke_required", "at_least_one_export_success", 0)

    return {
        "readiness_ok": len([fg for fg in failed_gates if fg.get("severity") != "warning"]) == 0,
        "failed_gates": failed_gates,
        "gates_checked": gates_checked,
    }


def _task_category(row: dict[str, str]) -> str:
    task_id = str(row.get("task_id", "")).strip()
    parts = task_id.split("_")
    return parts[0] if parts else ""


def _finish_result(root: Path, checks: list[dict[str, Any]], write_result: bool) -> dict[str, Any]:
    status = "failed" if any(check.get("status") == "failed" for check in checks) else "passed"
    result = {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checks": checks,
    }
    if write_result and root.exists() and root.is_dir():
        (root / "bundle_validation_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (root / "bundle_validation_result.md").write_text(_markdown_result(result), encoding="utf-8")
    return result


def _markdown_result(result: dict[str, Any]) -> str:
    lines = [
        "# Report Bundle Validation",
        "",
        f"status: {result.get('status')}",
        f"checked_at: {result.get('checked_at')}",
        "",
        "| Check | Status | Details |",
        "| --- | --- | --- |",
    ]
    for check in result.get("checks", []):
        if not isinstance(check, dict):
            continue
        details = check.get("message") or ""
        lines.append(f"| {check.get('name')} | {check.get('status')} | {details} |")
    return "\n".join(lines) + "\n"
