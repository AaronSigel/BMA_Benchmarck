from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.runner.error_classification import is_hard_model_failure, is_soft_success_diagnostic


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
        if isinstance(manifest, dict):
            metadata = manifest.get("metadata")
            if isinstance(metadata, dict):
                planned = metadata.get("planned_runs")
                expected_runs = metadata.get("expected_runs")
                if isinstance(planned, int) and isinstance(expected_runs, int) and planned != expected_runs:
                    add(
                        "manifest_planned_vs_expected_runs",
                        "failed",
                        expected=expected_runs,
                        actual=planned,
                        message=f"manifest planned_runs {planned} != expected_runs {expected_runs}",
                    )
                elif isinstance(planned, int) and isinstance(expected_runs, int):
                    add(
                        "manifest_planned_vs_expected_runs",
                        "passed",
                        expected=expected_runs,
                        actual=planned,
                    )

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
    gate_result: dict[str, Any] | None = None
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

    return _finish_result(root, checks, write_result, gate_result=gate_result)


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
    warning_gates: list[dict[str, Any]] = []
    gates_checked: list[str] = []

    if gates.get("require_hybrid_repair_used_on_failed_validation") is True:
        gates = {
            **gates,
            "hybrid_repair_activation_required_for_failed_validation": True,
        }

    def _affected_run_entry(row: dict[str, str]) -> dict[str, str]:
        artifact_dir = str(row.get("artifact_dir", "")).strip()
        source_file = ""
        if artifact_dir:
            source_file = f"{artifact_dir}/run_result.json"
        return {
            "run_id": str(row.get("run_id", "")).strip(),
            "task_id": str(row.get("task_id", "")).strip(),
            "strategy": str(row.get("strategy", "")).strip(),
            "agent_id": str(row.get("agent_id", "")).strip(),
            "mcp_profile": str(row.get("mcp_profile", "")).strip(),
            "error_type": str(row.get("error_type", "")).strip(),
            "error_class": str(row.get("error_class", "")).strip(),
            "is_infra_failure": str(row.get("is_infra_failure", "")).strip(),
            "source_file": source_file,
        }

    def _fail(
        name: str,
        expected: Any,
        actual: Any,
        *,
        severity: str = "blocking",
        gate_category: str | None = None,
        affected_runs: list[dict[str, str]] | None = None,
    ) -> None:
        entry = {"name": name, "expected": expected, "actual": actual, "severity": severity}
        if gate_category:
            entry["gate_category"] = gate_category
        if affected_runs:
            entry["affected_runs"] = affected_runs
        if severity == "warning":
            warning_gates.append(entry)
        else:
            failed_gates.append(entry)

    def _check_rate_gate(
        gate_name: str,
        *,
        actual_rate: float | None,
        comparator: str,
        threshold: float,
        severity: str = "blocking",
        gate_category: str | None = None,
    ) -> None:
        if gate_name not in gates or actual_rate is None:
            return
        gates_checked.append(gate_name)
        threshold_value = float(gates[gate_name])
        failed = (
            actual_rate < threshold_value if comparator == "min"
            else actual_rate > threshold_value
        )
        if failed:
            _fail(gate_name, threshold_value, round(actual_rate, 4), severity=severity, gate_category=gate_category)

    # --- invalid_tool_response_export_max ---
    # Max allowed InvalidToolResponse / InvalidJsonResponse on export tasks (response-parse failures only).
    gate_name = "invalid_tool_response_export_max"
    if gate_name in gates:
        gates_checked.append(gate_name)
        threshold = int(gates[gate_name])
        export_rows = [r for r in rows if _task_category(r) == "export"]
        invalid_response_types = {"InvalidToolResponse", "InvalidJsonResponse"}

        def _invalid_tool_response_export_row(row: dict[str, str]) -> bool:
            if _bool_metric(row.get("is_infra_failure")):
                return False
            return str(row.get("error_type", "")).strip() in invalid_response_types

        affected = [r for r in export_rows if _invalid_tool_response_export_row(r)]
        invalid_count = len(affected)
        if invalid_count > threshold:
            _fail(
                gate_name,
                threshold,
                invalid_count,
                affected_runs=[_affected_run_entry(r) for r in affected],
            )

    # --- clean_pass_with_error_type_max ---
    gate_name = "clean_pass_with_error_type_max"
    if gate_name in gates:
        gates_checked.append(gate_name)
        threshold = int(gates[gate_name])
        affected = [
            r for r in rows
            if str(r.get("pass_type", "")).strip() == "clean_pass"
            and str(r.get("error_type", "")).strip() not in {"", "null", "None"}
        ]
        count = len(affected)
        if count > threshold:
            _fail(
                gate_name,
                threshold,
                count,
                affected_runs=[_affected_run_entry(r) for r in affected],
            )

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
    gate_name = "require_react_repair_steps_on_validation_issues"
    if gates.get(gate_name) is True:
        gates_checked.append(gate_name)
        react_rows = [r for r in rows if str(r.get("strategy", "")).strip() == "react"]
        validation_issue_rows = [
            r for r in react_rows
            if str(r.get("pass_type", "")).strip() in {"failed_validation", "runtime_error", "soft_pass"}
            and str(r.get("validation_issues", "")).strip() not in {"", "null", "None"}
        ]
        if validation_issue_rows:
            has_repair_steps = any(_int_metric(r.get("react_repair_steps")) > 0 for r in validation_issue_rows)
            if not has_repair_steps:
                _fail(
                    gate_name,
                    "react_repair_steps > 0 on at least one run with validation issues",
                    0,
                )

    # --- hybrid_repair_activation_required_for_failed_validation ---
    gate_name = "hybrid_repair_activation_required_for_failed_validation"
    if gates.get(gate_name) is True:
        gates_checked.append(gate_name)
        hybrid_rows = [
            r for r in rows
            if str(r.get("strategy", "")).strip() == "plan_execute_react_repair"
            and str(r.get("pass_type", "")).strip() in {"failed_validation", "runtime_error", "soft_pass"}
        ]
        if hybrid_rows:
            activated = sum(
                1 for r in hybrid_rows
                if str(r.get("hybrid_repair_used", "")).strip().lower() in {"true", "1", "yes"}
            )
            if activated == 0:
                _fail(gate_name, "> 0 hybrid runs with hybrid_repair_used", 0)

    # --- snapshot_invalid_schema_max ---
    gate_name = "snapshot_invalid_schema_max"
    if gate_name in gates:
        gates_checked.append(gate_name)
        threshold = int(gates[gate_name])
        invalid_count = sum(
            1 for r in rows
            if str(r.get("repair_unavailable_reason", "")).strip() == "snapshot_invalid_schema"
        )
        if invalid_count > threshold:
            _fail(gate_name, threshold, invalid_count)

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

    _check_rate_gate(
        "reported_success_rate_min",
        actual_rate=_reported_success_rate(rows),
        comparator="min",
        threshold=float(gates.get("reported_success_rate_min", 0)),
    )
    _check_rate_gate(
        "runtime_error_rate_max",
        actual_rate=_runtime_error_rate(rows),
        comparator="max",
        threshold=float(gates.get("runtime_error_rate_max", 1)),
    )
    _check_rate_gate(
        "infra_error_rate_max",
        actual_rate=_infra_error_rate(rows),
        comparator="max",
        threshold=float(gates.get("infra_error_rate_max", 1)),
        gate_category="infra",
    )
    _check_rate_gate(
        "reset_failure_rate_max",
        actual_rate=_reset_failure_rate(rows),
        comparator="max",
        threshold=float(gates.get("reset_failure_rate_max", 1)),
        gate_category="infra",
    )
    _check_rate_gate(
        "snapshot_failure_rate_max",
        actual_rate=_snapshot_failure_rate(rows),
        comparator="max",
        threshold=float(gates.get("snapshot_failure_rate_max", 1)),
        gate_category="infra",
    )
    _check_rate_gate(
        "model_failure_rate_max",
        actual_rate=_model_failure_rate(rows),
        comparator="max",
        threshold=float(gates.get("model_failure_rate_max", 1)),
    )
    _check_rate_gate(
        "validation_failure_rate_max",
        actual_rate=_validation_failure_rate(rows),
        comparator="max",
        threshold=float(gates.get("validation_failure_rate_max", 1)),
    )
    if "socket_timeout_count_max" in gates:
        gates_checked.append("socket_timeout_count_max")
        timeout_count = sum(
            1 for r in rows
            if str(r.get("error_type", "")).strip() in {"ToolTimeout", "SocketTimeout"}
        )
        threshold = int(gates["socket_timeout_count_max"])
        if timeout_count > threshold:
            infra_rate = _infra_error_rate(rows)
            infra_ceiling = float(gates.get("infra_error_rate_max", 0.05))
            severity = "warning" if infra_rate is not None and infra_rate <= infra_ceiling else "blocking"
            _fail(
                "socket_timeout_count_max",
                threshold,
                timeout_count,
                severity=severity,
                gate_category="infra",
            )
    if "empty_socket_response_count_max" in gates:
        gates_checked.append("empty_socket_response_count_max")
        affected = [
            r for r in rows
            if str(r.get("error_type", "")).strip() == "EmptySocketResponse"
        ]
        empty_count = len(affected)
        threshold = int(gates["empty_socket_response_count_max"])
        if empty_count > threshold:
            infra_rate = _infra_error_rate(rows)
            infra_ceiling = float(gates.get("infra_error_rate_max", 0.05))
            severity = "warning" if infra_rate is not None and infra_rate <= infra_ceiling else "blocking"
            _fail(
                "empty_socket_response_count_max",
                threshold,
                empty_count,
                severity=severity,
                gate_category="infra",
                affected_runs=[_affected_run_entry(r) for r in affected],
            )
    for category, gate_name in (
        ("geometry", "geometry_success_rate_min"),
        ("materials", "materials_success_rate_min"),
        ("camera", "camera_success_rate_min"),
        ("lighting", "lighting_success_rate_min"),
        ("export", "export_success_rate_min"),
    ):
        if gate_name in gates:
            _check_rate_gate(
                gate_name,
                actual_rate=_category_success_rate(rows, category),
                comparator="min",
                threshold=float(gates[gate_name]),
            )

    return {
        "readiness_ok": len(failed_gates) == 0,
        "failed_gates": failed_gates,
        "warning_gates": warning_gates,
        "gates_checked": gates_checked,
    }


def _int_metric(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def _task_category(row: dict[str, str]) -> str:
    explicit = str(row.get("task_category", "")).strip()
    if explicit:
        return explicit
    task_id = str(row.get("task_id", "")).strip()
    parts = task_id.split("_")
    return parts[0] if parts else ""


def _reported_success_rate(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    passed = sum(1 for row in rows if str(row.get("pass_type", "")).strip() in {"clean_pass", "soft_pass"})
    return passed / len(rows)


def _runtime_error_rate(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    errors = sum(1 for row in rows if str(row.get("pass_type", "")).strip() == "runtime_error")
    return errors / len(rows)


def _bool_metric(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _infra_error_rate(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    count = sum(1 for row in rows if _bool_metric(row.get("is_infra_failure")))
    return count / len(rows)


def _row_is_hard_model_failure(row: dict[str, str]) -> bool:
    return is_hard_model_failure(
        is_model_failure=_bool_metric(row.get("is_model_failure")),
        is_infra_failure=_bool_metric(row.get("is_infra_failure")),
        error_class=str(row.get("error_class", "")).strip() or None,
        diagnostic_only=_bool_metric(row.get("diagnostic_only")),
        pass_type=str(row.get("pass_type", "")).strip() or None,
        scene_status=str(row.get("scene_status", "")).strip() or None,
        error_type=str(row.get("error_type", "")).strip() or None,
    )


def _row_is_soft_success_diagnostic(row: dict[str, str]) -> bool:
    return is_soft_success_diagnostic(
        error_class=str(row.get("error_class", "")).strip() or None,
        diagnostic_only=_bool_metric(row.get("diagnostic_only")),
    ) or (
        str(row.get("pass_type", "")).strip() == "soft_pass"
        and str(row.get("scene_status", "")).strip() == "passed"
        and str(row.get("error_type", "")).strip() in {"ReactMaxSteps", "ReactInvalidAction", "ReactNoProgress"}
        and not _bool_metric(row.get("is_infra_failure"))
    )


def _model_failure_rate(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    count = sum(1 for row in rows if _row_is_hard_model_failure(row))
    return count / len(rows)


def _soft_success_diagnostic_rate(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    count = sum(1 for row in rows if _row_is_soft_success_diagnostic(row))
    return count / len(rows)


def _validation_failure_rate(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    count = sum(1 for row in rows if _bool_metric(row.get("is_validation_failure")))
    return count / len(rows)


def _reset_failure_rate(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    count = sum(
        1 for row in rows
        if str(row.get("error_type", "")).strip() == "ResetSceneFailed"
        or str(row.get("failure_stage", "")).strip() == "reset_scene"
    )
    return count / len(rows)


def _snapshot_failure_rate(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    count = sum(
        1 for row in rows
        if str(row.get("error_type", "")).strip() == "SnapshotUnavailable"
        or "snapshot" in str(row.get("failure_stage", "")).lower()
    )
    return count / len(rows)


def _category_success_rate(rows: list[dict[str, str]], category: str) -> float | None:
    category_rows = [row for row in rows if _task_category(row) == category]
    if not category_rows:
        return None
    passed = sum(
        1 for row in category_rows
        if str(row.get("pass_type", "")).strip() in {"clean_pass", "soft_pass"}
    )
    return passed / len(category_rows)


def _finish_result(
    root: Path,
    checks: list[dict[str, Any]],
    write_result: bool,
    *,
    gate_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    structural_checks = [c for c in checks if not str(c.get("name", "")).startswith("readiness_gate")]
    structural_failed = any(check.get("status") == "failed" for check in structural_checks)
    structural_validity = "failed" if structural_failed else "passed"
    readiness_ok = True if gate_result is None else bool(gate_result.get("readiness_ok"))
    failed_gates = gate_result.get("failed_gates", []) if isinstance(gate_result, dict) else []
    warning_gates = gate_result.get("warning_gates", []) if isinstance(gate_result, dict) else []
    status = "failed" if structural_failed or (gate_result is not None and not readiness_ok) else "passed"
    result = {
        "status": status,
        "structural_validity": structural_validity,
        "readiness_ok": readiness_ok,
        "failed_gates": failed_gates,
        "warning_gates": warning_gates,
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checks": checks,
    }
    if write_result and root.exists() and root.is_dir():
        (root / "bundle_validation_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (root / "bundle_validation_result.md").write_text(_markdown_result(result), encoding="utf-8")
        readiness_payload = {
            "structural_validity": structural_validity,
            "readiness_ok": readiness_ok,
            "failed_gates": failed_gates,
            "warning_gates": warning_gates,
            "gates_checked": gate_result.get("gates_checked", []) if isinstance(gate_result, dict) else [],
        }
        (root / "readiness_result.json").write_text(json.dumps(readiness_payload, indent=2), encoding="utf-8")
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
