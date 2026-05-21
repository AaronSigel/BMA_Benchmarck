from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from benchmark.analysis.models import (
    ComparisonGroup,
    ComparisonReport,
    ErrorRecord,
    ExperimentAnalysisResult,
    ReportArtifact,
    ReportConfig,
    RunAnalysisResult,
)
from benchmark.analysis.report_builder import build_summary_table


def to_json(data: Any, indent: int = 2) -> str:
    if hasattr(data, "model_dump"):
        return json.dumps(data.model_dump(), indent=indent, default=str)
    if isinstance(data, list) and data and hasattr(data[0], "model_dump"):
        return json.dumps([item.model_dump() for item in data], indent=indent, default=str)
    return json.dumps(data, indent=indent, default=str)


def to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def to_markdown(rows: list[dict[str, Any]], title: str = "") -> str:
    if not rows:
        return f"# {title}\n\n_No data._\n" if title else "_No data._\n"
    headers = list(rows[0].keys())
    lines: list[str] = []
    if title:
        lines.append(f"# {title}\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        cells = [str(row.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_html(rows: list[dict[str, Any]], title: str = "") -> str:
    headers = list(rows[0].keys()) if rows else []
    header_row = "".join(f"<th>{h}</th>" for h in headers)
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{row.get(h, '')}</td>" for h in headers) + "</tr>"
        for row in rows
    )
    heading = f"<h1>{title}</h1>" if title else ""
    return (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head><body>"
        f"{heading}"
        f"<table border='1'><thead><tr>{header_row}</tr></thead>"
        f"<tbody>{body_rows}</tbody></table>"
        f"</body></html>"
    )


_RUN_METRICS_COLUMNS = [
    "run_id",
    "task_id",
    "task_category",
    "strategy",
    "model",
    "mcp_profile",
    "repetition",
    "pass_type",
    "run_status",
    "scene_status",
    "agent_status",
    "scene_passed_but_agent_error",
    "score",
    "validation_coverage",
    "duration_sec",
    "tool_call_count",
    "llm_call_count",
    "invalid_tool_call_count",
    "disabled_tool_call_count",
    "tool_error_count",
    "provider_name",
    "provider_reported_prompt_tokens",
    "provider_reported_completion_tokens",
    "provider_reported_total_tokens",
    "provider_reported_cost_usd",
    "provider_cost_available",
    "validation_issues",
    "agent_issues",
    "tool_issues",
    "export_issues",
    "error_type",
    "error_class",
    "error_source",
    "failure_stage",
    "is_model_failure",
    "is_agent_failure",
    "is_infra_failure",
    "is_validation_failure",
    "is_tool_runtime_failure",
    "is_scene_available",
    "scene_passed_before_error",
    "no_progress_reason",
    "early_stop_reason",
    "all_issues",
    "export_status",
    "export_failure_type",
    "artifact_dir",
    # Extra diagnostic columns retained for analysis convenience.
    "agent_id",
    "success",
    "validation_status",
    "validators_total",
    "validators_run",
    "validators_skipped",
    "validators_passed",
    "validators_failed",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "retry_count",
    "error_count",
    "most_common_error",
    "react_repair_steps",
    "deterministic_repair_steps",
    "hybrid_repair_used",
    "repair_unavailable_reason",
    "import_back_imported_object_count",
    "import_back_expected_object_count",
    "import_back_missing_objects",
    "import_back_extra_objects",
    "import_back_material_mismatches",
    "import_back_transform_mismatches",
    "import_back_import_error_type",
]


def _json_metric(value: Any) -> str:
    if value in (None, "", "null", "None"):
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _most_common_error(result: RunAnalysisResult) -> str:
    error_items = [
        (k[len("error."):], v)
        for k, v in result.metrics.items()
        if k.startswith("error.") and isinstance(v, int) and v > 0
    ]
    if not error_items:
        return ""
    return max(error_items, key=lambda x: x[1])[0]


def _run_to_row(result: RunAnalysisResult) -> dict[str, Any]:
    validation_issues = _format_issue_counts(result, "validation")
    agent_issues = _format_issue_counts(result, "agent")
    tool_issues = _format_issue_counts(result, "tool")
    export_issues = _format_issue_counts(result, "export")
    all_issues = _join_issues(validation_issues, agent_issues, tool_issues, export_issues)
    export_status, export_failure_type = _export_status(result, export_issues)
    return {
        "run_id": result.run_id,
        "task_id": result.task_id,
        "task_category": _task_category(result),
        "strategy": result.strategy,
        "model": result.model or "",
        "mcp_profile": result.mcp_profile or "",
        "repetition": result.metrics.get("run_summary.repetition", result.metrics.get("repetition", "")),
        "pass_type": _effective_pass_type(result),
        "run_status": result.run_status or "",
        "scene_status": result.scene_status or "",
        "agent_status": result.agent_status or "",
        "scene_passed_but_agent_error": _scene_passed_but_agent_error(result),
        "score": result.total_score,
        "validation_coverage": result.metrics.get("validation_coverage", ""),
        "duration_sec": result.duration_sec,
        "tool_call_count": result.tool_call_count,
        "llm_call_count": result.llm_call_count,
        "invalid_tool_call_count": result.invalid_tool_call_count,
        "disabled_tool_call_count": result.metrics.get("disabled_tool_call_count", ""),
        "tool_error_count": result.metrics.get("tool_error_count", ""),
        "provider_name": result.metrics.get("provider_name", ""),
        "provider_reported_prompt_tokens": result.metrics.get("provider_reported_prompt_tokens", ""),
        "provider_reported_completion_tokens": result.metrics.get("provider_reported_completion_tokens", ""),
        "provider_reported_total_tokens": result.metrics.get("provider_reported_total_tokens", ""),
        "provider_reported_cost_usd": result.metrics.get("provider_reported_cost_usd", ""),
        "provider_cost_available": result.metrics.get("provider_cost_available", ""),
        "validation_issues": validation_issues,
        "agent_issues": agent_issues,
        "tool_issues": tool_issues,
        "export_issues": export_issues,
        "error_type": result.metrics.get("structured_error_type", ""),
        "error_class": result.metrics.get("error_class", ""),
        "error_source": result.metrics.get("structured_error_source", ""),
        "failure_stage": result.metrics.get("failure_stage", ""),
        "is_model_failure": result.metrics.get("is_model_failure", ""),
        "is_agent_failure": result.metrics.get("is_agent_failure", ""),
        "is_infra_failure": result.metrics.get("is_infra_failure", ""),
        "is_validation_failure": result.metrics.get("is_validation_failure", ""),
        "is_tool_runtime_failure": result.metrics.get("is_tool_runtime_failure", ""),
        "is_scene_available": result.metrics.get("is_scene_available", ""),
        "scene_passed_before_error": result.metrics.get("scene_passed_before_error", ""),
        "no_progress_reason": result.metrics.get("no_progress_reason", ""),
        "early_stop_reason": result.metrics.get("early_stop_reason", ""),
        "all_issues": all_issues,
        "export_status": export_status,
        "export_failure_type": export_failure_type,
        "artifact_dir": _artifact_dir(result),
        "agent_id": result.agent_id,
        "success": result.success,
        "validation_status": result.validation_status or "",
        "validators_total": result.metrics.get("validators_total", ""),
        "validators_run": result.metrics.get("validators_run", ""),
        "validators_skipped": result.metrics.get("skipped_validator_count", ""),
        "validators_passed": result.metrics.get("passed_validator_count", ""),
        "validators_failed": result.metrics.get("failed_validator_count", ""),
        "prompt_tokens": result.metrics.get("prompt_tokens", ""),
        "completion_tokens": result.metrics.get("completion_tokens", ""),
        "total_tokens": result.metrics.get("total_tokens", ""),
        "retry_count": result.retry_count,
        "error_count": result.error_count,
        "most_common_error": _most_common_error(result),
        "react_repair_steps": result.metrics.get("react_repair_steps", ""),
        "deterministic_repair_steps": result.metrics.get("deterministic_repair_steps", ""),
        "hybrid_repair_used": result.metrics.get("hybrid_repair_used", ""),
        "repair_unavailable_reason": result.metrics.get("repair_unavailable_reason", ""),
        "import_back_imported_object_count": result.metrics.get("import_back.imported_object_count", ""),
        "import_back_expected_object_count": result.metrics.get("import_back.expected_object_count", ""),
        "import_back_missing_objects": _json_metric(result.metrics.get("import_back.missing_objects")),
        "import_back_extra_objects": _json_metric(result.metrics.get("import_back.extra_objects")),
        "import_back_material_mismatches": _json_metric(result.metrics.get("import_back.material_mismatches")),
        "import_back_transform_mismatches": _json_metric(result.metrics.get("import_back.transform_mismatches")),
        "import_back_import_error_type": result.metrics.get("import_back.import_error_type", ""),
    }


def _task_category(result: RunAnalysisResult) -> str:
    explicit = result.metrics.get("task_category") or result.metrics.get("run_summary.task_category")
    if isinstance(explicit, str) and explicit:
        return explicit
    task_id = result.task_id.lower()
    for category in ("geometry", "materials", "lighting", "camera", "export"):
        if category in task_id:
            return category
    return "unknown"


def _scene_passed_but_agent_error(result: RunAnalysisResult) -> bool:
    error_type = str(result.metrics.get("structured_error_type") or "").strip()
    has_error_type = error_type not in {"", "null", "None"}
    return (
        result.scene_status == "passed"
        and (
            (
                bool(result.agent_status)
                and result.agent_status not in {"completed", "completed_after_scene_passed"}
            )
            or has_error_type
        )
    )


def _effective_pass_type(result: RunAnalysisResult) -> str:
    if result.pass_type in {"clean_pass", "soft_pass", "failed_validation", "runtime_error"}:
        effective = str(result.pass_type)
    elif result.pass_type == "failed":
        effective = "failed_validation"
    elif result.pass_type == "error":
        effective = "runtime_error"
    elif result.success is True:
        effective = "clean_pass"
    elif result.success is False:
        effective = "failed_validation"
    else:
        effective = "runtime_error"
    if effective == "clean_pass" and _scene_passed_but_agent_error(result):
        return "soft_pass"
    error_type = str(result.metrics.get("structured_error_type") or "").strip()
    if effective == "clean_pass" and error_type not in {"", "null", "None"}:
        return "soft_pass"
    return effective


def _artifact_dir(result: RunAnalysisResult) -> str:
    for artifact in result.artifacts:
        path = Path(artifact)
        if path.name in {"agent_trace.json", "run_result.json", "validation_result.json", "scene_snapshot.json"}:
            return str(path.parent)
    return ""


def _join_issues(*values: str) -> str:
    items = [value for value in values if value and value != "null"]
    return "; ".join(items) if items else "null"


def _export_status(result: RunAnalysisResult, export_issues: str) -> tuple[str, str]:
    is_export = _task_category(result) == "export" or "export" in result.task_id.lower()
    if not is_export:
        return "not_applicable", ""
    pass_type = _effective_pass_type(result)
    if pass_type in {"clean_pass", "soft_pass"} and export_issues == "null":
        return "passed", "none"
    if pass_type == "runtime_error":
        return "failed", "pre_export_scene_incomplete"
    if export_issues != "null":
        first_issue = export_issues.split(":", 1)[0].split(";", 1)[0].strip()
        mapping = {
            "export_missing": "export_file_missing",
            "scene_export_missing": "export_file_missing",
            "export_empty_file": "export_file_invalid",
            "export_format_unsupported": "export_file_invalid",
            "export_file_invalid": "export_file_invalid",
            "export_import_missing": "import_back_failed",
            "export_import_file_too_small": "import_back_failed",
            "export_import_failed": "import_back_failed",
            "export_import_object_missing": "import_back_missing_objects",
            "export_import_mesh_count_mismatch": "import_back_missing_objects",
            "export_import_material_missing": "import_back_material_mismatch",
            "export_import_material_lost_after_export": "import_back_material_mismatch",
            "export_import_material_parameters_mismatch": "import_back_material_mismatch",
            "export_import_material_color_mismatch": "import_back_material_mismatch",
            "export_import_transform_mismatch": "import_back_transform_mismatch",
            "export_import_location_mismatch": "import_back_transform_mismatch",
            "export_import_scale_mismatch": "import_back_transform_mismatch",
            "export_import_rotation_mismatch": "import_back_transform_mismatch",
            "export_import_dimension_mismatch": "import_back_transform_mismatch",
        }
        return "failed", mapping.get(first_issue, first_issue or "export_tool_failed")
    return "failed", "pre_export_scene_incomplete"


def _format_issue_counts(result: RunAnalysisResult, kind: str) -> str:
    counts: dict[str, int] = {}
    if kind == "validation":
        for issue in result.issues:
            code = issue.get("code") if isinstance(issue, dict) else None
            if isinstance(code, str):
                counts[code] = counts.get(code, 0) + 1
    elif kind == "agent":
        for key in ("repeated_action_count", "duplicate_object_count", "wasted_step_count"):
            value = result.metrics.get(key)
            if isinstance(value, int) and value > 0:
                counts[key] = value
        structured = result.metrics.get("structured_error_type")
        if isinstance(structured, str) and structured:
            counts[structured] = counts.get(structured, 0) + 1
        if result.run_status == "error" or result.agent_status in {"runtime_error", "invalid_response", "max_steps_reached"}:
            counts[result.agent_status or "agent_error"] = 1
    elif kind == "tool":
        for key in ("invalid_tool_call_count", "disabled_tool_call_count", "tool_error_count"):
            value = result.metrics.get(key)
            if isinstance(value, int) and value > 0:
                counts[key] = value
    elif kind == "export":
        for issue in result.issues:
            code = issue.get("code") if isinstance(issue, dict) else None
            if isinstance(code, str) and ("export" in code or "glb" in code):
                counts[code] = counts.get(code, 0) + 1
    if not counts:
        return "null"
    return "; ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def write_run_analysis_json(result: RunAnalysisResult, path: Path | str) -> None:
    """Write a single RunAnalysisResult to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def write_experiment_analysis_json(result: ExperimentAnalysisResult, path: Path | str) -> None:
    """Write an ExperimentAnalysisResult to a JSON file (round-trips via Pydantic)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def write_run_metrics_csv(results: list[RunAnalysisResult], path: Path | str) -> None:
    """Write per-run metrics to a CSV file with a fixed column schema."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_RUN_METRICS_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(_run_to_row(result))


def write_group_comparison_csv(groups: list[ComparisonGroup], path: Path | str) -> None:
    """Write ComparisonGroup rows to a CSV file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dimension",
        "value",
        "run_count",
        "success_rate",
        "avg_score",
        "avg_tool_calls",
        "avg_duration_sec",
        "avg_cost",
        "validation_failures",
    ]
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for g in groups:
            writer.writerow({
                "dimension": g.dimension.value,
                "value": g.value,
                "run_count": g.run_count,
                "success_rate": g.success_rate,
                "avg_score": g.avg_score,
                "avg_tool_calls": g.avg_tool_calls,
                "avg_duration_sec": g.avg_duration_sec,
                "avg_cost": g.avg_cost,
                "validation_failures": g.validation_failures,
            })


def write_error_taxonomy_csv(
    errors: list[ErrorRecord] | dict[str, int],
    path: Path | str,
) -> None:
    """Write error taxonomy to a CSV file.

    Accepts either a list of ErrorRecord or an aggregated dict[category, count].
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(errors, dict):
        rows = [{"category": cat, "count": count} for cat, count in sorted(errors.items())]
        columns = ["category", "count"]
    else:
        rows = [
            {
                "run_id": e.run_id,
                "task_id": e.task_id,
                "step_index": e.step_index,
                "category": e.category.value,
                "tool_name": e.tool_name or "",
                "message": e.message,
            }
            for e in errors
        ]
        columns = ["run_id", "task_id", "step_index", "category", "tool_name", "message"]

    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export_reports(
    results: list[RunAnalysisResult],
    config: ReportConfig | None = None,
) -> list[ReportArtifact]:
    cfg = config or ReportConfig()
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    rows = build_summary_table(results)
    artifacts: list[ReportArtifact] = []

    writers = {
        "json": ("report.json", lambda: to_json(rows)),
        "csv": ("report.csv", lambda: to_csv(rows)),
        "markdown": ("report.md", lambda: to_markdown(rows, title=cfg.title)),
        "html": ("report.html", lambda: to_html(rows, title=cfg.title)),
    }

    for fmt in cfg.formats:
        if fmt not in writers:
            continue
        filename, render = writers[fmt]
        content = render()
        p = out / filename
        p.write_text(content, encoding="utf-8")
        artifacts.append(ReportArtifact(format=fmt, path=p, size_bytes=len(content.encode())))

    return artifacts


def export_comparison(
    report: ComparisonReport,
    config: ReportConfig | None = None,
) -> list[ReportArtifact]:
    cfg = config or ReportConfig()
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)

    rows = [g.model_dump() for g in report.groups]
    title = f"Comparison by {report.dimension.value}"
    artifacts: list[ReportArtifact] = []

    if "json" in cfg.formats:
        content = to_json(report)
        p = out / f"comparison_{report.dimension.value}.json"
        p.write_text(content, encoding="utf-8")
        artifacts.append(ReportArtifact(format="json", path=p, size_bytes=len(content.encode())))

    if "markdown" in cfg.formats:
        content = to_markdown(rows, title=title)
        p = out / f"comparison_{report.dimension.value}.md"
        p.write_text(content, encoding="utf-8")
        artifacts.append(ReportArtifact(format="markdown", path=p, size_bytes=len(content.encode())))

    return artifacts
