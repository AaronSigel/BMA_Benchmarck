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
    "agent_id",
    "strategy",
    "model",
    "mcp_profile",
    "success",
    "scene_total_score",
    "validation_status",
    "tool_call_count",
    "invalid_tool_call_count",
    "llm_call_count",
    "duration_sec",
    "error_count",
    "most_common_error",
]


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
    return {
        "run_id": result.run_id,
        "task_id": result.task_id,
        "agent_id": result.agent_id,
        "strategy": result.strategy,
        "model": result.model or "",
        "mcp_profile": result.mcp_profile or "",
        "success": result.success,
        "scene_total_score": result.total_score,
        "validation_status": result.validation_status or "",
        "tool_call_count": result.tool_call_count,
        "invalid_tool_call_count": result.invalid_tool_call_count,
        "llm_call_count": result.llm_call_count,
        "duration_sec": result.duration_sec,
        "error_count": result.error_count,
        "most_common_error": _most_common_error(result),
    }


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
    columns = ["dimension", "value", "run_count", "success_rate", "avg_score", "avg_tool_calls", "avg_duration_sec"]
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
