from __future__ import annotations

from collections import Counter
from typing import Any

from benchmark.agent.models import AgentTrace
from benchmark.analysis.agent_metrics import extract_agent_metrics
from benchmark.analysis.error_taxonomy import extract_errors
from benchmark.analysis.models import (
    ExperimentAnalysisResult,
    ReportConfig,
    RunAnalysisResult,
)
from benchmark.analysis.validation_metrics import (
    compute_validation_summary,
    extract_issues,
    extract_score_and_status,
    extract_validation_metrics,
)
from benchmark.validation.models import SceneValidationResult


def build_run_report(
    trace: AgentTrace,
    validation: SceneValidationResult | None = None,
    mcp_profile: str | None = None,
) -> RunAnalysisResult:
    result = extract_agent_metrics(trace)

    if mcp_profile is not None:
        result = result.model_copy(update={"mcp_profile": mcp_profile})

    val_summary = compute_validation_summary(validation)
    extra_metrics: dict[str, float | str | int | bool] = dict(result.metrics)

    # Always populate validation summary fields in metrics
    extra_metrics["scene_overall_status"] = val_summary.scene_overall_status
    extra_metrics["passed_validator_count"] = val_summary.passed_validator_count
    extra_metrics["failed_validator_count"] = val_summary.failed_validator_count
    extra_metrics["skipped_validator_count"] = val_summary.skipped_validator_count
    extra_metrics["validation_error_count"] = val_summary.validation_error_count
    extra_metrics["validation_warning_count"] = val_summary.validation_warning_count

    for field in (
        "object_score", "transform_score", "material_score",
        "light_score", "camera_score", "export_score",
    ):
        v = getattr(val_summary, field)
        if v is not None:
            extra_metrics[field] = v

    update: dict[str, object] = {
        "metrics": extra_metrics,
        "validation_status": val_summary.scene_overall_status,
    }

    if validation is not None:
        errors = extract_errors(trace)
        issues = extract_issues(validation)
        update["total_score"] = val_summary.scene_total_score
        update["issues"] = issues
        update["error_count"] = result.error_count + len(errors)

    result = result.model_copy(update=update)
    return result


def _na(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    header_row = "| " + " | ".join(headers) + " |"
    data_rows = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_row, sep] + data_rows) + "\n"


def _group_table(groups: list) -> str:
    if not groups:
        return "_No data._\n"
    headers = ["Value", "Runs", "Success Rate", "Avg Score", "Avg Tool Calls", "Avg Duration (s)"]
    rows = [
        [
            g.value,
            str(g.run_count),
            _na(round(g.success_rate, 3) if g.success_rate is not None else None),
            _na(round(g.avg_score, 4) if g.avg_score is not None else None),
            _na(round(g.avg_tool_calls, 1) if g.avg_tool_calls is not None else None),
            _na(round(g.avg_duration_sec, 1) if g.avg_duration_sec is not None else None),
        ]
        for g in groups
    ]
    return _md_table(headers, rows)


def _freshness_rows(metadata: dict[str, Any]) -> list[list[str]]:
    freshness = metadata.get("artifact_freshness")
    runtime = metadata.get("runtime")
    smoke = metadata.get("mcp_contract_smoke")
    rows: list[list[str]] = []
    if isinstance(freshness, dict):
        rows.extend(
            [
                ["Output root", _na(freshness.get("output_root"))],
                ["Run created at", _na(freshness.get("created_at"))],
                ["Clean output", _na(freshness.get("clean_output"))],
                ["Removed existing output", _na(freshness.get("removed_existing_output"))],
            ]
        )
    if isinstance(runtime, dict):
        socket_info = runtime.get("mcp_socket")
        socket_text = "N/A"
        if isinstance(socket_info, dict):
            socket_text = f"{socket_info.get('host')}:{socket_info.get('port')}"
        process = runtime.get("blender_process")
        pid = process.get("pid") if isinstance(process, dict) else None
        addon = process.get("addon_path") if isinstance(process, dict) else None
        rows.extend(
            [
                ["Latest artifact mtime", _na(runtime.get("latest_run_file_mtime"))],
                ["MCP profile", _na(runtime.get("mcp_profile"))],
                ["MCP socket", socket_text],
                ["Blender PID", _na(pid)],
                ["Addon path", _na(addon)],
                ["Tool contract hash", _na(runtime.get("tool_contract_hash"))],
                ["Git dirty", _na(runtime.get("git_dirty"))],
            ]
        )
    if isinstance(smoke, dict):
        rows.append(["Contract smoke", "passed" if smoke.get("ok") else "failed"])
    return rows


def build_markdown_report(
    analysis: ExperimentAnalysisResult,
    config: ReportConfig,
) -> str:
    from benchmark.analysis.comparison import (
        group_by_mcp_profile,
        group_by_model,
        group_by_strategy,
        group_by_task_category,
        rank_runs_by_score,
    )

    lines: list[str] = []
    s = analysis.summary
    runs = analysis.runs

    lines.append(f"# {config.title}")
    lines.append(f"\n**Experiment:** `{analysis.experiment_id}`\n")
    freshness_rows = _freshness_rows(analysis.metadata)
    if freshness_rows:
        lines.append("## Artifact Freshness\n")
        lines.append(_md_table(["Field", "Value"], freshness_rows))

    # ------------------------------------------------------------------
    # 1. Summary
    # ------------------------------------------------------------------
    lines.append("## 1. Summary\n")
    summary_headers = ["Metric", "Value"]
    summary_rows = [
        ["Total runs", str(s.total_runs)],
        ["Successful", str(s.successful_runs)],
        ["Failed", str(s.failed_runs)],
        ["Errors (no status)", str(s.error_runs)],
        ["Avg scene score", _na(round(s.average_scene_score, 4) if s.average_scene_score is not None else None)],
        ["Avg tool calls", _na(round(s.average_tool_call_count, 1) if s.average_tool_call_count is not None else None)],
        ["Avg duration (s)", _na(round(s.average_duration_sec, 1) if s.average_duration_sec is not None else None)],
        ["Avg LLM calls", _na(round(s.average_llm_calls, 1) if s.average_llm_calls is not None else None)],
        ["Best run", _na(s.best_run)],
        ["Worst run", _na(s.worst_run)],
    ]
    lines.append(_md_table(summary_headers, summary_rows))

    # ------------------------------------------------------------------
    # 2. Best / Worst runs
    # ------------------------------------------------------------------
    lines.append("## 2. Best / Worst Runs\n")
    if not runs:
        lines.append("_No runs._\n")
    else:
        ranked = rank_runs_by_score(runs)
        top3 = ranked[:3]
        bottom3 = [r for r in ranked[-3:] if r not in top3]

        lines.append("### Top runs\n")
        if top3:
            bw_headers = ["Rank", "Run ID", "Task", "Strategy", "Score", "Duration (s)", "Tool Calls"]
            bw_rows = [
                [
                    str(rr.rank),
                    rr.run.run_id,
                    rr.run.task_id,
                    rr.run.strategy,
                    _na(round(rr.score_used, 4) if rr.score_used is not None else None),
                    _na(round(rr.run.duration_sec, 1) if rr.run.duration_sec is not None else None),
                    str(rr.run.tool_call_count),
                ]
                for rr in top3
            ]
            lines.append(_md_table(bw_headers, bw_rows))
        else:
            lines.append("_No data._\n")

        lines.append("### Bottom runs\n")
        if bottom3:
            bw_rows_b = [
                [
                    str(rr.rank),
                    rr.run.run_id,
                    rr.run.task_id,
                    rr.run.strategy,
                    _na(round(rr.score_used, 4) if rr.score_used is not None else None),
                    _na(round(rr.run.duration_sec, 1) if rr.run.duration_sec is not None else None),
                    str(rr.run.tool_call_count),
                ]
                for rr in bottom3
            ]
            lines.append(_md_table(bw_headers, bw_rows_b))
        else:
            lines.append("_No data._\n")

    # ------------------------------------------------------------------
    # 3–6. Group comparisons (gated by config.include_group_comparison)
    # ------------------------------------------------------------------
    if config.include_group_comparison and runs:
        lines.append("## 3. Strategy Comparison\n")
        lines.append(_group_table(group_by_strategy(runs).groups))

        lines.append("## 4. MCP Profile Comparison\n")
        lines.append(_group_table(group_by_mcp_profile(runs).groups))

        lines.append("## 5. Model / Provider Comparison\n")
        lines.append(_group_table(group_by_model(runs).groups))

        lines.append("## 6. Task Category Comparison\n")
        lines.append(_group_table(group_by_task_category(runs).groups))

    # ------------------------------------------------------------------
    # 7. Error taxonomy (gated by config.include_error_taxonomy)
    # ------------------------------------------------------------------
    if config.include_error_taxonomy:
        lines.append("## 7. Error Taxonomy\n")
        error_counter: Counter[str] = Counter()
        for r in runs:
            for key, val in r.metrics.items():
                if key.startswith("error.") and isinstance(val, int) and val > 0:
                    error_counter[key[len("error."):]] += val
        if error_counter:
            err_headers = ["Category", "Count"]
            err_rows = [[cat, str(cnt)] for cat, cnt in error_counter.most_common()]
            lines.append(_md_table(err_headers, err_rows))
        else:
            lines.append("_No errors recorded._\n")

    # ------------------------------------------------------------------
    # 8. Run details (gated by config.include_runs)
    # ------------------------------------------------------------------
    if config.include_runs and runs:
        lines.append("## 8. Run Details\n")
        det_headers = ["Run ID", "Task", "Strategy", "Model", "MCP Profile", "Success", "Score", "Tool Calls", "LLM Calls", "Duration (s)", "Errors"]
        det_rows = [
            [
                r.run_id,
                r.task_id,
                r.strategy,
                _na(r.model),
                _na(r.mcp_profile),
                _na(r.success),
                _na(round(r.total_score, 4) if r.total_score is not None else None),
                str(r.tool_call_count),
                str(r.llm_call_count),
                _na(round(r.duration_sec, 1) if r.duration_sec is not None else None),
                str(r.error_count),
            ]
            for r in runs
        ]
        lines.append(_md_table(det_headers, det_rows))

    # ------------------------------------------------------------------
    # 9. Artifact links (gated by config.include_artifact_links)
    # ------------------------------------------------------------------
    if config.include_artifact_links and runs:
        lines.append("## 9. Artifact Links\n")
        any_artifacts = False
        for r in runs:
            if r.artifacts:
                any_artifacts = True
                lines.append(f"**{r.run_id}**\n")
                for art in r.artifacts:
                    lines.append(f"- [{art}]({art})")
                lines.append("")
        if not any_artifacts:
            lines.append("_No artifacts recorded._\n")

    return "\n".join(lines) + "\n"


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
  body { font-family: sans-serif; margin: 2em; color: #222; }
  h1 { border-bottom: 2px solid #555; padding-bottom: .3em; }
  h2 { margin-top: 2em; border-bottom: 1px solid #bbb; }
  h3 { margin-top: 1.2em; color: #444; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 1.5em; font-size: .9em; }
  th { background: #3a3a5c; color: #fff; padding: 6px 10px; text-align: left; }
  td { border: 1px solid #ddd; padding: 5px 10px; }
  tr:nth-child(even) { background: #f5f5f8; }
  .na { color: #aaa; font-style: italic; }
  .pass { color: #2a7a2a; font-weight: bold; }
  .fail { color: #b22; font-weight: bold; }
  a { color: #2255aa; }
  .meta { color: #666; font-size: .9em; margin-bottom: 1.5em; }
  .empty { color: #888; font-style: italic; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<p class="meta">Experiment: <code>{{ experiment_id }}</code></p>

{% if freshness_rows %}
<h2>Artifact Freshness</h2>
<table>
<tr><th>Field</th><th>Value</th></tr>
{% for label, val in freshness_rows %}
<tr><td>{{ label }}</td><td>{{ val }}</td></tr>
{% endfor %}
</table>
{% endif %}

<h2>1. Summary</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{% for label, val in summary_rows %}
<tr><td>{{ label }}</td><td>{{ val }}</td></tr>
{% endfor %}
</table>

<h2>2. Best / Worst Runs</h2>
<h3>Top runs</h3>
{% if top_runs %}
<table>
<tr>{% for h in run_rank_headers %}<th>{{ h }}</th>{% endfor %}</tr>
{% for row in top_runs %}
<tr>{% for cell in row %}<td>{{ cell | safe }}</td>{% endfor %}</tr>
{% endfor %}
</table>
{% else %}<p class="empty">No data.</p>{% endif %}
<h3>Bottom runs</h3>
{% if bottom_runs %}
<table>
<tr>{% for h in run_rank_headers %}<th>{{ h }}</th>{% endfor %}</tr>
{% for row in bottom_runs %}
<tr>{% for cell in row %}<td>{{ cell | safe }}</td>{% endfor %}</tr>
{% endfor %}
</table>
{% else %}<p class="empty">No data.</p>{% endif %}

{% if include_group_comparison %}
<h2>3. Strategy Comparison</h2>
{{ group_tables.strategy }}
<h2>4. MCP Profile Comparison</h2>
{{ group_tables.mcp_profile }}
<h2>5. Model / Provider Comparison</h2>
{{ group_tables.model }}
<h2>6. Task Category Comparison</h2>
{{ group_tables.task_category }}
{% endif %}

{% if include_error_taxonomy %}
<h2>7. Error Taxonomy</h2>
{% if error_rows %}
<table>
<tr><th>Category</th><th>Count</th></tr>
{% for cat, cnt in error_rows %}
<tr><td>{{ cat }}</td><td>{{ cnt }}</td></tr>
{% endfor %}
</table>
{% else %}<p class="empty">No errors recorded.</p>{% endif %}
{% endif %}

{% if include_runs and run_detail_rows %}
<h2>8. Run Details</h2>
<table>
<tr>{% for h in run_detail_headers %}<th>{{ h }}</th>{% endfor %}</tr>
{% for row in run_detail_rows %}
<tr>{% for cell in row %}<td>{{ cell | safe }}</td>{% endfor %}</tr>
{% endfor %}
</table>
{% endif %}

{% if include_artifact_links %}
<h2>9. Artifact Links</h2>
{% if artifact_sections %}
{% for run_id, arts in artifact_sections %}
<p><strong>{{ run_id }}</strong></p>
<ul>{% for art in arts %}<li><a href="{{ art }}">{{ art }}</a></li>{% endfor %}</ul>
{% endfor %}
{% else %}<p class="empty">No artifacts recorded.</p>{% endif %}
{% endif %}

</body>
</html>
"""


def _html_na(value: object) -> str:
    if value is None:
        return '<span class="na">N/A</span>'
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _html_bool(value: object) -> str:
    if value is True:
        return '<span class="pass">✓ pass</span>'
    if value is False:
        return '<span class="fail">✗ fail</span>'
    return '<span class="na">N/A</span>'


def _html_group_table(groups: list) -> str:
    if not groups:
        return '<p class="empty">No data.</p>'
    headers = ["Value", "Runs", "Success Rate", "Avg Score", "Avg Tool Calls", "Avg Duration (s)"]
    rows_html = "".join(
        "<tr>"
        + f"<td>{g.value}</td>"
        + f"<td>{g.run_count}</td>"
        + f"<td>{_html_na(round(g.success_rate, 3) if g.success_rate is not None else None)}</td>"
        + f"<td>{_html_na(round(g.avg_score, 4) if g.avg_score is not None else None)}</td>"
        + f"<td>{_html_na(round(g.avg_tool_calls, 1) if g.avg_tool_calls is not None else None)}</td>"
        + f"<td>{_html_na(round(g.avg_duration_sec, 1) if g.avg_duration_sec is not None else None)}</td>"
        + "</tr>"
        for g in groups
    )
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    return f"<table><tr>{header_html}</tr>{rows_html}</table>"


def build_html_report(
    analysis: ExperimentAnalysisResult,
    config: ReportConfig,
) -> str:
    from jinja2 import Environment
    from benchmark.analysis.comparison import (
        group_by_mcp_profile,
        group_by_model,
        group_by_strategy,
        group_by_task_category,
        rank_runs_by_score,
    )

    env = Environment(autoescape=False)
    template = env.from_string(_HTML_TEMPLATE)

    s = analysis.summary
    runs = analysis.runs

    summary_rows = [
        ("Total runs", str(s.total_runs)),
        ("Successful", str(s.successful_runs)),
        ("Failed", str(s.failed_runs)),
        ("Errors (no status)", str(s.error_runs)),
        ("Avg scene score", _html_na(round(s.average_scene_score, 4) if s.average_scene_score is not None else None)),
        ("Avg tool calls", _html_na(round(s.average_tool_call_count, 1) if s.average_tool_call_count is not None else None)),
        ("Avg duration (s)", _html_na(round(s.average_duration_sec, 1) if s.average_duration_sec is not None else None)),
        ("Avg LLM calls", _html_na(round(s.average_llm_calls, 1) if s.average_llm_calls is not None else None)),
        ("Best run", _html_na(s.best_run)),
        ("Worst run", _html_na(s.worst_run)),
    ]
    freshness_rows = _freshness_rows(analysis.metadata)

    run_rank_headers = ["Rank", "Run ID", "Task", "Strategy", "Score", "Duration (s)", "Tool Calls"]

    def _rank_row(rr: object) -> list[str]:
        return [
            str(rr.rank),
            rr.run.run_id,
            rr.run.task_id,
            rr.run.strategy,
            _html_na(round(rr.score_used, 4) if rr.score_used is not None else None),
            _html_na(round(rr.run.duration_sec, 1) if rr.run.duration_sec is not None else None),
            str(rr.run.tool_call_count),
        ]

    top_runs: list[list[str]] = []
    bottom_runs: list[list[str]] = []
    if runs:
        ranked = rank_runs_by_score(runs)
        top_runs = [_rank_row(rr) for rr in ranked[:3]]
        remaining = ranked[3:]
        bottom_runs = [_rank_row(rr) for rr in remaining[-3:]] if remaining else []

    group_tables: dict[str, str] = {}
    if config.include_group_comparison and runs:
        group_tables["strategy"] = _html_group_table(group_by_strategy(runs).groups)
        group_tables["mcp_profile"] = _html_group_table(group_by_mcp_profile(runs).groups)
        group_tables["model"] = _html_group_table(group_by_model(runs).groups)
        group_tables["task_category"] = _html_group_table(group_by_task_category(runs).groups)

    error_rows: list[tuple[str, int]] = []
    if config.include_error_taxonomy:
        ec: Counter[str] = Counter()
        for r in runs:
            for key, val in r.metrics.items():
                if key.startswith("error.") and isinstance(val, int) and val > 0:
                    ec[key[len("error."):]] += val
        error_rows = ec.most_common()

    run_detail_headers = ["Run ID", "Task", "Strategy", "Model", "MCP Profile", "Success", "Score", "Tool Calls", "LLM Calls", "Duration (s)", "Errors"]
    run_detail_rows: list[list[str]] = []
    if config.include_runs:
        for r in runs:
            run_detail_rows.append([
                r.run_id,
                r.task_id,
                r.strategy,
                _html_na(r.model),
                _html_na(r.mcp_profile),
                _html_bool(r.success),
                _html_na(round(r.total_score, 4) if r.total_score is not None else None),
                str(r.tool_call_count),
                str(r.llm_call_count),
                _html_na(round(r.duration_sec, 1) if r.duration_sec is not None else None),
                str(r.error_count),
            ])

    artifact_sections: list[tuple[str, list[str]]] = []
    if config.include_artifact_links:
        for r in runs:
            if r.artifacts:
                artifact_sections.append((r.run_id, r.artifacts))

    return template.render(
        title=config.title,
        experiment_id=analysis.experiment_id,
        summary_rows=summary_rows,
        freshness_rows=freshness_rows,
        run_rank_headers=run_rank_headers,
        top_runs=top_runs,
        bottom_runs=bottom_runs,
        include_group_comparison=config.include_group_comparison and bool(runs),
        group_tables=group_tables,
        include_error_taxonomy=config.include_error_taxonomy,
        error_rows=error_rows,
        include_runs=config.include_runs,
        run_detail_headers=run_detail_headers,
        run_detail_rows=run_detail_rows,
        include_artifact_links=config.include_artifact_links,
        artifact_sections=artifact_sections,
    )


def build_summary_table(results: list[RunAnalysisResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in results:
        rows.append({
            "run_id": r.run_id,
            "task_id": r.task_id,
            "agent_id": r.agent_id,
            "strategy": r.strategy,
            "model": r.model,
            "mcp_profile": r.mcp_profile,
            "success": r.success,
            "total_score": r.total_score,
            "validation_status": r.validation_status,
            "tool_call_count": r.tool_call_count,
            "invalid_tool_call_count": r.invalid_tool_call_count,
            "trajectory_length": r.trajectory_length,
            "llm_call_count": r.llm_call_count,
            "error_count": r.error_count,
            "duration_sec": r.duration_sec,
        })
    return rows
