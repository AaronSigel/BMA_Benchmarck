import csv
from pathlib import Path

from benchmark.metrics.models import MetricsSummary, RunMetric
from benchmark.runner.models import RunResult

SUMMARY_CSV_COLUMNS = [
    "run_id",
    "task_id",
    "model",
    "strategy",
    "mcp_profile",
    "repetition",
    "execution_mode",
    "pass_type",
    "run_status",
    "agent_status",
    "scene_status",
    "score",
    "status",
    "total_score",
    "overall_status",
    "validation_coverage",
    "duration_sec",
    "tool_call_count",
    "unique_tool_count",
    "invalid_tool_call_count",
    "disabled_tool_call_count",
    "tool_error_count",
    "retry_count",
    "llm_call_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "provider_name",
    "provider_reported_prompt_tokens",
    "provider_reported_completion_tokens",
    "provider_reported_total_tokens",
    "provider_reported_cost_usd",
    "provider_cost_available",
    "issues",
    "validation_issues",
    "agent_issues",
    "tool_issues",
    "export_issues",
    "error_type",
    "error_source",
    "failure_stage",
    "error",
    "artifact_dir",
]

METRICS_CSV_COLUMNS = [
    "run_id",
    "task_id",
    "name",
    "value",
    "group",
    "source",
]


def write_summary_json(summary: MetricsSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def write_run_results_json(results: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "[" + ",".join(result.model_dump_json(indent=2) for result in results) + "]"
    path.write_text(payload, encoding="utf-8")


def write_summary_csv(results: list[RunResult], path: Path) -> None:
    _write_csv(
        [_run_summary_row(result) for result in results],
        SUMMARY_CSV_COLUMNS,
        path,
    )


def write_metrics_csv(metrics: list[RunMetric], path: Path) -> None:
    _write_csv(
        [metric.model_dump(mode="json") for metric in metrics],
        METRICS_CSV_COLUMNS,
        path,
    )


def _run_summary_row(result: RunResult) -> dict[str, object]:
    validation = result.summary.get("validation", {}) if isinstance(result.summary, dict) else {}
    execution = result.summary.get("execution", {}) if isinstance(result.summary, dict) else {}
    structured_error = result.summary.get("structured_error", {}) if isinstance(result.summary, dict) else {}
    agent_run = execution.get("agent_run", {}) if isinstance(execution, dict) else {}
    agent_summary = agent_run.get("summary", {}) if isinstance(agent_run, dict) else {}
    trace_metrics = agent_summary.get("metrics", {}) if isinstance(agent_summary, dict) else {}
    issues = validation.get("issues") or validation.get("issue_count") if isinstance(validation, dict) else None
    return {
        "run_id": result.run_id,
        "task_id": result.task_id,
        "model": result.summary.get("model") if isinstance(result.summary, dict) else None,
        "strategy": result.summary.get("strategy") if isinstance(result.summary, dict) else None,
        "mcp_profile": result.summary.get("mcp_profile") if isinstance(result.summary, dict) else None,
        "repetition": result.summary.get("repetition") if isinstance(result.summary, dict) else None,
        "execution_mode": result.execution_mode.value,
        "pass_type": _pass_type(result),
        "run_status": (result.run_status or result.status).value,
        "agent_status": result.agent_status.value if result.agent_status else None,
        "scene_status": result.scene_status.value if result.scene_status else None,
        "score": result.total_score,
        "status": result.status.value,
        "total_score": result.total_score,
        "overall_status": result.overall_status,
        "validation_coverage": validation.get("validation_coverage") if isinstance(validation, dict) else None,
        "duration_sec": result.duration_sec,
        "tool_call_count": (
            trace_metrics.get("tool_call_count")
            if isinstance(trace_metrics, dict)
            else agent_summary.get("tool_calls_count") if isinstance(agent_summary, dict) else None
        ),
        "unique_tool_count": trace_metrics.get("unique_tool_count") if isinstance(trace_metrics, dict) else None,
        "invalid_tool_call_count": trace_metrics.get("invalid_tool_call_count") if isinstance(trace_metrics, dict) else None,
        "disabled_tool_call_count": trace_metrics.get("disabled_tool_call_count") if isinstance(trace_metrics, dict) else None,
        "tool_error_count": trace_metrics.get("tool_error_count") if isinstance(trace_metrics, dict) else None,
        "retry_count": trace_metrics.get("retry_count") if isinstance(trace_metrics, dict) else None,
        "llm_call_count": trace_metrics.get("llm_call_count") if isinstance(trace_metrics, dict) else None,
        "prompt_tokens": trace_metrics.get("prompt_tokens") if isinstance(trace_metrics, dict) else None,
        "completion_tokens": trace_metrics.get("completion_tokens") if isinstance(trace_metrics, dict) else None,
        "total_tokens": trace_metrics.get("total_tokens") if isinstance(trace_metrics, dict) else None,
        "provider_name": trace_metrics.get("provider_name") if isinstance(trace_metrics, dict) else None,
        "provider_reported_prompt_tokens": trace_metrics.get("provider_reported_prompt_tokens") if isinstance(trace_metrics, dict) else None,
        "provider_reported_completion_tokens": trace_metrics.get("provider_reported_completion_tokens") if isinstance(trace_metrics, dict) else None,
        "provider_reported_total_tokens": trace_metrics.get("provider_reported_total_tokens") if isinstance(trace_metrics, dict) else None,
        "provider_reported_cost_usd": trace_metrics.get("provider_reported_cost_usd") if isinstance(trace_metrics, dict) else None,
        "provider_cost_available": trace_metrics.get("provider_cost_available") if isinstance(trace_metrics, dict) else None,
        "issues": issues if issues is not None else result.error,
        "validation_issues": _format_issue_counts(validation.get("issue_counts") if isinstance(validation, dict) else None),
        "agent_issues": _format_issue_counts(_agent_issue_counts(result, trace_metrics)),
        "tool_issues": _format_issue_counts(_tool_issue_counts(trace_metrics)),
        "export_issues": _format_issue_counts(_export_issue_counts(validation)),
        "error_type": structured_error.get("error_type") if isinstance(structured_error, dict) else None,
        "error_source": structured_error.get("source") if isinstance(structured_error, dict) else None,
        "failure_stage": structured_error.get("failure_stage") if isinstance(structured_error, dict) else None,
        "error": result.error or "",
        "artifact_dir": str(result.artifacts_dir),
    }


def _format_issue_counts(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return "null"
    return "; ".join(f"{key}:{value[key]}" for key in sorted(value))


def _pass_type(result: RunResult) -> str:
    run_status = result.run_status or result.status
    if run_status.value == "error" or result.scene_status is None or result.scene_status.value in {"not_available", "skipped"}:
        return "runtime_error"
    if run_status.value == "passed" and result.scene_status.value == "passed":
        validation = result.summary.get("validation", {}) if isinstance(result.summary, dict) else {}
        issue_counts = validation.get("issue_counts") if isinstance(validation, dict) else None
        return "soft_pass" if isinstance(issue_counts, dict) and bool(issue_counts) else "clean_pass"
    if result.scene_status.value == "failed":
        return "failed_validation"
    return "runtime_error"


_AGENT_STATUS_TO_ISSUE: dict[str, str] = {
    "max_steps_reached": "max_steps_reached",
    "invalid_response": "invalid_response",
    "tool_error": "tool_error",
    "runtime_error": "runtime_error",
    "repeated_action_detected": "repeated_action",
    "duplicate_object_detected": "duplicate_object",
}


def _agent_issue_counts(result: RunResult, trace_metrics: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    if result.agent_status is not None:
        status_val = result.agent_status.value if hasattr(result.agent_status, "value") else str(result.agent_status)
        issue_code = _AGENT_STATUS_TO_ISSUE.get(status_val)
        if issue_code:
            counts[issue_code] = 1
    elif result.error:
        counts["runtime_error"] = 1
    if isinstance(trace_metrics, dict):
        for key in ("repeated_action_count", "duplicate_object_count", "wasted_step_count", "no_progress_step_count"):
            value = trace_metrics.get(key)
            if isinstance(value, int) and value > 0:
                counts[key] = value
    return counts


def _tool_issue_counts(trace_metrics: object) -> dict[str, int]:
    if not isinstance(trace_metrics, dict):
        return {}
    counts: dict[str, int] = {}
    for key in ("invalid_tool_call_count", "disabled_tool_call_count", "tool_error_count"):
        value = trace_metrics.get(key)
        if isinstance(value, int) and value > 0:
            counts[key] = value
    # Map aggregated error-type counts surfaced from trace steps
    for key in ("invalid_json_count", "socket_error_count", "unknown_tool_count"):
        value = trace_metrics.get(key)
        if isinstance(value, int) and value > 0:
            counts[key] = value
    return counts


def _export_issue_counts(validation: object) -> dict[str, int]:
    if not isinstance(validation, dict):
        return {}
    issue_counts = validation.get("issue_counts")
    if not isinstance(issue_counts, dict):
        return {}
    return {
        key: int(value)
        for key, value in issue_counts.items()
        if isinstance(value, int) and ("export" in key or "glb" in key)
    }


def _write_csv(rows: list[dict[str, object]], fieldnames: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: _csv_value(value) for key, value in row.items()} for row in rows])


def _csv_value(value: object) -> object:
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return str(value)
    return value
