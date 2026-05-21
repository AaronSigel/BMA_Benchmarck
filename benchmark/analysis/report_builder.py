from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
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


def _provider_cost_total(runs: list[RunAnalysisResult]) -> float | str:
    costs = [
        float(r.metrics["provider_reported_cost_usd"])
        for r in runs
        if isinstance(r.metrics.get("provider_reported_cost_usd"), (int, float))
    ]
    if not costs:
        return "not_available"
    return round(sum(costs), 6)


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
    extra_metrics["validators_total"] = val_summary.validators_total
    extra_metrics["validators_run"] = val_summary.validators_run
    extra_metrics["validation_error_count"] = val_summary.validation_error_count
    extra_metrics["validation_warning_count"] = val_summary.validation_warning_count
    if val_summary.validation_coverage is not None:
        extra_metrics["validation_coverage"] = val_summary.validation_coverage
    extra_metrics["parameter_correctness"] = _parameter_correctness_from_validation(validation)

    for field in (
        "object_score", "transform_score", "material_score",
        "light_score", "camera_score", "export_score", "export_import_score",
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


def _parameter_correctness_from_validation(validation: SceneValidationResult | None) -> float | str:
    if validation is None:
        return "not_available"
    scores = [
        validator.score
        for validator in validation.validators
        if validator.name in {
            "transform_validator",
            "material_validator",
            "light_validator",
            "camera_validator",
        }
        and validator.status.value != "skipped"
    ]
    if not scores:
        return "not_available"
    return sum(scores) / len(scores)


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
    headers = ["Value", "Runs", "Success Rate", "Avg Score", "Avg Tool Calls", "Avg Duration (s)", "Avg Provider Cost", "Validation Failures"]
    rows = [
        [
            g.value,
            str(g.run_count),
            _na(round(g.success_rate, 3) if g.success_rate is not None else None),
            _na(round(g.avg_score, 4) if g.avg_score is not None else None),
            _na(round(g.avg_tool_calls, 1) if g.avg_tool_calls is not None else None),
            _na(round(g.avg_duration_sec, 1) if g.avg_duration_sec is not None else None),
            _na(round(g.avg_cost, 6) if g.avg_cost is not None else None),
            str(g.validation_failures),
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
    rows.extend(
        [
            ["models_used", _na(_join_list(metadata.get("models_used")))],
            ["strategies_used", _na(_join_list(metadata.get("strategies_used")))],
            ["mcp_profiles_used", _na(_join_list(metadata.get("mcp_profiles_used")))],
            ["tasks_used", _na(_join_list(metadata.get("tasks_used")))],
            ["repetitions", _na(metadata.get("repetitions"))],
            ["planned_runs", _na(metadata.get("planned_runs") or metadata.get("expected_runs"))],
            ["executed_runs", _na(metadata.get("executed_runs"))],
            ["artifact_count", _na(metadata.get("artifact_count"))],
            ["missing_artifacts", _na(metadata.get("missing_artifacts"))],
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


def _join_list(value: Any) -> str | None:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if value is None:
        return None
    return str(value)


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
    lines.append("## Matrix Size\n")
    planned = analysis.metadata.get("planned_runs") or analysis.metadata.get("expected_runs")
    skipped = 0
    profile_preflight = analysis.metadata.get("mcp_profile_preflight")
    if isinstance(profile_preflight, dict):
        skipped = int(analysis.metadata.get("skipped_by_preflight") or 0)
    lines.append(_md_table(["Metric", "Value"], [
        ["planned_runs", _na(planned or len(runs) + skipped)],
        ["executed_runs", str(len(runs))],
        ["skipped_runs", str(skipped)],
        ["skipped_by_preflight", str(skipped)],
        ["estimated_runtime", _na(analysis.metadata.get("estimated_runtime"))],
        ["actual_runtime", _na(round(sum((r.duration_sec or 0.0) for r in runs), 1))],
        ["total_provider_reported_cost_usd", _na(_provider_cost_total(runs))],
        ["runs_with_provider_cost", str(sum(1 for r in runs if r.metrics.get("provider_cost_available") is True))],
        ["runs_without_provider_cost", str(sum(1 for r in runs if r.metrics.get("provider_cost_available") is not True))],
    ]))

    # ------------------------------------------------------------------
    # 1. Summary
    # ------------------------------------------------------------------
    lines.append("## 1. Summary\n")
    summary_headers = ["Metric", "Value"]
    summary_rows = [
        ["Total runs", str(s.total_runs)],
        ["Successful (passed)", str(s.successful_runs)],
        ["  clean_pass", str(s.clean_pass_count)],
        ["  soft_pass", str(s.soft_pass_count)],
        ["Failed", str(s.failed_runs)],
        ["Errors", str(s.error_runs)],
        ["clean_pass_rate", _na(round(s.clean_pass_rate, 4) if s.clean_pass_rate is not None else None)],
        ["soft_pass_rate", _na(round(s.soft_pass_rate, 4) if s.soft_pass_rate is not None else None)],
        ["strict_success_rate (clean only)", _na(round(s.strict_success_rate, 4) if s.strict_success_rate is not None else None)],
        ["reported_success_rate (clean+soft)", _na(round(s.reported_success_rate, 4) if s.reported_success_rate is not None else None)],
        ["agent_completed_count", str(s.agent_completed_count)],
        ["agent_completed_after_scene_passed_count", str(s.agent_completed_after_scene_passed_count)],
        ["agent_incomplete_count", str(s.agent_incomplete_count)],
        ["Avg scene score", _na(round(s.average_scene_score, 4) if s.average_scene_score is not None else None)],
        ["Average score strict", _na(round(s.average_score_strict, 4) if s.average_score_strict is not None else None)],
        ["Average score passed only", _na(round(s.average_score_passed_only, 4) if s.average_score_passed_only is not None else None)],
        ["Scene success rate", _na(round(s.scene_success_rate, 4) if s.scene_success_rate is not None else None)],
        ["Run success rate", _na(round(s.run_success_rate, 4) if s.run_success_rate is not None else None)],
        ["Agent completion rate", _na(round(s.agent_completion_rate, 4) if s.agent_completion_rate is not None else None)],
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

        lines.append("## Compact Highlights\n")
        model_groups = group_by_model(runs).groups
        strategy_groups = group_by_strategy(runs).groups
        task_groups = group_by_task_category(runs).groups
        best_model_scene = max(model_groups, key=lambda g: g.success_rate or -1, default=None)
        best_model_strict = max(model_groups, key=lambda g: g.avg_score or -1, default=None)
        best_strategy_scene = max(strategy_groups, key=lambda g: g.success_rate or -1, default=None)
        best_strategy_cost = min(
            [g for g in strategy_groups if g.avg_cost is not None],
            key=lambda g: g.avg_cost or 0,
            default=None,
        )
        worst_task = min(task_groups, key=lambda g: g.success_rate if g.success_rate is not None else 2, default=None)
        failure_counter: Counter[str] = Counter()
        for r in runs:
            if r.run_status and r.run_status != "passed":
                failure_counter[r.run_status] += 1
            for issue in r.issues:
                code = issue.get("code")
                if code:
                    failure_counter[str(code)] += 1
        common_failure = failure_counter.most_common(1)[0][0] if failure_counter else None
        lines.append(_md_table(["Metric", "Value"], [
            ["best_model_by_scene_success", _na(best_model_scene.value if best_model_scene else None)],
            ["best_model_by_strict_success", _na(best_model_strict.value if best_model_strict else None)],
            ["best_strategy_by_scene_success", _na(best_strategy_scene.value if best_strategy_scene else None)],
            ["best_strategy_by_cost", _na(best_strategy_cost.value if best_strategy_cost else None)],
            ["worst_task_category", _na(worst_task.value if worst_task else None)],
            ["most_common_failure", _na(common_failure)],
        ]))

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
    # 7b. Top Issues
    # ------------------------------------------------------------------
    if config.include_error_taxonomy and runs:
        lines.append("## Top Issues\n")

        # Validation issues aggregated from all runs
        val_issue_counter: Counter[str] = Counter()
        agent_issue_counter: Counter[str] = Counter()
        tool_issue_counter: Counter[str] = Counter()
        export_issue_counter: Counter[str] = Counter()
        for r in runs:
            for issue in r.issues:
                code = issue.get("code")
                if code:
                    val_issue_counter[str(code)] += 1
            for key, val in r.metrics.items():
                if key.startswith("error.") and isinstance(val, int) and val > 0:
                    agent_issue_counter[key[len("error."):]] += val

        if val_issue_counter:
            lines.append("### Top Validation Issues\n")
            lines.append(_md_table(
                ["Issue Code", "Count"],
                [[code, str(cnt)] for code, cnt in val_issue_counter.most_common(10)],
            ))

        # ReAct-specific agent metrics
        react_issue_counter: Counter[str] = Counter()
        for r in runs:
            if r.strategy == "react":
                for key in ("repeated_action_count", "duplicate_object_count", "wasted_step_count", "no_progress_step_count"):
                    v = r.metrics.get(key)
                    if isinstance(v, int) and v > 0:
                        react_issue_counter[key] += v
                if r.agent_status and r.agent_status not in ("completed", "completed_after_scene_passed"):
                    agent_issue_counter[f"react_{r.agent_status}"] += 1

        if agent_issue_counter:
            lines.append("### Top Agent Issues\n")
            lines.append(_md_table(
                ["Issue Code", "Count"],
                [[code, str(cnt)] for code, cnt in agent_issue_counter.most_common(10)],
            ))

        # Tool issues from trace metrics
        for r in runs:
            for key in ("invalid_tool_call_count", "disabled_tool_call_count", "tool_error_count"):
                v = r.metrics.get(key)
                if isinstance(v, int) and v > 0:
                    tool_issue_counter[key] += v

        if tool_issue_counter:
            lines.append("### Top Tool Issues\n")
            lines.append(_md_table(
                ["Issue Code", "Count"],
                [[code, str(cnt)] for code, cnt in tool_issue_counter.most_common(10)],
            ))

        # Export issues (validation issues with export/glb codes)
        for r in runs:
            for issue in r.issues:
                code = issue.get("code", "")
                if "export" in code or "glb" in code:
                    export_issue_counter[str(code)] += 1

        if export_issue_counter:
            lines.append("### Top Export Issues\n")
            lines.append(_md_table(
                ["Issue Code", "Count"],
                [[code, str(cnt)] for code, cnt in export_issue_counter.most_common(10)],
            ))

        # Top issues by strategy
        strategy_issue_map: dict[str, Counter[str]] = {}
        for r in runs:
            strat = r.strategy or "unknown"
            if strat not in strategy_issue_map:
                strategy_issue_map[strat] = Counter()
            for issue in r.issues:
                code = issue.get("code")
                if code:
                    strategy_issue_map[strat][str(code)] += 1
        if strategy_issue_map:
            lines.append("### Top Issues by Strategy\n")
            for strat, counter in sorted(strategy_issue_map.items()):
                if counter:
                    top = counter.most_common(5)
                    lines.append(f"**{strat}**: " + ", ".join(f"{c}:{n}" for c, n in top) + "\n")

        # Top issues by MCP profile
        profile_issue_map: dict[str, Counter[str]] = {}
        for r in runs:
            prof = r.mcp_profile or "unknown"
            if prof not in profile_issue_map:
                profile_issue_map[prof] = Counter()
            for issue in r.issues:
                code = issue.get("code")
                if code:
                    profile_issue_map[prof][str(code)] += 1
        if profile_issue_map:
            lines.append("### Top Issues by MCP Profile\n")
            for prof, counter in sorted(profile_issue_map.items()):
                if counter:
                    top = counter.most_common(5)
                    lines.append(f"**{prof}**: " + ", ".join(f"{c}:{n}" for c, n in top) + "\n")

        # Top issues by task
        task_issue_map: dict[str, Counter[str]] = {}
        for r in runs:
            tid = r.task_id or "unknown"
            if tid not in task_issue_map:
                task_issue_map[tid] = Counter()
            for issue in r.issues:
                code = issue.get("code")
                if code:
                    task_issue_map[tid][str(code)] += 1
        if any(c for c in task_issue_map.values()):
            lines.append("### Top Issues by Task\n")
            task_rows = []
            for tid, counter in sorted(task_issue_map.items()):
                top = counter.most_common(3)
                task_rows.append([tid, ", ".join(f"{c}:{n}" for c, n in top) if top else "—"])
            lines.append(_md_table(["Task", "Top Issues"], task_rows))

        # Top soft-pass issues (issues present in passed runs)
        soft_pass_issue_counter: Counter[str] = Counter()
        for r in runs:
            if r.pass_type == "soft_pass":
                for issue in r.issues:
                    code = issue.get("code")
                    if code:
                        soft_pass_issue_counter[str(code)] += 1
        if soft_pass_issue_counter:
            lines.append("### Top Soft-Pass Issues\n")
            lines.append(_md_table(
                ["Issue Code", "Count"],
                [[code, str(cnt)] for code, cnt in soft_pass_issue_counter.most_common(10)],
            ))

    # ------------------------------------------------------------------
    # 7c. ReAct Diagnostics
    # ------------------------------------------------------------------
    if config.include_error_taxonomy and runs:
        react_runs = [r for r in runs if r.strategy == "react"]
        if react_runs:
            lines.append("## ReAct Diagnostics\n")
            react_passed = sum(1 for r in react_runs if r.success)
            react_max_steps = sum(
                1 for r in react_runs
                if r.agent_status == "max_steps_reached"
            )
            react_repeated = sum(
                1 for r in react_runs
                if r.agent_status in ("repeated_action_detected",)
            )
            react_duplicate = sum(
                1 for r in react_runs
                if r.agent_status in ("duplicate_object_detected",)
            )
            react_no_progress = sum(
                int(r.metrics.get("no_progress_step_count") or 0)
                for r in react_runs
            )
            react_completed_early = sum(
                1 for r in react_runs
                if r.agent_status == "completed_after_scene_passed"
            )
            react_steps = [
                r.metrics.get("tool_call_count")
                for r in react_runs
                if isinstance(r.metrics.get("tool_call_count"), (int, float))
            ]
            avg_steps = round(sum(react_steps) / len(react_steps), 1) if react_steps else None
            react_no_progress_runs = sum(
                1 for r in react_runs if r.agent_status == "no_progress_detected"
            )
            react_repeated_actions_total = sum(
                int(r.metrics.get("repeated_action_count") or 0) for r in react_runs
            )
            react_duplicate_objs_total = sum(
                int(r.metrics.get("duplicate_object_count") or 0) for r in react_runs
            )
            react_repair_attempts = sum(
                int(r.metrics.get("repair_attempt_count") or 0) for r in react_runs
            )
            react_clean_pass = sum(1 for r in react_runs if r.pass_type == "clean_pass")
            react_soft_pass = sum(1 for r in react_runs if r.pass_type == "soft_pass")
            lines.append(_md_table(["Metric", "Value"], [
                ["react_runs", str(len(react_runs))],
                ["react_passed", str(react_passed)],
                ["react_clean_pass", str(react_clean_pass)],
                ["react_soft_pass", str(react_soft_pass)],
                ["react_success_rate", _na(round(react_passed / len(react_runs), 3) if react_runs else None)],
                ["react_max_steps_reached", str(react_max_steps)],
                ["react_repeated_action_runs", str(react_repeated)],
                ["react_repeated_actions_total", str(react_repeated_actions_total)],
                ["react_duplicate_object_runs", str(react_duplicate)],
                ["react_duplicate_objects_total", str(react_duplicate_objs_total)],
                ["react_no_progress_runs", str(react_no_progress_runs)],
                ["react_no_progress_steps_total", str(react_no_progress)],
                ["react_repair_attempts", str(react_repair_attempts)],
                ["react_completed_after_scene_passed", str(react_completed_early)],
                ["react_average_tool_calls", _na(avg_steps)],
            ]))

    # ------------------------------------------------------------------
    # 7d. Export Diagnostics
    # ------------------------------------------------------------------
    if config.include_error_taxonomy and runs:
        export_runs = [
            r for r in runs
            if r.task_id and "export" in r.task_id.lower()
        ]
        if export_runs:
            lines.append("## Export Diagnostics\n")
            export_passed = sum(1 for r in export_runs if r.success)
            export_clean = sum(1 for r in export_runs if r.pass_type == "clean_pass")
            export_soft = sum(1 for r in export_runs if r.pass_type == "soft_pass")
            export_failed = sum(1 for r in export_runs if r.pass_type == "failed")
            export_error = sum(1 for r in export_runs if r.pass_type == "error")

            def _count_issue(code: str) -> int:
                return sum(1 for r in export_runs if any(i.get("code") == code for i in r.issues))

            export_score_vals = [
                r.metrics.get("export_score")
                for r in export_runs
                if isinstance(r.metrics.get("export_score"), (int, float))
            ]
            import_score_vals = [
                r.metrics.get("export_import_score")
                for r in export_runs
                if isinstance(r.metrics.get("export_import_score"), (int, float))
            ]
            avg_export_score = round(sum(export_score_vals) / len(export_score_vals), 3) if export_score_vals else None
            avg_import_score = round(sum(import_score_vals) / len(import_score_vals), 3) if import_score_vals else None
            lines.append(_md_table(["Metric", "Value"], [
                ["export_runs", str(len(export_runs))],
                ["export_clean_pass", str(export_clean)],
                ["export_soft_pass", str(export_soft)],
                ["export_failed", str(export_failed)],
                ["export_error", str(export_error)],
                ["export_success_rate", _na(round(export_passed / len(export_runs), 3) if export_runs else None)],
                ["export_missing", str(_count_issue("export_missing"))],
                ["export_file_invalid", str(_count_issue("export_file_invalid"))],
                ["export_import_missing", str(_count_issue("export_import_missing"))],
                ["export_import_material_missing", str(_count_issue("export_import_material_missing"))],
                ["export_import_transform_mismatch", str(_count_issue("export_import_transform_mismatch"))],
                ["material_missing", str(_count_issue("material_missing"))],
                ["object_missing", str(_count_issue("object_missing"))],
                ["avg_export_score", _na(avg_export_score)],
                ["avg_import_back_score", _na(avg_import_score)],
            ]))

    # ------------------------------------------------------------------
    # 7e. Lighting Diagnostics
    # ------------------------------------------------------------------
    if config.include_error_taxonomy and runs:
        lighting_runs = [
            r for r in runs
            if r.task_id and "lighting" in r.task_id.lower()
        ]
        if lighting_runs:
            lines.append("## Lighting Diagnostics\n")
            lighting_passed = sum(1 for r in lighting_runs if r.success)
            lighting_clean = sum(1 for r in lighting_runs if r.pass_type == "clean_pass")
            lighting_soft = sum(1 for r in lighting_runs if r.pass_type == "soft_pass")
            lighting_failed = sum(1 for r in lighting_runs if r.pass_type == "failed")
            lighting_error = sum(1 for r in lighting_runs if r.pass_type == "error")

            def _lcount(code: str) -> int:
                return sum(1 for r in lighting_runs if any(i.get("code") == code for i in r.issues))

            light_score_vals = [
                r.metrics.get("light_score")
                for r in lighting_runs
                if isinstance(r.metrics.get("light_score"), (int, float))
            ]
            avg_light_score = round(sum(light_score_vals) / len(light_score_vals), 3) if light_score_vals else None
            lines.append(_md_table(["Metric", "Value"], [
                ["lighting_runs", str(len(lighting_runs))],
                ["lighting_clean_pass", str(lighting_clean)],
                ["lighting_soft_pass", str(lighting_soft)],
                ["lighting_failed", str(lighting_failed)],
                ["lighting_error", str(lighting_error)],
                ["lighting_success_rate", _na(round(lighting_passed / len(lighting_runs), 3) if lighting_runs else None)],
                ["light_missing", str(_lcount("light_missing"))],
                ["light_rotation_mismatch", str(_lcount("light_rotation_mismatch"))],
                ["light_direction_mismatch", str(_lcount("light_direction_mismatch"))],
                ["light_energy_mismatch", str(_lcount("light_energy_mismatch"))],
                ["light_type_mismatch", str(_lcount("light_type_mismatch"))],
                ["light_location_mismatch", str(_lcount("light_location_mismatch"))],
                ["avg_light_score", _na(avg_light_score)],
            ]))

    # ------------------------------------------------------------------
    # 8. Run details (gated by config.include_runs)
    # ------------------------------------------------------------------
    if config.include_runs and runs:
        lines.append("## 8. Run Details\n")
        det_headers = [
            "Run ID", "Task", "Strategy", "Model", "MCP Profile", "Pass Type", "Score",
            "Validation", "Coverage", "Validators Run", "Skipped", "Tool Calls",
            "Invalid", "Disabled", "Retries", "LLM Calls", "Tokens", "Provider Cost",
            "Duration (s)", "Errors",
        ]
        det_rows = [
            [
                r.run_id,
                r.task_id,
                r.strategy,
                _na(r.model),
                _na(r.mcp_profile),
                _na(r.pass_type),
                _na(round(r.total_score, 4) if r.total_score is not None else None),
                _na(r.validation_status),
                _na(round(float(r.metrics["validation_coverage"]), 3) if isinstance(r.metrics.get("validation_coverage"), (int, float)) else None),
                _na(r.metrics.get("validators_run")),
                _na(r.metrics.get("skipped_validator_count")),
                str(r.tool_call_count),
                str(r.invalid_tool_call_count),
                _na(r.metrics.get("disabled_tool_call_count")),
                str(r.retry_count),
                str(r.llm_call_count),
                _na(r.metrics.get("total_tokens")),
                _na(r.metrics.get("provider_reported_cost_usd") if r.metrics.get("provider_cost_available") is True else "not_available"),
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
                    lines.append(f"- [{art}]({_artifact_href(art, config.output_dir)})")
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
<ul>{% for label, href in arts %}<li><a href="{{ href }}">{{ label }}</a></li>{% endfor %}</ul>
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

    run_detail_headers = [
        "Run ID", "Task", "Strategy", "Model", "MCP Profile", "Success", "Score",
        "Validation", "Coverage", "Validators Run", "Skipped", "Tool Calls",
        "Invalid", "Disabled", "Retries", "LLM Calls", "Tokens", "Provider Cost",
        "Duration (s)", "Errors",
    ]
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
                _html_na(r.validation_status),
                _html_na(round(float(r.metrics["validation_coverage"]), 3) if isinstance(r.metrics.get("validation_coverage"), (int, float)) else None),
                _html_na(r.metrics.get("validators_run")),
                _html_na(r.metrics.get("skipped_validator_count")),
                str(r.tool_call_count),
                str(r.invalid_tool_call_count),
                _html_na(r.metrics.get("disabled_tool_call_count")),
                str(r.retry_count),
                str(r.llm_call_count),
                _html_na(r.metrics.get("total_tokens")),
                _html_na(r.metrics.get("provider_reported_cost_usd") if r.metrics.get("provider_cost_available") is True else "not_available"),
                _html_na(round(r.duration_sec, 1) if r.duration_sec is not None else None),
                str(r.error_count),
            ])

    artifact_sections: list[tuple[str, list[tuple[str, str]]]] = []
    if config.include_artifact_links:
        for r in runs:
            if r.artifacts:
                artifact_sections.append(
                    (
                        r.run_id,
                        [(art, _artifact_href(art, config.output_dir)) for art in r.artifacts],
                    )
                )

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
            "validators_total": r.metrics.get("validators_total", ""),
            "validators_run": r.metrics.get("validators_run", ""),
            "validators_skipped": r.metrics.get("skipped_validator_count", ""),
            "validators_passed": r.metrics.get("passed_validator_count", ""),
            "validators_failed": r.metrics.get("failed_validator_count", ""),
            "validation_coverage": r.metrics.get("validation_coverage", ""),
            "tool_call_count": r.tool_call_count,
            "invalid_tool_call_count": r.invalid_tool_call_count,
            "disabled_tool_call_count": r.metrics.get("disabled_tool_call_count", ""),
            "retry_count": r.retry_count,
            "trajectory_length": r.trajectory_length,
            "llm_call_count": r.llm_call_count,
            "prompt_tokens": r.metrics.get("prompt_tokens", ""),
            "completion_tokens": r.metrics.get("completion_tokens", ""),
            "total_tokens": r.metrics.get("total_tokens", ""),
            "provider_reported_cost_usd": r.metrics.get("provider_reported_cost_usd", ""),
            "provider_cost_available": r.metrics.get("provider_cost_available", ""),
            "error_count": r.error_count,
            "duration_sec": r.duration_sec,
        })
    return rows


def _artifact_href(artifact_path: str, report_output_dir: Path | str) -> str:
    path = Path(artifact_path)
    if _has_uri_scheme(artifact_path):
        return artifact_path
    output_dir = Path(report_output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    try:
        if path.is_absolute():
            rel = os.path.relpath(path, output_dir)
        else:
            rel = os.path.relpath(Path.cwd() / path, output_dir)
    except ValueError:
        rel = str(path)
    return Path(rel).as_posix()


def _has_uri_scheme(value: str) -> bool:
    return "://" in value or value.startswith("mailto:")


# ---------------------------------------------------------------------------
# Report-ready MVP report builders. These definitions intentionally override
# the legacy builders above while keeping their public function names.
# ---------------------------------------------------------------------------

def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _num(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _category(result: RunAnalysisResult) -> str:
    explicit = result.metrics.get("task_category") or result.metrics.get("run_summary.task_category")
    if isinstance(explicit, str) and explicit:
        return explicit
    task_id = result.task_id.lower()
    for category in ("geometry", "materials", "lighting", "camera", "export"):
        if category in task_id:
            return category
    return "unknown"


def _repetition(result: RunAnalysisResult) -> str:
    value = result.metrics.get("run_summary.repetition", result.metrics.get("repetition", ""))
    return str(value) if value not in (None, "") else ""


def _status_counts(runs: list[RunAnalysisResult]) -> dict[str, int]:
    return {
        "clean_pass": sum(1 for r in runs if _effective_pass_type(r) == "clean_pass"),
        "soft_pass": sum(1 for r in runs if _effective_pass_type(r) == "soft_pass"),
        "failed_validation": sum(1 for r in runs if _effective_pass_type(r) == "failed_validation"),
        "runtime_error": sum(1 for r in runs if _effective_pass_type(r) == "runtime_error"),
    }


def _effective_pass_type(result: RunAnalysisResult) -> str:
    if result.pass_type in {"clean_pass", "soft_pass", "failed_validation", "runtime_error"}:
        return str(result.pass_type)
    if result.pass_type == "failed":
        return "failed_validation"
    if result.pass_type == "error":
        return "runtime_error"
    if result.success is True:
        return "clean_pass"
    if result.success is False:
        return "failed_validation"
    return "runtime_error"


def _cost_total(runs: list[RunAnalysisResult]) -> float:
    return sum(
        float(r.metrics["provider_reported_cost_usd"])
        for r in runs
        if isinstance(r.metrics.get("provider_reported_cost_usd"), (int, float))
    )


def _avg_score(runs: list[RunAnalysisResult]) -> float | None:
    scores = [r.total_score for r in runs if r.total_score is not None]
    return sum(scores) / len(scores) if scores else None


def _avg_duration(runs: list[RunAnalysisResult]) -> float | None:
    values = [r.duration_sec for r in runs if r.duration_sec is not None]
    return sum(values) / len(values) if values else None


def _issue_counts(runs: list[RunAnalysisResult], kind: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    if kind in {"validation", "export"}:
        for r in runs:
            for issue in r.issues:
                code = issue.get("code") if isinstance(issue, dict) else None
                if not isinstance(code, str):
                    continue
                if kind == "export" and "export" not in code and "glb" not in code:
                    continue
                counts[code] += 1
    elif kind == "agent":
        for r in runs:
            for key in ("repeated_action_count", "duplicate_object_count", "wasted_step_count", "no_progress_step_count"):
                value = r.metrics.get(key)
                if isinstance(value, int) and value > 0:
                    counts[key] += value
            if r.agent_status and r.agent_status not in {"completed", "completed_after_scene_passed"}:
                counts[r.agent_status] += 1
            if r.pass_type == "runtime_error" and not r.agent_status:
                counts["runtime_error"] += 1
    elif kind == "tool":
        for r in runs:
            for key in ("invalid_tool_call_count", "disabled_tool_call_count", "tool_error_count"):
                value = r.metrics.get(key)
                if isinstance(value, int) and value > 0:
                    counts[key] += value
    return counts


def _group_rows(runs: list[RunAnalysisResult], key_fn) -> list[tuple[str, list[RunAnalysisResult]]]:
    grouped: dict[str, list[RunAnalysisResult]] = {}
    for run in runs:
        grouped.setdefault(str(key_fn(run)), []).append(run)
    return sorted(grouped.items(), key=lambda item: item[0])


def _strategy_table(runs: list[RunAnalysisResult]) -> list[list[str]]:
    rows: list[list[str]] = []
    for strategy, items in _group_rows(runs, lambda r: r.strategy):
        c = _status_counts(items)
        rows.append([
            strategy,
            str(len(items)),
            str(c["clean_pass"]),
            str(c["soft_pass"]),
            str(c["failed_validation"]),
            str(c["runtime_error"]),
            _pct((c["clean_pass"] + c["soft_pass"]) / len(items) if items else None),
            _num(_avg_score(items)),
            _num(_avg_duration(items), 1),
            _num(_cost_total(items), 6),
        ])
    return rows


def _profile_table(runs: list[RunAnalysisResult]) -> list[list[str]]:
    rows: list[list[str]] = []
    for profile, items in _group_rows(runs, lambda r: r.mcp_profile or "unknown"):
        c = _status_counts(items)
        rows.append([
            profile,
            str(len(items)),
            str(c["clean_pass"]),
            str(c["soft_pass"]),
            str(c["failed_validation"]),
            str(c["runtime_error"]),
            _pct((c["clean_pass"] + c["soft_pass"]) / len(items) if items else None),
            _num(_avg_score(items)),
        ])
    return rows


def _task_table(runs: list[RunAnalysisResult]) -> list[list[str]]:
    rows: list[list[str]] = []
    for task_id, items in _group_rows(runs, lambda r: r.task_id):
        c = _status_counts(items)
        issues = _issue_counts(items, "validation").most_common(3)
        rows.append([
            task_id,
            _category(items[0]),
            str(len(items)),
            str(c["clean_pass"]),
            str(c["soft_pass"]),
            str(c["failed_validation"]),
            str(c["runtime_error"]),
            _pct((c["clean_pass"] + c["soft_pass"]) / len(items) if items else None),
            ", ".join(f"{code}:{count}" for code, count in issues) or "none",
        ])
    return rows


def _category_table(runs: list[RunAnalysisResult]) -> list[list[str]]:
    rows: list[list[str]] = []
    for category, items in _group_rows(runs, _category):
        c = _status_counts(items)
        rows.append([
            category,
            str(len(items)),
            str(c["clean_pass"]),
            str(c["soft_pass"]),
            str(c["failed_validation"]),
            str(c["runtime_error"]),
            _pct((c["clean_pass"] + c["soft_pass"]) / len(items) if items else None),
            _num(_avg_score(items)),
        ])
    return rows


def _strategy_profile_table(runs: list[RunAnalysisResult]) -> list[list[str]]:
    rows: list[list[str]] = []
    for key, items in _group_rows(runs, lambda r: f"{r.strategy} x {r.mcp_profile or 'unknown'}"):
        strategy, profile = key.split(" x ", 1)
        c = _status_counts(items)
        rows.append([
            strategy,
            profile,
            str(len(items)),
            str(c["clean_pass"]),
            str(c["soft_pass"]),
            str(c["failed_validation"]),
            str(c["runtime_error"]),
            _pct((c["clean_pass"] + c["soft_pass"]) / len(items) if items else None),
            _num(_avg_score(items)),
        ])
    return rows


def _issue_table(runs: list[RunAnalysisResult], kind: str) -> list[list[str]]:
    return [[code, str(count)] for code, count in _issue_counts(runs, kind).most_common(10)]


def _best_strategy(runs: list[RunAnalysisResult]) -> tuple[str, int, int] | None:
    best: tuple[str, int, int] | None = None
    for strategy, items in _group_rows(runs, lambda r: r.strategy):
        c = _status_counts(items)
        passed = c["clean_pass"] + c["soft_pass"]
        candidate = (strategy, passed, len(items))
        if best is None or (candidate[1] / candidate[2] if candidate[2] else 0) > (best[1] / best[2] if best[2] else 0):
            best = candidate
    return best


def _worst_categories(runs: list[RunAnalysisResult]) -> str:
    ranked: list[tuple[str, float]] = []
    for category, items in _group_rows(runs, _category):
        c = _status_counts(items)
        rate = (c["clean_pass"] + c["soft_pass"]) / len(items) if items else 0.0
        ranked.append((category, rate))
    return ", ".join(name for name, _ in sorted(ranked, key=lambda item: item[1])[:2]) or "N/A"


def build_key_findings(analysis: ExperimentAnalysisResult) -> list[str]:
    runs = analysis.runs
    s = analysis.summary
    best = _best_strategy(runs)
    val_issue = _issue_counts(runs, "validation").most_common(1)
    tool_issue = _issue_counts(runs, "tool").most_common(1)
    findings = [
        f"Всего выполнено {s.total_runs} запусков; доля чисто успешных запусков составила {_pct(s.strict_success_rate)}.",
        f"Отчётная доля успешных запусков с учётом soft pass составила {_pct(s.reported_success_rate)}.",
    ]
    if best:
        findings.append(f"Наиболее устойчивой стратегией стала {best[0]}: {best[1]} успешных запусков из {best[2]}.")
    react = [r for r in runs if r.strategy == "react"]
    if react:
        c = _status_counts(react)
        findings.append(f"ReAct используется как диагностическая стратегия; runtime_error для неё: {c['runtime_error']} из {len(react)}.")
    findings.append(f"Наиболее проблемные категории задач: {_worst_categories(runs)}.")
    if val_issue:
        findings.append(f"Основной validation issue: {val_issue[0][0]} ({val_issue[0][1]} случаев).")
    if tool_issue:
        findings.append(f"Основной tool issue: {tool_issue[0][0]} ({tool_issue[0][1]} случаев).")
    findings.append(f"Фактическая стоимость OpenRouter по provider-reported данным: {_num(_cost_total(runs), 6)} USD.")
    return findings[:8]


def _overall_rows(analysis: ExperimentAnalysisResult) -> list[list[str]]:
    runs = analysis.runs
    s = analysis.summary
    models = sorted({r.model for r in runs if r.model})
    return [
        ["matrix_name", analysis.experiment_id],
        ["model", ", ".join(models) or "N/A"],
        ["total_runs", str(s.total_runs)],
        ["clean_pass", str(s.clean_pass_count)],
        ["soft_pass", str(s.soft_pass_count)],
        ["failed_validation", str(s.failed_validation_count or s.failed_count)],
        ["runtime_error", str(s.runtime_error_count or s.error_count)],
        ["reported_success_rate", _pct(s.reported_success_rate)],
        ["strict_success_rate", _pct(s.strict_success_rate)],
        ["avg_score_completed", _num(s.average_score_completed)],
        ["avg_score_strict", _num(s.average_score_strict)],
        ["total_duration_sec", _num(sum((r.duration_sec or 0.0) for r in runs), 1)],
        ["total_provider_reported_cost_usd", _num(_cost_total(runs), 6)],
    ]


def build_markdown_report(analysis: ExperimentAnalysisResult, config: ReportConfig) -> str:
    runs = analysis.runs
    lines = [f"# {config.title}", "", f"**Experiment:** `{analysis.experiment_id}`", ""]
    lines.extend(["## Key findings", ""])
    for idx, finding in enumerate(build_key_findings(analysis), start=1):
        lines.append(f"{idx}. {finding}")
    lines.append("")
    lines.extend(["## 1. Overall Summary", "", _md_table(["metric", "value"], _overall_rows(analysis))])
    lines.extend(["## 2. Results by Strategy", "", _md_table(
        ["strategy", "runs", "clean_pass", "soft_pass", "failed_validation", "runtime_error", "reported_success_rate", "avg_score", "avg_duration_sec", "provider_reported_cost_usd"],
        _strategy_table(runs),
    )])
    lines.extend(["## 3. Results by MCP Profile", "", _md_table(
        ["mcp_profile", "runs", "clean_pass", "soft_pass", "failed_validation", "runtime_error", "reported_success_rate", "avg_score"],
        _profile_table(runs),
    )])
    lines.extend(["## 4. Results by Task", "", _md_table(
        ["task_id", "task_category", "runs", "clean_pass", "soft_pass", "failed_validation", "runtime_error", "reported_success_rate", "top_issues"],
        _task_table(runs),
    )])
    lines.extend(["## 5. Results by Task Category", "", _md_table(
        ["task_category", "runs", "clean_pass", "soft_pass", "failed_validation", "runtime_error", "reported_success_rate", "avg_score"],
        _category_table(runs),
    )])
    lines.extend(["## 6. Strategy x MCP Profile", "", _md_table(
        ["strategy", "mcp_profile", "runs", "clean_pass", "soft_pass", "failed_validation", "runtime_error", "reported_success_rate", "avg_score"],
        _strategy_profile_table(runs),
    )])
    for title, kind in (
        ("Top Validation Issues", "validation"),
        ("Top Agent Issues", "agent"),
        ("Top Tool Issues", "tool"),
        ("Top Export Issues", "export"),
    ):
        rows = _issue_table(runs, kind)
        lines.extend([f"## {title}", "", _md_table(["issue_code", "count"], rows) if rows else "_No issues recorded._\n"])
    lines.extend(["## OpenRouter Cost Summary", "", _md_table(["metric", "value"], [
        ["total_provider_reported_cost_usd", _num(_cost_total(runs), 6)],
        ["runs_with_provider_cost", str(sum(1 for r in runs if r.metrics.get("provider_cost_available") is True))],
        ["runs_without_provider_cost", str(sum(1 for r in runs if r.metrics.get("provider_cost_available") is not True))],
    ])])
    figures = [
        "figures/success_by_strategy.png",
        "figures/success_by_profile.png",
        "figures/success_by_category.png",
        "figures/top_validation_issues.png",
        "figures/cost_by_strategy.png",
        "figures/score_by_strategy.png",
        "figures/error_breakdown.png",
    ]
    lines.extend(["## Figures", ""])
    lines.extend(f"- `{path}`" for path in figures)
    lines.append("")
    lines.extend(_legacy_compat_sections(analysis, config))
    return "\n".join(lines) + "\n"


def _legacy_compat_sections(analysis: ExperimentAnalysisResult, config: ReportConfig) -> list[str]:
    runs = analysis.runs
    out: list[str] = []
    freshness_rows = _freshness_rows(analysis.metadata)
    if freshness_rows:
        out.extend(["## Artifact Freshness", "", _md_table(["Field", "Value"], freshness_rows)])
    out.extend(["## 1. Summary", "", _md_table(["Metric", "Value"], [
        ["Total runs", str(analysis.summary.total_runs)],
        ["Successful (passed)", str(analysis.summary.successful_runs)],
        ["Failed", str(analysis.summary.failed_validation_count or analysis.summary.failed_count)],
        ["Errors", str(analysis.summary.runtime_error_count or analysis.summary.error_count)],
        ["Avg scene score", _num(analysis.summary.average_scene_score)],
    ])])
    out.extend(["## 2. Best / Worst Runs", ""])
    if runs:
        top = sorted(runs, key=lambda r: (r.total_score is None, -(r.total_score or 0.0)))[:3]
        bottom = sorted(runs, key=lambda r: (r.total_score is None, r.total_score or 0.0))[:3]
        headers = ["Run ID", "Task", "Strategy", "Score", "Duration (s)", "Tool Calls"]
        out.extend(["### Top runs", "", _md_table(headers, [[r.run_id, r.task_id, r.strategy, _num(r.total_score), _num(r.duration_sec, 1), str(r.tool_call_count)] for r in top])])
        out.extend(["### Bottom runs", "", _md_table(headers, [[r.run_id, r.task_id, r.strategy, _num(r.total_score), _num(r.duration_sec, 1), str(r.tool_call_count)] for r in bottom])])
    else:
        out.append("_No runs._\n")
    if config.include_group_comparison:
        out.extend(["## 3. Strategy Comparison", "", _md_table(
            ["Value", "Runs", "Success Rate", "Avg Score"],
            [[row[0], row[1], row[6], row[7]] for row in _strategy_table(runs)],
        )])
    if config.include_error_taxonomy:
        errors: Counter[str] = Counter()
        for r in runs:
            for key, value in r.metrics.items():
                if key.startswith("error.") and isinstance(value, int) and value > 0:
                    errors[key[len("error."):]] += value
        out.extend(["## 7. Error Taxonomy", "", _md_table(["Category", "Count"], [[k, str(v)] for k, v in errors.most_common()]) if errors else "_No errors recorded._\n"])
    if config.include_runs:
        out.extend(["## 8. Run Details", "", _md_table(
            ["Run ID", "Task", "Strategy", "Model", "MCP Profile", "Pass Type", "Success", "Score", "LLM Calls", "Duration (s)"],
            [[
                r.run_id,
                r.task_id,
                r.strategy,
                _num(r.model),
                _num(r.mcp_profile),
                _effective_pass_type(r),
                str(r.success),
                _num(r.total_score),
                str(r.llm_call_count),
                _num(r.duration_sec, 1),
            ] for r in runs],
        )])
    if config.include_artifact_links:
        out.extend(["## 9. Artifact Links", ""])
        if any(r.artifacts for r in runs):
            for r in runs:
                if r.artifacts:
                    out.append(f"**{r.run_id}**")
                    out.extend(f"- [{art}]({_artifact_href(art, config.output_dir)})" for art in r.artifacts)
                    out.append("")
        else:
            out.append("_No artifacts recorded._\n")
    return out


def build_html_report(analysis: ExperimentAnalysisResult, config: ReportConfig) -> str:
    import html

    md = build_markdown_report(analysis, config)
    body: list[str] = []
    in_list = False
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            if in_list:
                body.append("</ol>")
                in_list = False
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            if in_list:
                body.append("</ol>")
                in_list = False
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("| "):
            if in_list:
                body.append("</ol>")
                in_list = False
            table_lines: list[str] = []
            while i < len(lines) and lines[i].startswith("| "):
                table_lines.append(lines[i])
                i += 1
            body.append(_html_table_from_md(table_lines))
            continue
        elif line[:3].isdigit() and ". " in line[:4]:
            if not in_list:
                body.append("<ol>")
                in_list = True
            body.append(f"<li>{html.escape(line.split('. ', 1)[1])}</li>")
        elif line.startswith("- "):
            if in_list:
                body.append("</ol>")
                in_list = False
            raw = line[2:].strip()
            if raw.startswith("[") and "](" in raw and raw.endswith(")"):
                label, href = raw[1:].split("](", 1)
                body.append(f'<p><a href="{html.escape(href[:-1])}">{html.escape(label)}</a></p>')
            else:
                value = html.escape(raw.strip("`"))
                body.append(f"<p>{value}</p>")
        elif line.strip():
            if in_list:
                body.append("</ol>")
                in_list = False
            body.append(f"<p>{html.escape(line)}</p>")
        i += 1
    if in_list:
        body.append("</ol>")
    for run in analysis.runs:
        if run.success is True:
            body.append('<span class="pass">pass</span>')
        elif run.success is False:
            body.append('<span class="fail">fail</span>')
        else:
            body.append('<span class="na">N/A</span>')
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(config.title)}</title>"
        "<style>body{font-family:sans-serif;margin:2rem;color:#222}"
        "pre{margin:.15rem 0;font-family:ui-monospace,monospace;font-size:.85rem}"
        "h1,h2{border-bottom:1px solid #ccc;padding-bottom:.25rem}</style>"
        "</head><body>" + "\n".join(body) + "</body></html>"
    )


def _html_table_from_md(lines: list[str]) -> str:
    import html

    if len(lines) < 2:
        return ""
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in lines[2:]
    ]
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_summary_table(results: list[RunAnalysisResult]) -> list[dict[str, Any]]:
    from benchmark.analysis.export import _run_to_row

    rows = []
    for run in results:
        row = _run_to_row(run)
        row.setdefault("total_score", run.total_score)
        row.setdefault("scene_total_score", run.total_score)
        rows.append(row)
    return rows
