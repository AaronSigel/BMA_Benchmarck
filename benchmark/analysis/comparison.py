from __future__ import annotations

import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from benchmark.analysis.models import (
    ComparisonDimension,
    ComparisonGroup,
    ComparisonReport,
    ExperimentAnalysisResult,
    ExperimentSummary,
    RankedGroup,
    RankedRun,
    RunAnalysisResult,
)
from benchmark.runner.error_classification import is_hard_model_failure, is_soft_success_diagnostic


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------------------
# Key-extraction helpers
# ---------------------------------------------------------------------------

# Task category: keywords mapped to canonical category names
_TASK_CATEGORY_KEYWORDS: list[tuple[str, re.Pattern[str]]] = [
    ("geometry",  re.compile(r"geom|object|primitive|mesh|cube|sphere|cylinder|cone|plane", re.I)),
    ("materials", re.compile(r"material|texture|color|colour|bsdf|roughness|metallic", re.I)),
    ("lighting",  re.compile(r"light|lamp|sun|point|spot|area|illuminat", re.I)),
    ("camera",    re.compile(r"camera|cam|focal|lens|view", re.I)),
    ("export",    re.compile(r"export|gltf|glb|obj|fbx|usd|save", re.I)),
]

_DIFFICULTY_PATTERN = re.compile(r"\b(easy|medium|hard)\b", re.I)

# Remote provider: detect from model string or stored metrics
_PROVIDER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic",  re.compile(r"anthropic|claude", re.I)),
    ("openrouter", re.compile(r"openrouter", re.I)),
    ("openai",     re.compile(r"openai|gpt-", re.I)),
    ("mock",       re.compile(r"^mock$", re.I)),
]


def _extract_task_category(result: RunAnalysisResult) -> str:
    """Infer task category from task_id, metrics, or validation scores."""
    # 1. Explicit metric key (set by run pipeline)
    cat = result.metrics.get("task_category") or result.metrics.get("run_summary.task_category")
    if cat and isinstance(cat, str):
        return cat.lower()

    # 2. Keyword match in task_id
    for name, pattern in _TASK_CATEGORY_KEYWORDS:
        if pattern.search(result.task_id):
            return name

    # 3. Infer from validation scores present in metrics
    if result.metrics.get("object_score") is not None:
        return "geometry"
    if result.metrics.get("material_score") is not None:
        return "materials"
    if result.metrics.get("light_score") is not None:
        return "lighting"
    if result.metrics.get("camera_score") is not None:
        return "camera"
    if result.metrics.get("export_score") is not None:
        return "export"

    return "unknown"


def _extract_difficulty(result: RunAnalysisResult) -> str:
    """Infer difficulty from task_id or metrics."""
    diff = result.metrics.get("difficulty") or result.metrics.get("run_summary.difficulty")
    if diff and isinstance(diff, str):
        return diff.lower()

    m = _DIFFICULTY_PATTERN.search(result.task_id)
    if m:
        return m.group(1).lower()

    return "unknown"


def _extract_remote_provider(result: RunAnalysisResult) -> str:
    """Infer remote provider from model string or metrics."""
    # Explicit metric
    provider = result.metrics.get("provider") or result.metrics.get("run_summary.provider")
    if provider and isinstance(provider, str):
        return provider.lower()

    # Try model string
    model_str = result.model or ""
    for name, pattern in _PROVIDER_PATTERNS:
        if pattern.search(model_str):
            return name

    # Execution mode can hint at remote vs local
    exec_mode = result.metrics.get("execution_mode", "")
    if "remote" in str(exec_mode).lower():
        return "remote_agent"

    return "unknown"


def _key_for_dimension(result: RunAnalysisResult, dimension: ComparisonDimension) -> str:
    if dimension == ComparisonDimension.STRATEGY:
        return result.strategy
    if dimension == ComparisonDimension.MODEL:
        return result.model or "unknown"
    if dimension == ComparisonDimension.MCP_PROFILE:
        return result.mcp_profile or "unknown"
    if dimension == ComparisonDimension.RUN:
        return result.run_id
    if dimension == ComparisonDimension.AGENT_ID:
        return result.agent_id
    if dimension == ComparisonDimension.TASK_CATEGORY:
        return _extract_task_category(result)
    if dimension == ComparisonDimension.DIFFICULTY:
        return _extract_difficulty(result)
    if dimension == ComparisonDimension.REMOTE_PROVIDER:
        return _extract_remote_provider(result)
    return "unknown"


# ---------------------------------------------------------------------------
# Core grouping engine
# ---------------------------------------------------------------------------


def _group_results(
    results: list[RunAnalysisResult],
    key_fn: Callable[[RunAnalysisResult], str],
    dimension: ComparisonDimension,
) -> ComparisonReport:
    groups: dict[str, list[RunAnalysisResult]] = defaultdict(list)
    for r in results:
        groups[key_fn(r)].append(r)

    stats: list[ComparisonGroup] = []
    for value, items in sorted(groups.items()):
        successes = [i for i in items if _effective_pass_type(i) in {"clean_pass", "soft_pass"}]
        success_rate = len(successes) / len(items) if items else None

        scores = [i.total_score for i in items if i.total_score is not None]
        avg_score = _avg(scores)  # type: ignore[arg-type]
        avg_tool_calls = _avg([float(i.tool_call_count) for i in items])
        avg_duration = _avg([i.duration_sec for i in items if i.duration_sec is not None])  # type: ignore[arg-type]
        costs = [
            float(i.metrics["provider_reported_cost_usd"])
            for i in items
            if isinstance(i.metrics.get("provider_reported_cost_usd"), (int, float))
        ]
        validation_failures = sum(
            1
            for i in items
            if i.validation_status == "failed"
            or (
                isinstance(i.metrics.get("failed_validator_count"), int)
                and int(i.metrics["failed_validator_count"]) > 0
            )
        )

        stats.append(
            ComparisonGroup(
                dimension=dimension,
                value=value,
                run_count=len(items),
                success_rate=success_rate,
                avg_score=avg_score,
                avg_tool_calls=avg_tool_calls,
                avg_duration_sec=avg_duration,
                avg_cost=_avg(costs),
                validation_failures=validation_failures,
            )
        )

    return ComparisonReport(dimension=dimension, groups=stats)


def compare_runs(
    results: list[RunAnalysisResult],
    dimension: ComparisonDimension = ComparisonDimension.STRATEGY,
) -> ComparisonReport:
    """Generic comparison grouped by any ComparisonDimension."""
    return _group_results(
        results,
        key_fn=lambda r: _key_for_dimension(r, dimension),
        dimension=dimension,
    )


# ---------------------------------------------------------------------------
# Named grouping functions (acceptance criteria surface)
# ---------------------------------------------------------------------------


def group_by_strategy(results: list[RunAnalysisResult]) -> ComparisonReport:
    """Compare ReAct vs Direct vs Plan-and-Execute."""
    return compare_runs(results, ComparisonDimension.STRATEGY)


def group_by_model(results: list[RunAnalysisResult]) -> ComparisonReport:
    """Compare runs by LLM model name."""
    return compare_runs(results, ComparisonDimension.MODEL)


def group_by_agent_id(results: list[RunAnalysisResult]) -> ComparisonReport:
    """Compare runs by agent_id."""
    return compare_runs(results, ComparisonDimension.AGENT_ID)


def group_by_mcp_profile(results: list[RunAnalysisResult]) -> ComparisonReport:
    """Compare minimal vs inspection_enabled vs no_python vs python_enabled vs full."""
    return compare_runs(results, ComparisonDimension.MCP_PROFILE)


def group_by_task_category(results: list[RunAnalysisResult]) -> ComparisonReport:
    """Compare geometry vs materials vs camera vs lighting vs export tasks."""
    return compare_runs(results, ComparisonDimension.TASK_CATEGORY)


def group_by_difficulty(results: list[RunAnalysisResult]) -> ComparisonReport:
    """Compare easy vs medium vs hard tasks."""
    return compare_runs(results, ComparisonDimension.DIFFICULTY)


def group_by_remote_provider(results: list[RunAnalysisResult]) -> ComparisonReport:
    """Compare OpenRouter vs Anthropic vs OpenAI vs remote_agent runs."""
    return compare_runs(results, ComparisonDimension.REMOTE_PROVIDER)


# ---------------------------------------------------------------------------
# Convenience aliases kept from earlier implementation
# ---------------------------------------------------------------------------


def compare_strategies(results: list[RunAnalysisResult]) -> ComparisonReport:
    return group_by_strategy(results)


def compare_models(results: list[RunAnalysisResult]) -> ComparisonReport:
    return group_by_model(results)


# ---------------------------------------------------------------------------
# Experiment summary computation
# ---------------------------------------------------------------------------

_INFRA_TIMEOUT_TYPES = frozenset({"ToolTimeout", "SocketTimeout"})
_TERMINAL_INFRA_TYPES = _INFRA_TIMEOUT_TYPES | frozenset({
    "EmptySocketResponse",
    "BlenderWorkerUnhealthy",
    "ResetSceneFailed",
    "BlenderSocketUnavailable",
    "BlenderSocketNoResponse",
    "SocketError",
})


def _structured_error_type(result: RunAnalysisResult) -> str:
    return str(
        result.metrics.get("structured_error_type") or result.metrics.get("error_type") or ""
    ).strip()


def count_terminal_infra_from_runs(results: list[RunAnalysisResult]) -> dict[str, int]:
    """Terminal infra counts from final run-level error types (readiness gate source of truth)."""
    socket_timeouts = sum(
        1 for r in results if _structured_error_type(r) in _INFRA_TIMEOUT_TYPES
    )
    empty_socket_responses = sum(
        1 for r in results if _structured_error_type(r) == "EmptySocketResponse"
    )
    worker_unhealthy = sum(
        1 for r in results if _structured_error_type(r) == "BlenderWorkerUnhealthy"
    )
    reset_scene_failed = sum(
        1 for r in results if _structured_error_type(r) == "ResetSceneFailed"
    )
    return {
        "socket_timeout_count": socket_timeouts,
        "empty_socket_response_count": empty_socket_responses,
        "worker_unhealthy_count": worker_unhealthy,
        "reset_scene_failed_count": reset_scene_failed,
        "terminal_infra_count": sum(
            1 for r in results if _structured_error_type(r) in _TERMINAL_INFRA_TYPES
        ),
    }


def build_infra_reliability_payload(
    summary: ExperimentSummary,
    metadata: dict | None = None,
) -> dict[str, object]:
    """Dual-layer infra metrics: terminal run failures vs watchdog event totals."""
    meta = metadata if isinstance(metadata, dict) else {}
    watchdog = meta.get("watchdog_counters")
    watchdog_payload = watchdog if isinstance(watchdog, dict) else {}
    return {
        "terminal_socket_timeout_count": summary.infra_socket_timeouts,
        "terminal_empty_socket_response_count": summary.infra_empty_socket_responses,
        "terminal_worker_restart_count": summary.infra_worker_restarts,
        "watchdog_socket_timeout_count": int(watchdog_payload.get("infra_socket_timeouts", 0) or 0),
        "watchdog_empty_socket_response_count": int(
            watchdog_payload.get("infra_empty_socket_responses", 0) or 0
        ),
        "watchdog_worker_restart_count": int(watchdog_payload.get("infra_worker_restarts", 0) or 0),
    }


def _build_summary(results: list[RunAnalysisResult]) -> ExperimentSummary:
    total = len(results)
    successful = [r for r in results if _effective_pass_type(r) in {"clean_pass", "soft_pass"}]
    failed = [r for r in results if _effective_pass_type(r) in {"failed_validation", "runtime_error"}]

    active_runs = [r for r in results if r.success is not None]
    scored_runs = [r for r in active_runs if r.total_score is not None]

    avg_score = _avg([r.total_score for r in scored_runs])  # type: ignore[misc]
    passed_scores = [r.total_score for r in results if r.run_status == "passed" and r.total_score is not None]
    avg_tool_calls = _avg([float(r.tool_call_count) for r in active_runs]) if active_runs else None
    avg_duration = _avg([r.duration_sec for r in active_runs if r.duration_sec is not None]) if active_runs else None
    avg_llm = _avg([float(r.llm_call_count) for r in active_runs]) if active_runs else None

    error_counter: Counter[str] = Counter()
    for r in results:
        for key, val in r.metrics.items():
            if key.startswith("error.") and isinstance(val, int) and val > 0:
                error_counter[key[len("error."):]] += val
    most_common = error_counter.most_common(10)

    scored_success = [(r.run_id, r.total_score) for r in successful if r.total_score is not None]
    best_run: str | None = max(scored_success, key=lambda x: x[1])[0] if scored_success else None
    if best_run is None and scored_runs:
        best_run = max(scored_runs, key=lambda r: r.total_score).run_id  # type: ignore[return-value]

    all_scored = [(r.run_id, r.total_score) for r in results if r.total_score is not None]
    worst_run: str | None = min(all_scored, key=lambda x: x[1])[0] if all_scored else (failed[0].run_id if failed else None)

    clean_pass = sum(1 for r in results if _effective_pass_type(r) == "clean_pass")
    soft_pass = sum(1 for r in results if _effective_pass_type(r) == "soft_pass")
    failed_ct = sum(1 for r in results if _effective_pass_type(r) == "failed_validation")
    error_ct = sum(1 for r in results if _effective_pass_type(r) == "runtime_error")
    agent_completed = sum(1 for r in results if r.agent_status == "completed")
    agent_completed_after = sum(1 for r in results if r.agent_status == "completed_after_scene_passed")
    agent_incomplete = sum(
        1 for r in results
        if r.agent_status in {"max_steps_reached", "invalid_response", "repeated_action_detected",
                              "duplicate_object_detected", "no_progress_detected"}
    )
    agent_err = sum(
        1 for r in results
        if r.agent_status in {"tool_error", "runtime_error", None}
        and r.run_status == "error"
    )

    infra_runs = sum(1 for r in results if bool(r.metrics.get("is_infra_failure")))
    model_failures = sum(
        1 for r in results
        if is_hard_model_failure(
            is_model_failure=bool(r.metrics.get("is_model_failure")),
            is_infra_failure=bool(r.metrics.get("is_infra_failure")),
            error_class=str(r.metrics.get("error_class") or "") or None,
            diagnostic_only=bool(r.metrics.get("diagnostic_only")),
            pass_type=str(r.pass_type or "") or None,
            scene_status=str(r.scene_status or "") or None,
            error_type=str(r.metrics.get("structured_error_type") or r.metrics.get("react_error_type") or "") or None,
        )
    )
    soft_success_diagnostics = sum(
        1 for r in results
        if is_soft_success_diagnostic(
            error_class=str(r.metrics.get("error_class") or "") or None,
            diagnostic_only=bool(r.metrics.get("diagnostic_only")),
        )
        or (
            str(r.pass_type or "") == "soft_pass"
            and str(r.scene_status or "") == "passed"
            and str(r.metrics.get("structured_error_type") or r.metrics.get("react_error_type") or "").strip()
            in {"ReactMaxSteps", "ReactInvalidAction", "ReactNoProgress"}
            and not bool(r.metrics.get("is_infra_failure"))
        )
    )
    validation_failures = sum(1 for r in results if bool(r.metrics.get("is_validation_failure")))
    tool_runtime_failures = sum(1 for r in results if bool(r.metrics.get("is_tool_runtime_failure")))
    success_excluding_infra = sum(
        1 for r in results
        if _effective_pass_type(r) in {"clean_pass", "soft_pass"}
        and not bool(r.metrics.get("is_infra_failure"))
    )
    clean_pass_excluding_infra = sum(
        1 for r in results
        if _effective_pass_type(r) == "clean_pass"
        and not bool(r.metrics.get("is_infra_failure"))
    )
    healthy_total = total - infra_runs
    no_progress_by_reason: dict[str, int] = {}
    for r in results:
        reason = str(r.metrics.get("no_progress_reason") or "").strip()
        if reason:
            no_progress_by_reason[reason] = no_progress_by_reason.get(reason, 0) + 1

    terminal_infra = count_terminal_infra_from_runs(results)

    return ExperimentSummary(
        total_runs=total,
        successful_runs=len(successful),
        failed_runs=failed_ct,
        error_runs=error_ct,
        clean_pass_count=clean_pass,
        soft_pass_count=soft_pass,
        failed_validation_count=failed_ct,
        runtime_error_count=error_ct,
        failure_rate=((failed_ct + error_ct) / total if total else None),
        failed_count=failed_ct,
        error_count=error_ct,
        clean_pass_rate=(clean_pass / total if total else None),
        soft_pass_rate=(soft_pass / total if total else None),
        strict_success_rate=(clean_pass / total if total else None),
        reported_success_rate=((clean_pass + soft_pass) / total if total else None),
        reported_success_rate_all_runs=((clean_pass + soft_pass) / total if total else None),
        reported_success_rate_excluding_infra=(
            success_excluding_infra / healthy_total if healthy_total else None
        ),
        strict_success_rate_excluding_infra=(
            clean_pass_excluding_infra / healthy_total if healthy_total else None
        ),
        infra_error_rate=(infra_runs / total if total else None),
        model_failure_rate=(model_failures / total if total else None),
        soft_success_diagnostic_rate=(soft_success_diagnostics / total if total else None),
        validation_failure_rate=(validation_failures / total if total else None),
        tool_runtime_failure_rate=(tool_runtime_failures / total if total else None),
        infra_socket_timeouts=terminal_infra["socket_timeout_count"],
        infra_empty_socket_responses=terminal_infra["empty_socket_response_count"],
        no_progress_by_reason=no_progress_by_reason,
        agent_completed_count=agent_completed,
        agent_completed_after_scene_passed_count=agent_completed_after,
        agent_incomplete_count=agent_incomplete,
        agent_error_count=agent_err,
        average_scene_score=avg_score,
        average_score_completed=avg_score,
        average_score_strict=(
            sum((r.total_score or 0.0) for r in results) / total if total else None
        ),
        average_score_passed_only=_avg(passed_scores),  # type: ignore[arg-type]
        scene_success_rate=(
            sum(1 for r in results if (r.scene_status or r.validation_status) == "passed") / total
            if total else None
        ),
        run_success_rate=(sum(1 for r in results if r.run_status == "passed" or r.success is True) / total if total else None),
        agent_completion_rate=(
            sum(1 for r in results if r.agent_status in {"completed", "completed_after_scene_passed"}) / total
            if total else None
        ),
        average_tool_call_count=avg_tool_calls,
        average_duration_sec=avg_duration,
        average_llm_calls=avg_llm,
        most_common_errors=most_common,
        best_run=best_run,
        worst_run=worst_run,
    )


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


def analyze_run_results(
    results: list[RunAnalysisResult],
    experiment_id: str = "experiment",
    metadata: dict | None = None,
) -> ExperimentAnalysisResult:
    """Aggregate a list of RunAnalysisResult into an ExperimentAnalysisResult."""
    meta = dict(metadata or {})
    summary = _build_summary(results)
    watchdog = meta.get("watchdog_counters")
    if isinstance(watchdog, dict):
        summary = summary.model_copy(
            update={
                "infra_worker_restarts": int(watchdog.get("infra_worker_restarts", 0) or 0),
            }
        )
    return ExperimentAnalysisResult(
        experiment_id=experiment_id,
        runs=results,
        summary=summary,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Ranking functions
# ---------------------------------------------------------------------------


def _apply_top_bottom(items: list, top_n: int | None, bottom_n: int | None) -> list:
    if top_n is not None and bottom_n is not None:
        seen: set[int] = set()
        result = []
        for item in items[:top_n]:
            if id(item) not in seen:
                seen.add(id(item))
                result.append(item)
        for item in items[-bottom_n:]:
            if id(item) not in seen:
                seen.add(id(item))
                result.append(item)
        return result
    if top_n is not None:
        return items[:top_n]
    if bottom_n is not None:
        return items[-bottom_n:]
    return items


def rank_runs_by_score(
    results: list[RunAnalysisResult],
    top_n: int | None = None,
    bottom_n: int | None = None,
) -> list[RankedRun]:
    """Rank runs by total_score descending. Runs with no score sort last."""
    scored = sorted(
        results,
        key=lambda r: (r.total_score is None, -(r.total_score or 0.0)),
    )
    ranked: list[RankedRun] = []
    for i, run in enumerate(scored, start=1):
        score = run.total_score
        time_eff: float | None = None
        tool_eff: float | None = None
        if score is not None:
            dur = run.duration_sec if run.duration_sec is not None else 0.0
            time_eff = score / max(dur, 1.0)
            tool_eff = score / max(float(run.tool_call_count), 1.0)
        ranked.append(RankedRun(rank=i, run=run, score_used=score, time_efficiency=time_eff, tool_efficiency=tool_eff))
    return _apply_top_bottom(ranked, top_n, bottom_n)


def rank_groups_by_average_score(
    groups: list[ComparisonGroup],
    top_n: int | None = None,
    bottom_n: int | None = None,
) -> list[RankedGroup]:
    """Rank ComparisonGroups by avg_score descending. Groups with no score sort last."""
    sorted_groups = sorted(
        groups,
        key=lambda g: (g.avg_score is None, -(g.avg_score or 0.0)),
    )
    ranked: list[RankedGroup] = []
    for i, group in enumerate(sorted_groups, start=1):
        score = group.avg_score
        time_eff: float | None = None
        tool_eff: float | None = None
        if score is not None:
            dur = group.avg_duration_sec if group.avg_duration_sec is not None else 0.0
            time_eff = score / max(dur, 1.0)
            tc = group.avg_tool_calls if group.avg_tool_calls is not None else 0.0
            tool_eff = score / max(tc, 1.0)
        ranked.append(RankedGroup(rank=i, group=group, score_used=score, time_efficiency=time_eff, tool_efficiency=tool_eff))
    return _apply_top_bottom(ranked, top_n, bottom_n)


def rank_groups_by_success_rate(
    groups: list[ComparisonGroup],
    top_n: int | None = None,
    bottom_n: int | None = None,
) -> list[RankedGroup]:
    """Rank ComparisonGroups by success_rate descending. Groups with no rate sort last."""
    sorted_groups = sorted(
        groups,
        key=lambda g: (g.success_rate is None, -(g.success_rate or 0.0)),
    )
    ranked: list[RankedGroup] = []
    for i, group in enumerate(sorted_groups, start=1):
        rate = group.success_rate
        time_eff: float | None = None
        tool_eff: float | None = None
        if rate is not None:
            dur = group.avg_duration_sec if group.avg_duration_sec is not None else 0.0
            time_eff = rate / max(dur, 1.0)
            tc = group.avg_tool_calls if group.avg_tool_calls is not None else 0.0
            tool_eff = rate / max(tc, 1.0)
        ranked.append(RankedGroup(rank=i, group=group, score_used=rate, time_efficiency=time_eff, tool_efficiency=tool_eff))
    return _apply_top_bottom(ranked, top_n, bottom_n)


def rank_groups_by_efficiency(
    groups: list[ComparisonGroup],
    top_n: int | None = None,
    bottom_n: int | None = None,
) -> list[RankedGroup]:
    """Rank ComparisonGroups by time_efficiency (avg_score / max(avg_duration_sec, 1)) descending."""
    def _time_eff(g: ComparisonGroup) -> float:
        if g.avg_score is None:
            return 0.0
        dur = g.avg_duration_sec if g.avg_duration_sec is not None else 0.0
        return g.avg_score / max(dur, 1.0)

    sorted_groups = sorted(
        groups,
        key=lambda g: (g.avg_score is None, -_time_eff(g)),
    )
    ranked: list[RankedGroup] = []
    for i, group in enumerate(sorted_groups, start=1):
        score = group.avg_score
        time_eff: float | None = None
        tool_eff: float | None = None
        if score is not None:
            dur = group.avg_duration_sec if group.avg_duration_sec is not None else 0.0
            time_eff = score / max(dur, 1.0)
            tc = group.avg_tool_calls if group.avg_tool_calls is not None else 0.0
            tool_eff = score / max(tc, 1.0)
        ranked.append(RankedGroup(rank=i, group=group, score_used=time_eff, time_efficiency=time_eff, tool_efficiency=tool_eff))
    return _apply_top_bottom(ranked, top_n, bottom_n)


def analyze_experiment(experiment_dir: Path | str) -> ExperimentAnalysisResult:
    """Discover and analyze all runs under *experiment_dir* (idempotent)."""
    import json
    from benchmark.analysis.run_analysis import analyze_run
    from benchmark.analysis.trace_reader import discover_run_artifacts, load_run_bundle

    root = Path(experiment_dir)
    run_dirs = discover_run_artifacts(root)
    results: list[RunAnalysisResult] = []

    for run_dir in run_dirs:
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                bundle = load_run_bundle(run_dir)
            results.append(analyze_run(bundle))
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning("Skipping run dir %s: %s", run_dir, exc)

    metadata: dict = {}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_metadata = manifest.get("metadata")
            if isinstance(raw_metadata, dict):
                metadata.update(raw_metadata)
                matrix_policy = raw_metadata.get("matrix_policy")
                if isinstance(matrix_policy, dict):
                    if isinstance(matrix_policy.get("analysis_policy"), dict):
                        metadata["analysis_policy"] = matrix_policy["analysis_policy"]
                    if isinstance(matrix_policy.get("reporting"), dict):
                        metadata["reporting"] = matrix_policy["reporting"]
                    if isinstance(matrix_policy.get("cost"), dict):
                        metadata["cost"] = matrix_policy["cost"]
            metadata["manifest_path"] = str(manifest_path)
            metadata["manifest_generated_at"] = manifest.get("generated_at")
            metadata["models_used"] = manifest.get("models")
            metadata["strategies_used"] = manifest.get("agent_ids")
            metadata["mcp_profiles_used"] = manifest.get("mcp_profiles")
            metadata["tasks_used"] = manifest.get("task_ids")
            metadata["_manifest_agent_ids"] = manifest.get("agent_ids")
            metadata["repetitions"] = manifest.get("repetitions")
            metadata["tool_contract_hash"] = raw_metadata.get("runtime", {}).get("tool_contract_hash") if isinstance(raw_metadata, dict) else None
        except (OSError, json.JSONDecodeError) as exc:
            metadata["manifest_error"] = str(exc)
    # Derive strategies and profiles from actual run results (more reliable than manifest agent IDs)
    unique_strategies = sorted(set(
        r.strategy for r in results if r.strategy and r.strategy not in {"unknown", ""}
    ))
    if unique_strategies:
        metadata["strategies_used"] = unique_strategies
    unique_profiles = sorted(set(
        r.mcp_profile for r in results if r.mcp_profile and r.mcp_profile not in {"unknown", ""}
    ))
    if unique_profiles:
        metadata["mcp_profiles_used"] = unique_profiles
    unique_models = sorted(set(
        r.model for r in results if r.model and r.model not in {"unknown", ""}
    ))
    if unique_models:
        metadata["models_used"] = unique_models
    unique_tasks = sorted(set(
        r.task_id for r in results if r.task_id and r.task_id not in {"unknown", ""}
    ))
    if unique_tasks:
        metadata["tasks_used"] = unique_tasks

    metadata["executed_runs"] = len(results)
    metadata["artifact_count"] = sum(len(r.artifacts) for r in results)
    provider_costs = [
        float(r.metrics["provider_reported_cost_usd"])
        for r in results
        if isinstance(r.metrics.get("provider_reported_cost_usd"), (int, float))
    ]
    metadata["total_provider_reported_cost_usd"] = sum(provider_costs) if provider_costs else None
    metadata["runs_with_provider_cost"] = sum(1 for r in results if r.metrics.get("provider_cost_available") is True)
    metadata["runs_without_provider_cost"] = sum(1 for r in results if r.metrics.get("provider_cost_available") is not True)
    planned = metadata.get("planned_runs") or metadata.get("expected_runs")
    metadata["missing_artifacts"] = max(0, int(planned) - len(results)) if isinstance(planned, int) else 0

    experiment_result_path = root / "experiment_result.json"
    if experiment_result_path.exists():
        try:
            experiment_payload = json.loads(experiment_result_path.read_text(encoding="utf-8"))
            experiment_summary = experiment_payload.get("summary")
            if isinstance(experiment_summary, dict):
                watchdog_keys = (
                    "infra_socket_timeouts",
                    "infra_empty_socket_responses",
                    "infra_worker_restarts",
                    "infra_reset_failures",
                    "infra_snapshot_failures",
                    "restart_reasons",
                    "runs_since_last_restart",
                )
                metadata["watchdog_counters"] = {
                    key: experiment_summary[key]
                    for key in watchdog_keys
                    if key in experiment_summary
                }
        except (OSError, json.JSONDecodeError) as exc:
            metadata["experiment_result_error"] = str(exc)

    return analyze_run_results(results, experiment_id=root.name, metadata=metadata)
