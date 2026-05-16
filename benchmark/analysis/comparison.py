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
        successes = [i for i in items if i.success is True]
        success_rate = len(successes) / len(items) if items else None

        scores = [i.total_score for i in items if i.total_score is not None]
        avg_score = _avg(scores)  # type: ignore[arg-type]
        avg_tool_calls = _avg([float(i.tool_call_count) for i in items])
        avg_duration = _avg([i.duration_sec for i in items if i.duration_sec is not None])  # type: ignore[arg-type]

        stats.append(
            ComparisonGroup(
                dimension=dimension,
                value=value,
                run_count=len(items),
                success_rate=success_rate,
                avg_score=avg_score,
                avg_tool_calls=avg_tool_calls,
                avg_duration_sec=avg_duration,
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


def _build_summary(results: list[RunAnalysisResult]) -> ExperimentSummary:
    total = len(results)
    successful = [r for r in results if r.success is True]
    failed = [r for r in results if r.success is False]
    errored = [r for r in results if r.success is None]

    active_runs = [r for r in results if r.success is not None]
    scored_runs = [r for r in active_runs if r.total_score is not None]

    avg_score = _avg([r.total_score for r in scored_runs])  # type: ignore[misc]
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

    return ExperimentSummary(
        total_runs=total,
        successful_runs=len(successful),
        failed_runs=len(failed),
        error_runs=len(errored),
        average_scene_score=avg_score,
        average_tool_call_count=avg_tool_calls,
        average_duration_sec=avg_duration,
        average_llm_calls=avg_llm,
        most_common_errors=most_common,
        best_run=best_run,
        worst_run=worst_run,
    )


def analyze_run_results(
    results: list[RunAnalysisResult],
    experiment_id: str = "experiment",
    metadata: dict | None = None,
) -> ExperimentAnalysisResult:
    """Aggregate a list of RunAnalysisResult into an ExperimentAnalysisResult."""
    return ExperimentAnalysisResult(
        experiment_id=experiment_id,
        runs=results,
        summary=_build_summary(results),
        metadata=metadata or {},
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

    return analyze_run_results(results, experiment_id=root.name)
