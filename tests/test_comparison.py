"""Tests for benchmark.analysis.comparison — grouping and ranking."""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.analysis.comparison import (
    analyze_experiment,
    analyze_run_results,
    compare_runs,
    group_by_agent_id,
    group_by_difficulty,
    group_by_mcp_profile,
    group_by_model,
    group_by_strategy,
    group_by_task_category,
    rank_groups_by_average_score,
    rank_groups_by_efficiency,
    rank_groups_by_success_rate,
    rank_runs_by_score,
)
from benchmark.analysis.models import ComparisonDimension, ComparisonGroup, RunAnalysisResult

FIXTURES = Path(__file__).parent / "fixtures" / "analysis"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    run_id: str = "r1",
    strategy: str = "react",
    model: str | None = "claude-3",
    mcp_profile: str | None = "minimal",
    total_score: float | None = None,
    success: bool | None = None,
    duration_sec: float | None = None,
    tool_call_count: int = 0,
    task_id: str = "task_easy",
    agent_id: str = "agent_a",
    metrics: dict | None = None,
) -> RunAnalysisResult:
    return RunAnalysisResult(
        run_id=run_id,
        task_id=task_id,
        agent_id=agent_id,
        strategy=strategy,
        model=model,
        mcp_profile=mcp_profile,
        total_score=total_score,
        success=success,
        duration_sec=duration_sec,
        tool_call_count=tool_call_count,
        metrics=metrics or {},
    )


def _group(
    value: str,
    dimension: ComparisonDimension = ComparisonDimension.STRATEGY,
    run_count: int = 3,
    avg_score: float | None = None,
    success_rate: float | None = None,
    avg_tool_calls: float | None = None,
    avg_duration_sec: float | None = None,
) -> ComparisonGroup:
    return ComparisonGroup(
        dimension=dimension,
        value=value,
        run_count=run_count,
        avg_score=avg_score,
        success_rate=success_rate,
        avg_tool_calls=avg_tool_calls,
        avg_duration_sec=avg_duration_sec,
    )


# ---------------------------------------------------------------------------
# group_by_strategy
# ---------------------------------------------------------------------------

class TestGroupByStrategy:
    def test_single_strategy(self):
        runs = [_run("r1", strategy="react"), _run("r2", strategy="react")]
        report = group_by_strategy(runs)
        assert len(report.groups) == 1
        assert report.groups[0].value == "react"
        assert report.groups[0].run_count == 2

    def test_multiple_strategies(self):
        runs = [
            _run("r1", strategy="react"),
            _run("r2", strategy="direct_tool_calling"),
            _run("r3", strategy="react"),
            _run("r4", strategy="plan_and_execute"),
        ]
        report = group_by_strategy(runs)
        values = {g.value for g in report.groups}
        assert values == {"react", "direct_tool_calling", "plan_and_execute"}
        react_g = next(g for g in report.groups if g.value == "react")
        assert react_g.run_count == 2

    def test_success_rate_computed(self):
        runs = [
            _run("r1", strategy="react", success=True),
            _run("r2", strategy="react", success=True),
            _run("r3", strategy="react", success=False),
        ]
        report = group_by_strategy(runs)
        g = report.groups[0]
        assert g.success_rate == pytest.approx(2 / 3)

    def test_avg_score_computed(self):
        runs = [
            _run("r1", strategy="react", total_score=0.8),
            _run("r2", strategy="react", total_score=0.6),
        ]
        report = group_by_strategy(runs)
        assert report.groups[0].avg_score == pytest.approx(0.7)

    def test_none_scores_excluded_from_avg(self):
        runs = [
            _run("r1", strategy="react", total_score=1.0),
            _run("r2", strategy="react", total_score=None),
        ]
        report = group_by_strategy(runs)
        assert report.groups[0].avg_score == pytest.approx(1.0)

    def test_dimension_set_correctly(self):
        report = group_by_strategy([_run()])
        assert report.dimension == ComparisonDimension.STRATEGY
        assert report.groups[0].dimension == ComparisonDimension.STRATEGY

    def test_empty_runs_returns_empty_groups(self):
        report = group_by_strategy([])
        assert report.groups == []


# ---------------------------------------------------------------------------
# group_by_mcp_profile
# ---------------------------------------------------------------------------

class TestGroupByMcpProfile:
    def test_groups_by_mcp_profile(self):
        runs = [
            _run("r1", mcp_profile="minimal"),
            _run("r2", mcp_profile="full"),
            _run("r3", mcp_profile="minimal"),
        ]
        report = group_by_mcp_profile(runs)
        values = {g.value for g in report.groups}
        assert values == {"minimal", "full"}
        minimal = next(g for g in report.groups if g.value == "minimal")
        assert minimal.run_count == 2

    def test_none_mcp_profile_becomes_unknown(self):
        runs = [_run("r1", mcp_profile=None)]
        report = group_by_mcp_profile(runs)
        assert report.groups[0].value == "unknown"

    def test_dimension_is_mcp_profile(self):
        report = group_by_mcp_profile([_run()])
        assert report.dimension == ComparisonDimension.MCP_PROFILE


# ---------------------------------------------------------------------------
# group_by_model
# ---------------------------------------------------------------------------

class TestGroupByModel:
    def test_groups_by_model(self):
        runs = [
            _run("r1", model="claude-3"),
            _run("r2", model="gpt-4"),
            _run("r3", model="claude-3"),
        ]
        report = group_by_model(runs)
        values = {g.value for g in report.groups}
        assert values == {"claude-3", "gpt-4"}

    def test_none_model_becomes_unknown(self):
        runs = [_run("r1", model=None)]
        report = group_by_model(runs)
        assert report.groups[0].value == "unknown"

    def test_avg_duration_computed(self):
        runs = [
            _run("r1", model="claude-3", duration_sec=10.0),
            _run("r2", model="claude-3", duration_sec=20.0),
        ]
        report = group_by_model(runs)
        assert report.groups[0].avg_duration_sec == pytest.approx(15.0)

    def test_avg_tool_calls_computed(self):
        runs = [
            _run("r1", model="claude-3", tool_call_count=4),
            _run("r2", model="claude-3", tool_call_count=8),
        ]
        report = group_by_model(runs)
        assert report.groups[0].avg_tool_calls == pytest.approx(6.0)

    def test_dimension_is_model(self):
        report = group_by_model([_run()])
        assert report.dimension == ComparisonDimension.MODEL


# ---------------------------------------------------------------------------
# compare_runs generic
# ---------------------------------------------------------------------------

class TestCompareRuns:
    def test_by_agent_id(self):
        runs = [
            _run("r1", agent_id="agent_a"),
            _run("r2", agent_id="agent_b"),
            _run("r3", agent_id="agent_a"),
        ]
        report = compare_runs(runs, ComparisonDimension.AGENT_ID)
        assert report.dimension == ComparisonDimension.AGENT_ID
        assert len(report.groups) == 2

    def test_task_category_from_task_id_keyword(self):
        runs = [
            _run("r1", task_id="geometry_add_cube_easy"),
            _run("r2", task_id="materials_set_roughness"),
            _run("r3", task_id="geometry_add_sphere"),
        ]
        report = compare_runs(runs, ComparisonDimension.TASK_CATEGORY)
        values = {g.value for g in report.groups}
        assert "geometry" in values
        assert "materials" in values

    def test_difficulty_from_task_id_keyword(self):
        # difficulty keywords need a word boundary — use hyphen or space, not underscore
        runs = [
            _run("r1", task_id="geometry-easy-cube"),
            _run("r2", task_id="material-hard"),
            _run("r3", task_id="lighting-medium"),
        ]
        report = compare_runs(runs, ComparisonDimension.DIFFICULTY)
        values = {g.value for g in report.groups}
        assert "easy" in values
        assert "hard" in values
        assert "medium" in values


# ---------------------------------------------------------------------------
# analyze_run_results
# ---------------------------------------------------------------------------

class TestAnalyzeRunResults:
    def test_summary_computed(self):
        runs = [
            _run("r1", success=True, total_score=0.9),
            _run("r2", success=False, total_score=0.3),
            _run("r3", success=None),
        ]
        exp = analyze_run_results(runs, experiment_id="test_exp")
        assert exp.experiment_id == "test_exp"
        assert exp.summary.total_runs == 3
        assert exp.summary.successful_runs == 1
        assert exp.summary.failed_runs == 1
        assert exp.summary.error_runs == 1

    def test_avg_score_in_summary(self):
        # summary average_scene_score only counts runs with success is not None
        runs = [
            _run("r1", total_score=0.8, success=True),
            _run("r2", total_score=0.6, success=False),
        ]
        exp = analyze_run_results(runs)
        assert exp.summary.average_scene_score == pytest.approx(0.7)

    def test_best_and_worst_run(self):
        runs = [
            _run("best_run", success=True, total_score=1.0),
            _run("worst_run", success=False, total_score=0.1),
        ]
        exp = analyze_run_results(runs)
        assert exp.summary.best_run == "best_run"
        assert exp.summary.worst_run == "worst_run"

    def test_empty_runs(self):
        exp = analyze_run_results([])
        assert exp.summary.total_runs == 0
        assert exp.summary.average_scene_score is None


# ---------------------------------------------------------------------------
# rank_runs_by_score
# ---------------------------------------------------------------------------

class TestRankRunsByScore:
    def test_descending_order(self):
        runs = [
            _run("r1", total_score=0.5),
            _run("r2", total_score=0.9),
            _run("r3", total_score=0.7),
        ]
        ranked = rank_runs_by_score(runs)
        scores = [rr.score_used for rr in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_field_assigned(self):
        runs = [_run("r1", total_score=0.8), _run("r2", total_score=0.5)]
        ranked = rank_runs_by_score(runs)
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2

    def test_none_score_sorts_last(self):
        runs = [
            _run("r1", total_score=None),
            _run("r2", total_score=0.9),
            _run("r3", total_score=0.5),
        ]
        ranked = rank_runs_by_score(runs)
        assert ranked[-1].run.run_id == "r1"
        assert ranked[-1].score_used is None

    def test_top_n_slicing(self):
        runs = [_run(f"r{i}", total_score=float(i) / 10) for i in range(10)]
        ranked = rank_runs_by_score(runs, top_n=3)
        assert len(ranked) == 3
        assert ranked[0].rank == 1

    def test_bottom_n_slicing(self):
        runs = [_run(f"r{i}", total_score=float(i) / 10) for i in range(10)]
        ranked = rank_runs_by_score(runs, bottom_n=2)
        assert len(ranked) == 2
        assert ranked[-1].run.total_score == pytest.approx(0.0)

    def test_no_crash_on_zero_duration(self):
        runs = [_run("r1", total_score=0.8, duration_sec=0.0)]
        ranked = rank_runs_by_score(runs)
        assert ranked[0].time_efficiency == pytest.approx(0.8)

    def test_no_crash_on_none_duration(self):
        runs = [_run("r1", total_score=0.8, duration_sec=None)]
        ranked = rank_runs_by_score(runs)
        assert ranked[0].time_efficiency == pytest.approx(0.8)

    def test_efficiency_computed(self):
        runs = [_run("r1", total_score=0.8, duration_sec=4.0, tool_call_count=2)]
        ranked = rank_runs_by_score(runs)
        assert ranked[0].time_efficiency == pytest.approx(0.8 / 4.0)
        assert ranked[0].tool_efficiency == pytest.approx(0.8 / 2.0)

    def test_empty_list(self):
        assert rank_runs_by_score([]) == []


# ---------------------------------------------------------------------------
# rank_groups_by_success_rate
# ---------------------------------------------------------------------------

class TestRankGroupsBySuccessRate:
    def test_descending_by_success_rate(self):
        groups = [
            _group("a", success_rate=0.5),
            _group("b", success_rate=0.9),
            _group("c", success_rate=0.1),
        ]
        ranked = rank_groups_by_success_rate(groups)
        rates = [rg.score_used for rg in ranked]
        assert rates == sorted(rates, reverse=True)

    def test_none_rate_sorts_last(self):
        groups = [
            _group("a", success_rate=None),
            _group("b", success_rate=0.7),
        ]
        ranked = rank_groups_by_success_rate(groups)
        assert ranked[-1].group.value == "a"

    def test_ranks_assigned_from_one(self):
        groups = [_group("x", success_rate=0.6), _group("y", success_rate=0.3)]
        ranked = rank_groups_by_success_rate(groups)
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2

    def test_top_n_slicing(self):
        groups = [_group(str(i), success_rate=float(i) / 10) for i in range(8)]
        ranked = rank_groups_by_success_rate(groups, top_n=2)
        assert len(ranked) == 2

    def test_bottom_n_slicing(self):
        groups = [_group(str(i), success_rate=float(i) / 10) for i in range(8)]
        ranked = rank_groups_by_success_rate(groups, bottom_n=3)
        assert len(ranked) == 3

    def test_empty_list(self):
        assert rank_groups_by_success_rate([]) == []


# ---------------------------------------------------------------------------
# rank_groups_by_efficiency
# ---------------------------------------------------------------------------

class TestRankGroupsByEfficiency:
    def test_higher_score_per_duration_ranks_first(self):
        groups = [
            _group("slow", avg_score=0.5, avg_duration_sec=50.0),
            _group("fast", avg_score=0.5, avg_duration_sec=5.0),
        ]
        ranked = rank_groups_by_efficiency(groups)
        assert ranked[0].group.value == "fast"

    def test_no_crash_on_zero_duration(self):
        groups = [_group("a", avg_score=0.8, avg_duration_sec=0.0)]
        ranked = rank_groups_by_efficiency(groups)
        assert ranked[0].time_efficiency == pytest.approx(0.8)

    def test_no_crash_on_none_duration(self):
        groups = [_group("a", avg_score=0.8, avg_duration_sec=None)]
        ranked = rank_groups_by_efficiency(groups)
        assert ranked[0].time_efficiency == pytest.approx(0.8)

    def test_none_score_sorts_last(self):
        groups = [
            _group("a", avg_score=None),
            _group("b", avg_score=0.7, avg_duration_sec=5.0),
        ]
        ranked = rank_groups_by_efficiency(groups)
        assert ranked[-1].group.value == "a"

    def test_efficiency_uses_time(self):
        groups = [_group("a", avg_score=0.6, avg_duration_sec=3.0)]
        ranked = rank_groups_by_efficiency(groups)
        assert ranked[0].time_efficiency == pytest.approx(0.6 / 3.0)

    def test_tool_efficiency_also_computed(self):
        groups = [_group("a", avg_score=0.8, avg_tool_calls=4.0)]
        ranked = rank_groups_by_efficiency(groups)
        assert ranked[0].tool_efficiency == pytest.approx(0.8 / 4.0)

    def test_empty_list(self):
        assert rank_groups_by_efficiency([]) == []


# ---------------------------------------------------------------------------
# rank_groups_by_average_score
# ---------------------------------------------------------------------------

class TestRankGroupsByAverageScore:
    def test_descending_order(self):
        groups = [
            _group("a", avg_score=0.3),
            _group("b", avg_score=0.9),
            _group("c", avg_score=0.6),
        ]
        ranked = rank_groups_by_average_score(groups)
        assert ranked[0].group.value == "b"
        assert ranked[1].group.value == "c"
        assert ranked[2].group.value == "a"

    def test_none_score_last(self):
        groups = [_group("a", avg_score=None), _group("b", avg_score=0.5)]
        ranked = rank_groups_by_average_score(groups)
        assert ranked[-1].group.value == "a"

    def test_rank_one_is_highest(self):
        groups = [_group("lo", avg_score=0.2), _group("hi", avg_score=0.8)]
        ranked = rank_groups_by_average_score(groups)
        assert ranked[0].rank == 1
        assert ranked[0].group.value == "hi"


# ---------------------------------------------------------------------------
# analyze_experiment (integration with fixture dirs)
# ---------------------------------------------------------------------------

class TestAnalyzeExperiment:
    def test_loads_runs_from_fixture_dir(self, tmp_path):
        import json
        from benchmark.runner.models import RunResult, RunStatus, ExecutionMode
        from benchmark.validation.models import ValidationStatus

        now = "2026-05-16T12:00:00Z"
        for i, (rid, score, status) in enumerate([
            ("r1", 0.9, RunStatus.PASSED),
            ("r2", 0.5, RunStatus.FAILED),
        ]):
            d = tmp_path / rid
            d.mkdir()
            rr = RunResult(
                run_id=rid, task_id=f"task_{i}",
                status=status, execution_mode=ExecutionMode.AGENT_MCP,
                validation_result_path=None, scene_snapshot_path=None,
                artifacts_dir=d,
                total_score=score,
                overall_status=ValidationStatus.PASSED.value if status == RunStatus.PASSED else ValidationStatus.FAILED.value,
                started_at=now, finished_at=now, duration_sec=float(i * 5 + 5),
                summary={"strategy": "react", "agent_id": "agent_a"},
            )
            (d / "run_result.json").write_text(rr.model_dump_json())

        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            exp = analyze_experiment(tmp_path)

        assert exp.summary.total_runs == 2
        assert exp.summary.successful_runs == 1
        assert exp.summary.failed_runs == 1
