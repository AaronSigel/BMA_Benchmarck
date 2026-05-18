"""Tests for benchmark.analysis.models — serialization round-trips."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.analysis.models import (
    ComparisonDimension,
    ComparisonGroup,
    ComparisonReport,
    ErrorCategory,
    ErrorRecord,
    ExperimentAnalysisResult,
    ExperimentSummary,
    RankedGroup,
    RankedRun,
    RunAnalysisResult,
    ToolCallMetric,
    ValidationMetric,
)

FIXTURES = Path(__file__).parent / "fixtures" / "analysis"


# ---------------------------------------------------------------------------
# RunAnalysisResult
# ---------------------------------------------------------------------------

class TestRunAnalysisResult:
    def _make(self, **kw) -> RunAnalysisResult:
        defaults = dict(
            run_id="r1",
            task_id="task_easy",
            agent_id="agent_a",
            strategy="react",
        )
        defaults.update(kw)
        return RunAnalysisResult(**defaults)

    def test_minimal_construction(self):
        r = self._make()
        assert r.run_id == "r1"
        assert r.total_score is None
        assert r.success is None
        assert r.tool_call_count == 0
        assert r.metrics == {}

    def test_full_construction(self):
        r = self._make(
            model="claude-3-5-sonnet",
            mcp_profile="minimal",
            total_score=0.85,
            success=True,
            duration_sec=12.5,
            tool_call_count=5,
            llm_call_count=3,
            error_count=1,
            metrics={"error.tool_runtime_error": 1, "object_score": 0.9},
            issues=[{"code": "object_missing", "message": "Cube not found"}],
            artifacts=["/tmp/run1/agent_trace.json"],
        )
        assert r.total_score == 0.85
        assert r.model == "claude-3-5-sonnet"
        assert r.metrics["object_score"] == 0.9

    def test_json_round_trip(self):
        r = self._make(total_score=0.75, success=True, duration_sec=5.0)
        serialized = r.model_dump_json()
        restored = RunAnalysisResult.model_validate_json(serialized)
        assert restored.run_id == r.run_id
        assert restored.total_score == r.total_score
        assert restored.success == r.success

    def test_model_dump_and_reconstruct(self):
        r = self._make(metrics={"foo": 1}, artifacts=["a", "b"])
        data = r.model_dump()
        r2 = RunAnalysisResult(**data)
        assert r2.metrics == {"foo": 1}
        assert r2.artifacts == ["a", "b"]

    def test_score_bounds(self):
        with pytest.raises(Exception):
            self._make(total_score=1.5)
        with pytest.raises(Exception):
            self._make(total_score=-0.1)

    def test_counter_fields_non_negative(self):
        with pytest.raises(Exception):
            self._make(tool_call_count=-1)
        with pytest.raises(Exception):
            self._make(error_count=-1)

    def test_from_fixture_direct_success(self):
        raw = (FIXTURES / "agent_trace_direct_success.json").read_text()
        from benchmark.agent.models import AgentTrace
        from benchmark.analysis.trace_reader import RunArtifactBundle
        from benchmark.analysis.run_analysis import analyze_run
        trace = AgentTrace.model_validate_json(raw)
        result = analyze_run(RunArtifactBundle(run_dir=FIXTURES, agent_trace=trace))
        assert result.tool_call_count == 1
        assert result.llm_call_count == 1
        assert result.strategy == "direct_tool_calling"

    def test_from_fixture_tool_error(self):
        raw = (FIXTURES / "agent_trace_tool_error.json").read_text()
        from benchmark.agent.models import AgentTrace
        from benchmark.analysis.trace_reader import RunArtifactBundle
        from benchmark.analysis.run_analysis import analyze_run
        trace = AgentTrace.model_validate_json(raw)
        result = analyze_run(RunArtifactBundle(run_dir=FIXTURES, agent_trace=trace))
        assert result.error_count >= 2
        assert "error.tool_runtime_error" in result.metrics or "error.tool_disabled" in result.metrics

    def test_mcp_profile_from_run_result_execution_summary(self):
        from benchmark.analysis.run_analysis import analyze_run
        from benchmark.analysis.trace_reader import RunArtifactBundle
        from benchmark.runner.models import ExecutionMode, RunResult, RunStatus

        run_result = RunResult(
            run_id="pilot__task__agent__no_python__r1",
            task_id="task",
            status=RunStatus.ERROR,
            execution_mode=ExecutionMode.AGENT_MCP,
            validation_result_path=None,
            scene_snapshot_path=None,
            artifacts_dir=Path("artifacts"),
            total_score=None,
            overall_status=None,
            started_at="2026-05-16T10:00:00Z",
            finished_at="2026-05-16T10:00:01Z",
            duration_sec=1.0,
            summary={"execution": {"mcp_profile": "no_python"}},
        )

        result = analyze_run(RunArtifactBundle(run_dir=FIXTURES, run_result=run_result))

        assert result.mcp_profile == "no_python"

    def test_nested_agent_trace_is_listed_as_artifact(self, tmp_path: Path):
        from benchmark.analysis.run_analysis import analyze_run
        from benchmark.analysis.trace_reader import RunArtifactBundle
        from benchmark.runner.models import ExecutionMode, RunResult, RunStatus

        nested = tmp_path / "agent_runs" / "agent1"
        nested.mkdir(parents=True)
        (nested / "agent_trace.json").write_text("{}", encoding="utf-8")
        run_result = RunResult(
            run_id="r1",
            task_id="task",
            status=RunStatus.PASSED,
            execution_mode=ExecutionMode.AGENT_MCP,
            validation_result_path=None,
            scene_snapshot_path=None,
            artifacts_dir=tmp_path,
            total_score=None,
            overall_status=None,
            started_at="2026-05-16T10:00:00Z",
            finished_at="2026-05-16T10:00:01Z",
        )

        result = analyze_run(RunArtifactBundle(run_dir=tmp_path, run_result=run_result))

        assert str(nested / "agent_trace.json") in result.artifacts


# ---------------------------------------------------------------------------
# ExperimentSummary
# ---------------------------------------------------------------------------

class TestExperimentSummary:
    def test_defaults(self):
        s = ExperimentSummary()
        assert s.total_runs == 0
        assert s.average_scene_score is None
        assert s.most_common_errors == []

    def test_construction(self):
        s = ExperimentSummary(
            total_runs=10,
            successful_runs=7,
            failed_runs=2,
            error_runs=1,
            average_scene_score=0.82,
            best_run="r1",
            worst_run="r9",
            most_common_errors=[("tool_runtime_error", 5), ("unknown_error", 2)],
        )
        assert s.successful_runs == 7
        assert s.most_common_errors[0] == ("tool_runtime_error", 5)

    def test_json_round_trip(self):
        s = ExperimentSummary(total_runs=3, average_scene_score=0.5)
        s2 = ExperimentSummary.model_validate_json(s.model_dump_json())
        assert s2.total_runs == 3
        assert s2.average_scene_score == 0.5


# ---------------------------------------------------------------------------
# ExperimentAnalysisResult
# ---------------------------------------------------------------------------

class TestExperimentAnalysisResult:
    def _run(self, run_id: str = "r1", **kw) -> RunAnalysisResult:
        return RunAnalysisResult(run_id=run_id, task_id="t", agent_id="a", strategy="react", **kw)

    def test_computed_properties_empty(self):
        exp = ExperimentAnalysisResult(experiment_id="e1")
        assert exp.total_runs == 0
        assert exp.passed_runs == 0
        assert exp.avg_score is None

    def test_computed_properties_with_runs(self):
        runs = [
            self._run("r1", success=True, total_score=0.8),
            self._run("r2", success=True, total_score=0.6),
            self._run("r3", success=False, total_score=0.2),
        ]
        exp = ExperimentAnalysisResult(experiment_id="e1", runs=runs)
        assert exp.total_runs == 3
        assert exp.passed_runs == 2
        assert abs(exp.avg_score - (0.8 + 0.6 + 0.2) / 3) < 1e-6

    def test_json_round_trip(self):
        runs = [self._run(total_score=0.9, success=True)]
        summary = ExperimentSummary(total_runs=1, successful_runs=1)
        exp = ExperimentAnalysisResult(
            experiment_id="exp_42",
            runs=runs,
            summary=summary,
            metadata={"version": "1.0"},
        )
        raw = exp.model_dump_json()
        exp2 = ExperimentAnalysisResult.model_validate_json(raw)
        assert exp2.experiment_id == "exp_42"
        assert exp2.total_runs == 1
        assert exp2.summary.successful_runs == 1
        assert exp2.metadata["version"] == "1.0"

    def test_from_fixture_mixed(self):
        from benchmark.runner.models import ExperimentResult
        raw = (FIXTURES / "experiment_result_mixed.json").read_text()
        exp_result = ExperimentResult.model_validate_json(raw)
        assert len(exp_result.runs) == 4


# ---------------------------------------------------------------------------
# ComparisonGroup / ComparisonReport
# ---------------------------------------------------------------------------

class TestComparisonModels:
    def test_comparison_group_defaults(self):
        g = ComparisonGroup(
            dimension=ComparisonDimension.STRATEGY,
            value="react",
            run_count=5,
        )
        assert g.avg_score is None
        assert g.success_rate is None

    def test_comparison_report_empty_groups(self):
        report = ComparisonReport(dimension=ComparisonDimension.MODEL)
        assert report.groups == []

    def test_comparison_report_round_trip(self):
        g = ComparisonGroup(
            dimension=ComparisonDimension.STRATEGY,
            value="direct",
            run_count=3,
            avg_score=0.7,
            success_rate=0.67,
        )
        report = ComparisonReport(dimension=ComparisonDimension.STRATEGY, groups=[g])
        raw = report.model_dump_json()
        report2 = ComparisonReport.model_validate_json(raw)
        assert report2.groups[0].value == "direct"
        assert report2.groups[0].avg_score == 0.7


# ---------------------------------------------------------------------------
# RankedRun / RankedGroup
# ---------------------------------------------------------------------------

class TestRankedModels:
    def test_ranked_run_construction(self):
        run = RunAnalysisResult(run_id="r1", task_id="t", agent_id="a", strategy="react")
        rr = RankedRun(rank=1, run=run, score_used=0.9, time_efficiency=0.1, tool_efficiency=0.2)
        assert rr.rank == 1
        assert rr.score_used == 0.9

    def test_ranked_run_rank_ge_1(self):
        run = RunAnalysisResult(run_id="r1", task_id="t", agent_id="a", strategy="react")
        with pytest.raises(Exception):
            RankedRun(rank=0, run=run)

    def test_ranked_group_construction(self):
        g = ComparisonGroup(
            dimension=ComparisonDimension.STRATEGY,
            value="react",
            run_count=2,
        )
        rg = RankedGroup(rank=1, group=g, score_used=0.5)
        assert rg.rank == 1
        assert rg.group.value == "react"


# ---------------------------------------------------------------------------
# Supporting models
# ---------------------------------------------------------------------------

class TestSupportingModels:
    def test_tool_call_metric(self):
        m = ToolCallMetric(
            tool_name="add_object",
            total_calls=5,
            succeeded=4,
            failed=1,
            success_rate=0.8,
            avg_duration_sec=0.3,
            total_duration_sec=1.5,
        )
        assert m.success_rate == 0.8

    def test_validation_metric(self):
        v = ValidationMetric(
            validator_name="object_validator",
            score=1.0,
            status="passed",
            issue_count=0,
        )
        assert v.score == 1.0

    def test_error_record(self):
        e = ErrorRecord(
            run_id="r1",
            task_id="t1",
            step_index=2,
            category=ErrorCategory.TOOL_RUNTIME_ERROR,
            message="execution failed",
            tool_name="add_object",
        )
        assert e.category == ErrorCategory.TOOL_RUNTIME_ERROR
        assert e.tool_name == "add_object"

    def test_error_category_values(self):
        assert ErrorCategory.TOOL_DISABLED.value == "tool_disabled"
        assert ErrorCategory.AGENT_STEP_LIMIT.value == "agent_step_limit"
        assert ErrorCategory.UNKNOWN_ERROR.value == "unknown_error"
