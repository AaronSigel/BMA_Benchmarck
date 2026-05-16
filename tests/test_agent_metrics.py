"""Tests for benchmark.analysis.agent_metrics."""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.agent.models import AgentStep, AgentStepType, AgentTrace, AgentStrategy
from benchmark.analysis.agent_metrics import (
    AgentMetricsSummary,
    _aggregate_token_usage,
    compute_agent_summary,
    extract_agent_metrics,
)

FIXTURES = Path(__file__).parent / "fixtures" / "analysis"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = "2026-05-16T10:00:00Z"


def _trace(
    *steps: AgentStep,
    run_id: str = "r1",
    task_id: str = "t1",
    error: str | None = None,
    duration_sec: float | None = 5.0,
    metadata: dict | None = None,
) -> AgentTrace:
    return AgentTrace(
        run_id=run_id,
        task_id=task_id,
        agent_id="agent",
        strategy=AgentStrategy.REACT,
        model="mock",
        steps=list(steps),
        error=error,
        started_at=_NOW,
        finished_at=_NOW,
        duration_sec=duration_sec,
        metadata=metadata or {},
    )


def _step(
    step_type: AgentStepType,
    *,
    index: int = 0,
    error: str | None = None,
    duration_sec: float | None = None,
    raw_llm_response: dict | None = None,
    tool_name: str | None = None,
) -> AgentStep:
    return AgentStep(
        step_index=index,
        step_type=step_type,
        tool_name=tool_name,
        tool_arguments={} if step_type == AgentStepType.TOOL_CALL else {},
        error=error,
        duration_sec=duration_sec,
        raw_llm_response=raw_llm_response,
        started_at=_NOW,
        finished_at=_NOW,
    )


def _llm(index: int = 0, duration_sec: float | None = None,
         prompt: int | None = None, completion: int | None = None) -> AgentStep:
    raw = None
    if prompt is not None or completion is not None:
        usage: dict = {}
        if prompt is not None:
            usage["prompt_tokens"] = prompt
        if completion is not None:
            usage["completion_tokens"] = completion
        raw = {"usage": usage}
    return _step(AgentStepType.LLM_CALL, index=index,
                 duration_sec=duration_sec, raw_llm_response=raw)


def _tool(index: int = 0, name: str = "bma_create_object",
          error: str | None = None, duration_sec: float | None = None) -> AgentStep:
    return AgentStep(
        step_index=index,
        step_type=AgentStepType.TOOL_CALL,
        tool_name=name,
        tool_arguments={"x": 1},
        error=error,
        duration_sec=duration_sec,
        started_at=_NOW,
        finished_at=_NOW,
    )


def _plan(index: int = 0, duration_sec: float | None = None) -> AgentStep:
    return _step(AgentStepType.PLAN, index=index, duration_sec=duration_sec)


def _obs(index: int = 0, duration_sec: float | None = None) -> AgentStep:
    return _step(AgentStepType.OBSERVATION, index=index, duration_sec=duration_sec)


def _final(index: int = 0) -> AgentStep:
    return _step(AgentStepType.FINAL, index=index)


# ---------------------------------------------------------------------------
# Empty trace baseline
# ---------------------------------------------------------------------------

class TestEmptyTrace:
    def test_all_zeros_on_empty_trace(self):
        s = compute_agent_summary(_trace())
        assert s.llm_call_count == 0
        assert s.planning_step_count == 0
        assert s.observation_count == 0
        assert s.error_count == 0
        assert s.final_step_present is False
        assert s.step_limit_reached is False
        assert s.average_step_duration_sec is None
        assert s.max_step_duration_sec is None
        assert s.prompt_tokens is None
        assert s.completion_tokens is None
        assert s.total_tokens is None
        assert s.estimated_cost is None


# ---------------------------------------------------------------------------
# llm_call_count
# ---------------------------------------------------------------------------

class TestLlmCallCount:
    def test_counts_llm_steps(self):
        s = compute_agent_summary(_trace(_llm(0), _llm(1), _llm(2)))
        assert s.llm_call_count == 3

    def test_excludes_non_llm_steps(self):
        s = compute_agent_summary(_trace(_llm(0), _tool(1), _plan(2), _obs(3)))
        assert s.llm_call_count == 1

    def test_fixture_direct_success(self):
        from benchmark.agent.models import AgentTrace as AT
        t = AT.model_validate_json((FIXTURES / "agent_trace_direct_success.json").read_text())
        s = compute_agent_summary(t)
        assert s.llm_call_count == 1

    def test_fixture_react_success(self):
        from benchmark.agent.models import AgentTrace as AT
        t = AT.model_validate_json((FIXTURES / "agent_trace_react_success.json").read_text())
        s = compute_agent_summary(t)
        assert s.llm_call_count == 2


# ---------------------------------------------------------------------------
# planning_step_count
# ---------------------------------------------------------------------------

class TestPlanningStepCount:
    def test_counts_plan_steps(self):
        s = compute_agent_summary(_trace(_plan(0), _plan(1)))
        assert s.planning_step_count == 2

    def test_zero_when_no_plan_steps(self):
        s = compute_agent_summary(_trace(_llm(0), _tool(1)))
        assert s.planning_step_count == 0

    def test_fixture_plan_execute(self):
        from benchmark.agent.models import AgentTrace as AT
        t = AT.model_validate_json(
            (FIXTURES / "agent_trace_plan_execute_success.json").read_text()
        )
        s = compute_agent_summary(t)
        assert s.planning_step_count == 1

    def test_plan_not_counted_as_llm(self):
        s = compute_agent_summary(_trace(_plan(0)))
        assert s.planning_step_count == 1
        assert s.llm_call_count == 0


# ---------------------------------------------------------------------------
# observation_count
# ---------------------------------------------------------------------------

class TestObservationCount:
    def test_counts_observation_steps(self):
        s = compute_agent_summary(_trace(_obs(0), _obs(1), _obs(2)))
        assert s.observation_count == 3

    def test_zero_when_no_observations(self):
        s = compute_agent_summary(_trace(_llm(0), _tool(1)))
        assert s.observation_count == 0

    def test_fixture_react_has_observation(self):
        from benchmark.agent.models import AgentTrace as AT
        t = AT.model_validate_json((FIXTURES / "agent_trace_react_success.json").read_text())
        s = compute_agent_summary(t)
        assert s.observation_count == 1

    def test_observation_not_counted_as_llm_or_plan(self):
        s = compute_agent_summary(_trace(_obs(0)))
        assert s.observation_count == 1
        assert s.llm_call_count == 0
        assert s.planning_step_count == 0


# ---------------------------------------------------------------------------
# error_count
# ---------------------------------------------------------------------------

class TestErrorCount:
    def test_counts_steps_with_errors(self):
        s = compute_agent_summary(_trace(
            _tool(0, error="boom"),
            _tool(1),
            _tool(2, error="bust"),
        ))
        assert s.error_count == 2

    def test_zero_when_no_errors(self):
        s = compute_agent_summary(_trace(_llm(0), _tool(1)))
        assert s.error_count == 0

    def test_error_on_any_step_type(self):
        s = compute_agent_summary(_trace(
            _step(AgentStepType.LLM_CALL, index=0, error="parse error"),
            _tool(1, error="exec error"),
        ))
        assert s.error_count == 2

    def test_fixture_tool_error_trace(self):
        from benchmark.agent.models import AgentTrace as AT
        t = AT.model_validate_json((FIXTURES / "agent_trace_tool_error.json").read_text())
        s = compute_agent_summary(t)
        assert s.error_count == 2


# ---------------------------------------------------------------------------
# step_limit_reached
# ---------------------------------------------------------------------------

class TestStepLimitReached:
    def test_false_when_no_limit(self):
        s = compute_agent_summary(_trace(_llm(0), _tool(1)))
        assert s.step_limit_reached is False

    def test_detected_from_trace_error(self):
        s = compute_agent_summary(_trace(error="AgentStepLimitError: max_steps reached"))
        assert s.step_limit_reached is True

    def test_detected_from_trace_error_step_limit_variant(self):
        s = compute_agent_summary(_trace(error="step_limit reached after 20 steps"))
        assert s.step_limit_reached is True

    def test_detected_from_last_step_error(self):
        s = compute_agent_summary(_trace(
            _llm(0),
            _step(AgentStepType.ERROR, index=1, error="max_steps exceeded"),
        ))
        assert s.step_limit_reached is True

    def test_not_triggered_by_other_errors(self):
        s = compute_agent_summary(_trace(
            _tool(0, error="execution failed"),
        ))
        assert s.step_limit_reached is False

    def test_fixture_error_run_result(self):
        from benchmark.runner.models import RunResult
        rr = RunResult.model_validate_json(
            (FIXTURES / "run_result_error.json").read_text()
        )
        # run_result_error has step-limit error message; verify it can be detected
        assert "max_steps" in (rr.error or "")


# ---------------------------------------------------------------------------
# final_step_present
# ---------------------------------------------------------------------------

class TestFinalStepPresent:
    def test_true_when_final_step_exists(self):
        s = compute_agent_summary(_trace(_llm(0), _final(1)))
        assert s.final_step_present is True

    def test_false_when_no_final_step(self):
        s = compute_agent_summary(_trace(_llm(0), _tool(1)))
        assert s.final_step_present is False

    def test_fixture_direct_success_has_final(self):
        from benchmark.agent.models import AgentTrace as AT
        t = AT.model_validate_json((FIXTURES / "agent_trace_direct_success.json").read_text())
        s = compute_agent_summary(t)
        assert s.final_step_present is True


# ---------------------------------------------------------------------------
# average_step_duration_sec
# ---------------------------------------------------------------------------

class TestAverageStepDuration:
    def test_none_when_no_durations(self):
        s = compute_agent_summary(_trace(_llm(0, duration_sec=None)))
        assert s.average_step_duration_sec is None

    def test_single_step(self):
        s = compute_agent_summary(_trace(_llm(0, duration_sec=4.0)))
        assert s.average_step_duration_sec == pytest.approx(4.0)

    def test_average_across_all_step_types(self):
        s = compute_agent_summary(_trace(
            _llm(0, duration_sec=2.0),
            _tool(1, duration_sec=4.0),
            _obs(2, duration_sec=0.0),
        ))
        assert s.average_step_duration_sec == pytest.approx(2.0)

    def test_max_step_duration(self):
        s = compute_agent_summary(_trace(
            _llm(0, duration_sec=1.0),
            _llm(1, duration_sec=5.0),
            _llm(2, duration_sec=3.0),
        ))
        assert s.max_step_duration_sec == pytest.approx(5.0)

    def test_partial_durations_only_counts_present(self):
        s = compute_agent_summary(_trace(
            _llm(0, duration_sec=6.0),
            _llm(1, duration_sec=None),
            _llm(2, duration_sec=2.0),
        ))
        assert s.average_step_duration_sec == pytest.approx(4.0)

    def test_fixture_has_duration_data(self):
        from benchmark.agent.models import AgentTrace as AT
        t = AT.model_validate_json((FIXTURES / "agent_trace_react_success.json").read_text())
        s = compute_agent_summary(t)
        assert s.average_step_duration_sec is not None
        assert s.average_step_duration_sec >= 0.0


# ---------------------------------------------------------------------------
# Token usage (optional)
# ---------------------------------------------------------------------------

class TestTokenUsage:
    def test_none_when_no_usage_reported(self):
        s = compute_agent_summary(_trace(_llm(0), _llm(1)))
        assert s.prompt_tokens is None
        assert s.completion_tokens is None
        assert s.total_tokens is None
        assert s.estimated_cost is None

    def test_sums_across_llm_steps(self):
        s = compute_agent_summary(_trace(
            _llm(0, prompt=100, completion=50),
            _llm(1, prompt=200, completion=80),
        ))
        assert s.prompt_tokens == 300
        assert s.completion_tokens == 130
        assert s.total_tokens == 430

    def test_estimated_cost_computed(self):
        s = compute_agent_summary(_trace(_llm(0, prompt=500_000, completion=500_000)))
        # 1M tokens × $2/M = $2.00
        assert s.total_tokens == 1_000_000
        assert s.estimated_cost == pytest.approx(2.0, rel=1e-3)

    def test_partial_usage_per_step(self):
        # One step with usage, one without
        step_with = _llm(0, prompt=100, completion=50)
        step_without = _llm(1)
        s = compute_agent_summary(_trace(step_with, step_without))
        assert s.prompt_tokens == 100
        assert s.completion_tokens == 50

    def test_fixture_direct_success_has_token_data(self):
        from benchmark.agent.models import AgentTrace as AT
        t = AT.model_validate_json((FIXTURES / "agent_trace_direct_success.json").read_text())
        s = compute_agent_summary(t)
        assert s.prompt_tokens == 120
        assert s.completion_tokens == 40
        assert s.total_tokens == 160
        assert s.estimated_cost == pytest.approx(160 * 2e-6)

    def test_fixture_react_success_aggregates_all_llm_steps(self):
        from benchmark.agent.models import AgentTrace as AT
        t = AT.model_validate_json((FIXTURES / "agent_trace_react_success.json").read_text())
        s = compute_agent_summary(t)
        # react fixture has two LLM calls: (200+60) + (280+50) = 590 total
        assert s.total_tokens == 590

    def test_aggregate_token_usage_helper_no_usage(self):
        trace = _trace(_llm(0))
        p, c, total = _aggregate_token_usage(trace)
        assert p is None and c is None and total is None

    def test_aggregate_token_usage_helper_with_usage(self):
        trace = _trace(_llm(0, prompt=10, completion=5))
        p, c, total = _aggregate_token_usage(trace)
        assert p == 10
        assert c == 5
        assert total == 15


# ---------------------------------------------------------------------------
# self_correction_attempts / tool_error_recovery_count
# ---------------------------------------------------------------------------

class TestControlFlow:
    def test_self_correction_detected(self):
        # error step → LLM step = self-correction
        s = compute_agent_summary(_trace(
            _tool(0, error="boom"),
            _llm(1),
        ))
        assert s.self_correction_attempts == 1

    def test_no_self_correction_without_followup_llm(self):
        s = compute_agent_summary(_trace(
            _tool(0, error="boom"),
            _tool(1),
        ))
        assert s.self_correction_attempts == 0

    def test_tool_error_recovery_detected(self):
        # tool error → next tool call = recovery
        s = compute_agent_summary(_trace(
            _tool(0, error="exec fail"),
            _tool(1),
        ))
        assert s.tool_error_recovery_count == 1

    def test_no_recovery_at_end_of_trace(self):
        s = compute_agent_summary(_trace(
            _tool(0, error="boom"),
        ))
        assert s.tool_error_recovery_count == 0

    def test_retry_count_from_metadata(self):
        s = compute_agent_summary(_trace(
            _llm(0),
            metadata={"retry_count": 3},
        ))
        assert s.retry_count == 3

    def test_retry_count_zero_by_default(self):
        s = compute_agent_summary(_trace(_llm(0)))
        assert s.retry_count == 0


# ---------------------------------------------------------------------------
# extract_agent_metrics (RunAnalysisResult builder)
# ---------------------------------------------------------------------------

class TestExtractAgentMetrics:
    def test_returns_run_analysis_result(self):
        from benchmark.analysis.models import RunAnalysisResult
        trace = _trace(_llm(0, prompt=100, completion=40), _tool(1))
        result = extract_agent_metrics(trace)
        assert isinstance(result, RunAnalysisResult)

    def test_identifiers_copied_from_trace(self):
        trace = _trace(_llm(0), run_id="my_run", task_id="my_task")
        result = extract_agent_metrics(trace)
        assert result.run_id == "my_run"
        assert result.task_id == "my_task"
        assert result.strategy == "react"

    def test_metrics_dict_populated(self):
        trace = _trace(_llm(0, prompt=50, completion=30), _tool(1))
        result = extract_agent_metrics(trace)
        assert "llm_call_count" in result.metrics
        assert "tool_call_count" in result.metrics
        assert result.metrics["llm_call_count"] == 1
        assert result.metrics["tool_call_count"] == 1

    def test_token_metrics_in_metrics_dict(self):
        trace = _trace(_llm(0, prompt=100, completion=50))
        result = extract_agent_metrics(trace)
        assert "prompt_tokens" in result.metrics
        assert result.metrics["prompt_tokens"] == 100
        assert "total_tokens" in result.metrics
        assert "estimated_cost" in result.metrics

    def test_no_token_metrics_when_absent(self):
        trace = _trace(_llm(0))
        result = extract_agent_metrics(trace)
        assert "prompt_tokens" not in result.metrics
        assert "estimated_cost" not in result.metrics

    def test_fixture_direct_full_pipeline(self):
        from benchmark.agent.models import AgentTrace as AT
        t = AT.model_validate_json((FIXTURES / "agent_trace_direct_success.json").read_text())
        result = extract_agent_metrics(t)
        assert result.llm_call_count == 1
        assert result.tool_call_count == 1
        assert result.success is True
        assert result.metrics.get("total_tokens") == 160
