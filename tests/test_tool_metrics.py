"""Tests for benchmark.analysis.tool_metrics."""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.agent.models import AgentStep, AgentStepType, AgentStrategyName, AgentTrace
from benchmark.analysis.tool_metrics import (
    ToolCallSummary,
    compute_tool_summary,
    extract_tool_metrics,
    tool_call_count,
    tool_failure_count,
)

FIXTURES = Path(__file__).parent / "fixtures" / "analysis"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = "2026-05-16T10:00:00Z"


def _trace(*steps: AgentStep, run_id: str = "r1", task_id: str = "t1") -> AgentTrace:
    return AgentTrace(
        run_id=run_id,
        task_id=task_id,
        agent_id="agent",
        strategy=AgentStrategyName.REACT,
        model="mock",
        steps=list(steps),
        started_at=_NOW,
        finished_at=_NOW,
        duration_sec=1.0,
    )


def _tool_step(
    tool_name: str,
    *,
    index: int = 0,
    error: str | None = None,
    duration_sec: float | None = None,
) -> AgentStep:
    return AgentStep(
        step_index=index,
        step_type=AgentStepType.TOOL_CALL,
        tool_name=tool_name,
        tool_arguments={"arg": "val"},
        observation=None if error else "ok",
        error=error,
        duration_sec=duration_sec,
        started_at=_NOW,
        finished_at=_NOW,
    )


def _llm_step(index: int = 0) -> AgentStep:
    return AgentStep(
        step_index=index,
        step_type=AgentStepType.LLM_CALL,
        tool_arguments={},
        started_at=_NOW,
        finished_at=_NOW,
    )


# ---------------------------------------------------------------------------
# Empty / no-tool traces
# ---------------------------------------------------------------------------

class TestEmptyTrace:
    def test_empty_trace_returns_zeros(self):
        summary = compute_tool_summary(_trace())
        assert summary.tool_call_count == 0
        assert summary.unique_tool_count == 0
        assert summary.invalid_tool_call_count == 0
        assert summary.disabled_tool_call_count == 0
        assert summary.python_tool_call_count == 0
        assert summary.inspection_tool_count == 0
        assert summary.mutation_tool_count == 0
        assert summary.average_tool_duration_sec is None

    def test_llm_only_trace_no_tool_counts(self):
        trace = _trace(_llm_step(0), _llm_step(1))
        summary = compute_tool_summary(trace)
        assert summary.tool_call_count == 0
        assert summary.trajectory_length == 2


# ---------------------------------------------------------------------------
# tool_call_count
# ---------------------------------------------------------------------------

class TestToolCallCount:
    def test_counts_all_tool_steps(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0),
            _tool_step("bma_set_transform", index=1),
            _tool_step("bma_create_object", index=2),
        )
        summary = compute_tool_summary(trace)
        assert summary.tool_call_count == 3

    def test_excludes_non_tool_steps(self):
        trace = _trace(
            _llm_step(0),
            _tool_step("bma_create_object", index=1),
            _llm_step(2),
        )
        summary = compute_tool_summary(trace)
        assert summary.tool_call_count == 1

    def test_convenience_function_total(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0),
            _tool_step("bma_set_transform", index=1),
        )
        assert tool_call_count(trace) == 2

    def test_convenience_function_by_name(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0),
            _tool_step("bma_create_object", index=1),
            _tool_step("bma_set_transform", index=2),
        )
        assert tool_call_count(trace, "bma_create_object") == 2
        assert tool_call_count(trace, "bma_set_transform") == 1
        assert tool_call_count(trace, "nonexistent") == 0

    def test_fixture_direct_success(self):
        from benchmark.agent.models import AgentTrace as AT
        trace = AT.model_validate_json((FIXTURES / "agent_trace_direct_success.json").read_text())
        summary = compute_tool_summary(trace)
        assert summary.tool_call_count == 1

    def test_fixture_react_success(self):
        from benchmark.agent.models import AgentTrace as AT
        trace = AT.model_validate_json((FIXTURES / "agent_trace_react_success.json").read_text())
        summary = compute_tool_summary(trace)
        assert summary.tool_call_count == 2


# ---------------------------------------------------------------------------
# unique_tool_count
# ---------------------------------------------------------------------------

class TestUniqueToolCount:
    def test_single_tool_unique_one(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0),
            _tool_step("bma_create_object", index=1),
            _tool_step("bma_create_object", index=2),
        )
        summary = compute_tool_summary(trace)
        assert summary.unique_tool_count == 1

    def test_three_different_tools(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0),
            _tool_step("bma_set_transform", index=1),
            _tool_step("get_scene_info", index=2),
        )
        summary = compute_tool_summary(trace)
        assert summary.unique_tool_count == 3

    def test_two_unique_from_five_calls(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0),
            _tool_step("bma_set_material", index=1),
            _tool_step("bma_create_object", index=2),
            _tool_step("bma_set_material", index=3),
            _tool_step("bma_create_object", index=4),
        )
        summary = compute_tool_summary(trace)
        assert summary.unique_tool_count == 2


# ---------------------------------------------------------------------------
# invalid_tool_call_count / disabled_tool_call_count
# ---------------------------------------------------------------------------

class TestErrorCounts:
    def test_no_errors(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0),
            _tool_step("bma_set_transform", index=1),
        )
        summary = compute_tool_summary(trace)
        assert summary.invalid_tool_call_count == 0
        assert summary.disabled_tool_call_count == 0
        assert summary.tool_error_count == 0

    def test_runtime_error_counts_as_invalid(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0,
                        error="ToolInvocationError: execution failed"),
        )
        summary = compute_tool_summary(trace)
        assert summary.invalid_tool_call_count == 1
        assert summary.disabled_tool_call_count == 0
        assert summary.tool_error_count == 1

    def test_disabled_error_counts_separately(self):
        trace = _trace(
            _tool_step("execute_blender_code", index=0,
                        error="ToolDisabledError: tool is disabled in current profile"),
        )
        summary = compute_tool_summary(trace)
        assert summary.disabled_tool_call_count == 1
        assert summary.invalid_tool_call_count == 0
        assert summary.tool_error_count == 1

    def test_not_allowed_message_is_disabled(self):
        trace = _trace(
            _tool_step("some_tool", index=0, error="tool not allowed"),
        )
        summary = compute_tool_summary(trace)
        assert summary.disabled_tool_call_count == 1

    def test_mixed_errors(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0, error="execution error"),
            _tool_step("execute_blender_code", index=1, error="is disabled"),
            _tool_step("bma_set_transform", index=2),
        )
        summary = compute_tool_summary(trace)
        assert summary.tool_error_count == 2
        assert summary.disabled_tool_call_count == 1
        assert summary.invalid_tool_call_count == 1

    def test_fixture_tool_error_trace(self):
        from benchmark.agent.models import AgentTrace as AT
        trace = AT.model_validate_json((FIXTURES / "agent_trace_tool_error.json").read_text())
        summary = compute_tool_summary(trace)
        assert summary.tool_error_count == 2
        assert summary.disabled_tool_call_count == 1
        assert summary.invalid_tool_call_count == 1

    def test_tool_failure_count_convenience(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0, error="boom"),
            _tool_step("bma_create_object", index=1),
            _tool_step("bma_set_transform", index=2, error="bust"),
        )
        assert tool_failure_count(trace) == 2
        assert tool_failure_count(trace, "bma_create_object") == 1
        assert tool_failure_count(trace, "bma_set_transform") == 1


# ---------------------------------------------------------------------------
# python_tool_call_count
# ---------------------------------------------------------------------------

class TestPythonToolCount:
    def test_execute_blender_code_is_python(self):
        trace = _trace(
            _tool_step("execute_blender_code", index=0),
            _tool_step("execute_blender_code", index=1),
        )
        summary = compute_tool_summary(trace)
        assert summary.python_tool_call_count == 2

    def test_non_python_not_counted(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0),
            _tool_step("get_scene_info", index=1),
        )
        summary = compute_tool_summary(trace)
        assert summary.python_tool_call_count == 0

    def test_mixed_python_and_others(self):
        trace = _trace(
            _tool_step("execute_blender_code", index=0),
            _tool_step("bma_create_object", index=1),
            _tool_step("execute_blender_code", index=2),
        )
        summary = compute_tool_summary(trace)
        assert summary.python_tool_call_count == 2
        assert summary.mutation_tool_count == 1


# ---------------------------------------------------------------------------
# inspection_tool_count
# ---------------------------------------------------------------------------

class TestInspectionToolCount:
    def test_inspection_tools_counted(self):
        # get_scene_info and bma_get_scene_info are both inspection
        trace = _trace(
            _tool_step("get_scene_info", index=0),
            _tool_step("bma_get_scene_info", index=1),
            _tool_step("get_viewport_screenshot", index=2),
        )
        summary = compute_tool_summary(trace)
        assert summary.inspection_tool_count == 3

    def test_inspection_not_counted_as_mutation(self):
        trace = _trace(_tool_step("get_scene_info", index=0))
        summary = compute_tool_summary(trace)
        assert summary.inspection_tool_count == 1
        assert summary.mutation_tool_count == 0

    def test_fixture_react_uses_inspection(self):
        from benchmark.agent.models import AgentTrace as AT
        trace = AT.model_validate_json((FIXTURES / "agent_trace_react_success.json").read_text())
        summary = compute_tool_summary(trace)
        # react trace calls get_scene_objects (inspection category) and set_material_property
        assert summary.tool_call_count == 2


# ---------------------------------------------------------------------------
# mutation_tool_count
# ---------------------------------------------------------------------------

class TestMutationToolCount:
    def test_object_tool_is_mutation(self):
        trace = _trace(_tool_step("bma_create_object", index=0))
        summary = compute_tool_summary(trace)
        assert summary.mutation_tool_count == 1

    def test_transform_tool_is_mutation(self):
        trace = _trace(_tool_step("bma_set_transform", index=0))
        summary = compute_tool_summary(trace)
        assert summary.mutation_tool_count == 1

    def test_material_tool_is_mutation(self):
        trace = _trace(_tool_step("bma_set_material", index=0))
        summary = compute_tool_summary(trace)
        assert summary.mutation_tool_count == 1

    def test_light_tool_is_mutation(self):
        trace = _trace(_tool_step("bma_create_light", index=0))
        summary = compute_tool_summary(trace)
        assert summary.mutation_tool_count == 1

    def test_camera_tool_is_mutation(self):
        trace = _trace(_tool_step("bma_create_camera", index=0))
        summary = compute_tool_summary(trace)
        assert summary.mutation_tool_count == 1

    def test_export_tool_is_mutation(self):
        trace = _trace(_tool_step("bma_export_scene", index=0))
        summary = compute_tool_summary(trace)
        assert summary.mutation_tool_count == 1

    def test_multiple_mutation_tools(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0),
            _tool_step("bma_set_transform", index=1),
            _tool_step("bma_set_material", index=2),
            _tool_step("bma_create_light", index=3),
        )
        summary = compute_tool_summary(trace)
        assert summary.mutation_tool_count == 4

    def test_mutation_excludes_inspection_and_python(self):
        trace = _trace(
            _tool_step("get_scene_info", index=0),
            _tool_step("execute_blender_code", index=1),
            _tool_step("bma_create_object", index=2),
        )
        summary = compute_tool_summary(trace)
        assert summary.mutation_tool_count == 1
        assert summary.inspection_tool_count == 1
        assert summary.python_tool_call_count == 1


# ---------------------------------------------------------------------------
# average_tool_duration_sec
# ---------------------------------------------------------------------------

class TestAverageToolDuration:
    def test_none_when_no_durations(self):
        trace = _trace(_tool_step("bma_create_object", index=0, duration_sec=None))
        summary = compute_tool_summary(trace)
        assert summary.average_tool_duration_sec is None

    def test_single_step_with_duration(self):
        trace = _trace(_tool_step("bma_create_object", index=0, duration_sec=2.0))
        summary = compute_tool_summary(trace)
        assert summary.average_tool_duration_sec == pytest.approx(2.0)

    def test_average_of_multiple_durations(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0, duration_sec=1.0),
            _tool_step("bma_set_transform", index=1, duration_sec=3.0),
        )
        summary = compute_tool_summary(trace)
        assert summary.average_tool_duration_sec == pytest.approx(2.0)

    def test_partial_durations_only_counts_present(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0, duration_sec=4.0),
            _tool_step("bma_set_transform", index=1, duration_sec=None),
            _tool_step("bma_set_material", index=2, duration_sec=2.0),
        )
        summary = compute_tool_summary(trace)
        assert summary.average_tool_duration_sec == pytest.approx(3.0)

    def test_zero_duration_included(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0, duration_sec=0.0),
            _tool_step("bma_set_transform", index=1, duration_sec=2.0),
        )
        summary = compute_tool_summary(trace)
        assert summary.average_tool_duration_sec == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# tool repetition count
# ---------------------------------------------------------------------------

class TestToolRepetition:
    def test_no_repetition(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0),
            _tool_step("bma_set_transform", index=1),
        )
        summary = compute_tool_summary(trace)
        assert summary.tool_repetition_count == 0

    def test_consecutive_same_tool_is_repetition(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0),
            _tool_step("bma_create_object", index=1),
            _tool_step("bma_create_object", index=2),
        )
        summary = compute_tool_summary(trace)
        assert summary.tool_repetition_count == 2

    def test_non_consecutive_same_tool_not_repetition(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0),
            _tool_step("bma_set_transform", index=1),
            _tool_step("bma_create_object", index=2),
        )
        summary = compute_tool_summary(trace)
        assert summary.tool_repetition_count == 0


# ---------------------------------------------------------------------------
# extract_tool_metrics (per-tool breakdown)
# ---------------------------------------------------------------------------

class TestExtractToolMetrics:
    def test_empty_trace_returns_empty_list(self):
        assert extract_tool_metrics(_trace()) == []

    def test_single_tool_all_succeeded(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0, duration_sec=1.0),
            _tool_step("bma_create_object", index=1, duration_sec=3.0),
        )
        metrics = extract_tool_metrics(trace)
        assert len(metrics) == 1
        m = metrics[0]
        assert m.tool_name == "bma_create_object"
        assert m.total_calls == 2
        assert m.succeeded == 2
        assert m.failed == 0
        assert m.success_rate == 1.0
        assert m.avg_duration_sec == pytest.approx(2.0)
        assert m.total_duration_sec == pytest.approx(4.0)

    def test_tool_with_failures(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0),
            _tool_step("bma_create_object", index=1, error="boom"),
            _tool_step("bma_create_object", index=2),
        )
        metrics = extract_tool_metrics(trace)
        m = metrics[0]
        assert m.total_calls == 3
        assert m.succeeded == 2
        assert m.failed == 1
        assert m.success_rate == pytest.approx(2 / 3)

    def test_multiple_tools_sorted_alphabetically(self):
        trace = _trace(
            _tool_step("get_scene_info", index=0),
            _tool_step("bma_create_object", index=1),
            _tool_step("execute_blender_code", index=2),
        )
        metrics = extract_tool_metrics(trace)
        names = [m.tool_name for m in metrics]
        assert names == sorted(names)

    def test_success_rate_zero_when_all_fail(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0, error="fail"),
        )
        metrics = extract_tool_metrics(trace)
        assert metrics[0].success_rate == 0.0

    def test_no_duration_when_not_recorded(self):
        trace = _trace(
            _tool_step("bma_create_object", index=0, duration_sec=None),
        )
        metrics = extract_tool_metrics(trace)
        assert metrics[0].avg_duration_sec is None
        assert metrics[0].total_duration_sec == 0.0
