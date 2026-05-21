"""Tests for benchmark.analysis.error_taxonomy."""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.agent.models import AgentStep, AgentStepType, AgentStrategyName, AgentTrace
from benchmark.analysis.error_taxonomy import (
    aggregate_errors,
    classify_trace_error,
    classify_validation_issue,
    extract_errors,
    summarize_errors,
)
from benchmark.analysis.models import ErrorCategory
from benchmark.analysis.trace_reader import RunArtifactBundle
from benchmark.validation.models import ValidationIssue, ValidationSeverity

FIXTURES = Path(__file__).parent / "fixtures" / "analysis"

_NOW = "2026-05-16T10:00:00Z"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step_with_error(error: str, index: int = 0,
                     step_type: AgentStepType = AgentStepType.TOOL_CALL) -> AgentStep:
    return AgentStep(
        step_index=index,
        step_type=step_type,
        tool_arguments={},
        error=error,
        started_at=_NOW,
        finished_at=_NOW,
    )


def _trace(*steps: AgentStep) -> AgentTrace:
    return AgentTrace(
        run_id="r1", task_id="t1", agent_id="a",
        strategy=AgentStrategyName.REACT, model="mock",
        steps=list(steps),
        started_at=_NOW, finished_at=_NOW, duration_sec=1.0,
    )


def _issue(code: str, message: str = "") -> ValidationIssue:
    return ValidationIssue(code=code, message=message, severity="error")


# ---------------------------------------------------------------------------
# classify_trace_error — issue code → error category
# ---------------------------------------------------------------------------

class TestClassifyTraceError:
    def test_tool_disabled_not_allowed(self):
        step = _step_with_error("tool not allowed in this profile")
        assert classify_trace_error(step) == ErrorCategory.TOOL_DISABLED

    def test_tool_disabled_is_disabled(self):
        step = _step_with_error("ToolDisabledError: tool is disabled")
        assert classify_trace_error(step) == ErrorCategory.TOOL_DISABLED

    def test_tool_unknown_not_found(self):
        step = _step_with_error("tool not found: add_invalid_tool")
        assert classify_trace_error(step) == ErrorCategory.TOOL_UNKNOWN

    def test_tool_unknown_unrecognised(self):
        step = _step_with_error("command unrecognised: do_something")
        assert classify_trace_error(step) == ErrorCategory.TOOL_UNKNOWN

    def test_agent_step_limit(self):
        step = _step_with_error("AgentStepLimitError: max_steps reached")
        assert classify_trace_error(step) == ErrorCategory.AGENT_STEP_LIMIT

    def test_agent_step_limit_variant(self):
        step = _step_with_error("step_limit reached after 20 iterations")
        assert classify_trace_error(step) == ErrorCategory.AGENT_STEP_LIMIT

    def test_llm_timeout(self):
        step = _step_with_error("AgentTimeoutError: request timed out")
        assert classify_trace_error(step) == ErrorCategory.LLM_TIMEOUT

    def test_llm_timeout_timed_out(self):
        step = _step_with_error("operation timed-out after 30s")
        assert classify_trace_error(step) == ErrorCategory.LLM_TIMEOUT

    def test_llm_parse_error(self):
        step = _step_with_error("failed to parse LLM response: json decode error")
        assert classify_trace_error(step) == ErrorCategory.LLM_PARSE_ERROR

    def test_llm_parse_error_invalid_json(self):
        step = _step_with_error("invalid JSON in response body")
        assert classify_trace_error(step) == ErrorCategory.LLM_PARSE_ERROR

    def test_mcp_connection_error(self):
        step = _step_with_error("BlenderSocketUnavailable: connection refused")
        assert classify_trace_error(step) == ErrorCategory.MCP_CONNECTION_ERROR

    def test_mcp_connection_error_socket(self):
        step = _step_with_error("socket connection error: refused")
        assert classify_trace_error(step) == ErrorCategory.MCP_CONNECTION_ERROR

    def test_remote_agent_error(self):
        step = _step_with_error("RemoteAgentError: upstream agent failed")
        assert classify_trace_error(step) == ErrorCategory.REMOTE_AGENT_ERROR

    def test_tool_invalid_arguments(self):
        step = _step_with_error("argument 'location' is invalid: expected list")
        assert classify_trace_error(step) == ErrorCategory.TOOL_INVALID_ARGUMENTS

    def test_tool_runtime_error_execution(self):
        step = _step_with_error("execution error in blender script: NameError")
        assert classify_trace_error(step) == ErrorCategory.TOOL_RUNTIME_ERROR

    def test_tool_runtime_error_traceback(self):
        step = _step_with_error("Traceback (most recent call last): ...")
        assert classify_trace_error(step) == ErrorCategory.TOOL_RUNTIME_ERROR

    def test_tool_invocation_error(self):
        step = _step_with_error("ToolInvocationError: execution failed: runtime error")
        assert classify_trace_error(step) == ErrorCategory.TOOL_RUNTIME_ERROR

    def test_unknown_error_fallback(self):
        step = _step_with_error("something went completely wrong")
        assert classify_trace_error(step) == ErrorCategory.UNKNOWN_ERROR

    def test_disabled_takes_priority_over_unknown(self):
        step = _step_with_error("tool is disabled and not allowed")
        assert classify_trace_error(step) == ErrorCategory.TOOL_DISABLED


# ---------------------------------------------------------------------------
# classify_validation_issue — issue code → error category
# ---------------------------------------------------------------------------

class TestClassifyValidationIssue:
    def test_object_missing(self):
        assert classify_validation_issue(_issue("object_missing")) == ErrorCategory.SCENE_OBJECT_MISSING

    def test_object_type_mismatch(self):
        assert classify_validation_issue(_issue("object_type_mismatch")) == ErrorCategory.SCENE_OBJECT_MISSING

    def test_primitive_mismatch(self):
        assert classify_validation_issue(_issue("primitive_mismatch")) == ErrorCategory.SCENE_TRANSFORM_MISMATCH

    def test_dimensions_mismatch(self):
        assert classify_validation_issue(_issue("dimensions_mismatch")) == ErrorCategory.SCENE_TRANSFORM_MISMATCH

    def test_transform_location_mismatch(self):
        assert classify_validation_issue(_issue("location_mismatch")) == ErrorCategory.SCENE_TRANSFORM_MISMATCH

    def test_transform_rotation_mismatch(self):
        assert classify_validation_issue(_issue("rotation_mismatch")) == ErrorCategory.SCENE_TRANSFORM_MISMATCH

    def test_transform_scale_mismatch(self):
        assert classify_validation_issue(_issue("scale_mismatch")) == ErrorCategory.SCENE_TRANSFORM_MISMATCH

    def test_material_code(self):
        assert classify_validation_issue(_issue("material_color_mismatch")) == ErrorCategory.SCENE_MATERIAL_MISMATCH

    def test_material_object_missing(self):
        assert classify_validation_issue(_issue("object_missing_for_material")) == ErrorCategory.SCENE_MATERIAL_MISMATCH

    def test_light_code(self):
        assert classify_validation_issue(_issue("light_energy_mismatch")) == ErrorCategory.SCENE_LIGHT_MISMATCH

    def test_camera_code(self):
        assert classify_validation_issue(_issue("camera_fov_mismatch")) == ErrorCategory.SCENE_CAMERA_MISMATCH

    def test_active_camera_code(self):
        assert classify_validation_issue(_issue("active_camera_missing")) == ErrorCategory.SCENE_CAMERA_MISMATCH

    def test_export_code(self):
        assert classify_validation_issue(_issue("export_file_missing")) == ErrorCategory.SCENE_EXPORT_MISSING

    def test_unknown_code_fallback(self):
        assert classify_validation_issue(_issue("completely_custom_code")) == ErrorCategory.UNKNOWN_ERROR

    def test_fixture_partial_material_issue(self):
        issue = _issue("material_roughness_mismatch")
        assert classify_validation_issue(issue) == ErrorCategory.SCENE_MATERIAL_MISMATCH


# ---------------------------------------------------------------------------
# extract_errors
# ---------------------------------------------------------------------------

class TestExtractErrors:
    def test_no_errors_returns_empty(self):
        step = AgentStep(
            step_index=0, step_type=AgentStepType.TOOL_CALL,
            tool_arguments={}, started_at=_NOW, finished_at=_NOW,
        )
        assert extract_errors(_trace(step)) == []

    def test_extracts_error_steps(self):
        trace = _trace(
            _step_with_error("execution error", index=0),
            _step_with_error("tool not allowed", index=1),
        )
        records = extract_errors(trace)
        assert len(records) == 2

    def test_record_fields_populated(self):
        trace = _trace(_step_with_error("execution error in blender", index=3))
        records = extract_errors(trace)
        r = records[0]
        assert r.run_id == "r1"
        assert r.task_id == "t1"
        assert r.step_index == 3
        assert r.category == ErrorCategory.TOOL_RUNTIME_ERROR
        assert "execution" in r.message

    def test_tool_name_captured_for_tool_steps(self):
        step = AgentStep(
            step_index=0, step_type=AgentStepType.TOOL_CALL,
            tool_name="bma_create_object", tool_arguments={},
            error="execution error in blender",
            started_at=_NOW, finished_at=_NOW,
        )
        records = extract_errors(_trace(step))
        assert records[0].tool_name == "bma_create_object"

    def test_fixture_tool_error_trace(self):
        from benchmark.agent.models import AgentTrace as AT
        t = AT.model_validate_json((FIXTURES / "agent_trace_tool_error.json").read_text())
        records = extract_errors(t)
        cats = {r.category for r in records}
        assert ErrorCategory.TOOL_RUNTIME_ERROR in cats
        assert ErrorCategory.TOOL_DISABLED in cats


# ---------------------------------------------------------------------------
# aggregate_errors
# ---------------------------------------------------------------------------

class TestAggregateErrors:
    def test_empty_bundle_returns_empty_dict(self):
        bundle = RunArtifactBundle(run_dir=FIXTURES)
        assert aggregate_errors(bundle) == {}

    def test_counts_trace_errors(self):
        trace = _trace(
            _step_with_error("execution error in blender", index=0),
            _step_with_error("execution error in blender", index=1),
        )
        bundle = RunArtifactBundle(run_dir=FIXTURES, agent_trace=trace)
        counts = aggregate_errors(bundle)
        assert counts.get(ErrorCategory.TOOL_RUNTIME_ERROR.value, 0) == 2

    def test_counts_validation_issues(self):
        from benchmark.analysis.trace_reader import read_validation_result
        val = read_validation_result(FIXTURES / "validation_result_partial.json")
        bundle = RunArtifactBundle(run_dir=FIXTURES, validation_result=val)
        counts = aggregate_errors(bundle)
        assert counts.get(ErrorCategory.SCENE_MATERIAL_MISMATCH.value, 0) >= 1

    def test_combines_trace_and_validation(self):
        from benchmark.analysis.trace_reader import read_validation_result
        trace = _trace(_step_with_error("tool not allowed", index=0))
        val = read_validation_result(FIXTURES / "validation_result_partial.json")
        bundle = RunArtifactBundle(run_dir=FIXTURES, agent_trace=trace, validation_result=val)
        counts = aggregate_errors(bundle)
        assert counts.get(ErrorCategory.TOOL_DISABLED.value, 0) >= 1
        assert counts.get(ErrorCategory.SCENE_MATERIAL_MISMATCH.value, 0) >= 1

    def test_returns_dict_with_string_keys(self):
        trace = _trace(_step_with_error("execution error in blender"))
        bundle = RunArtifactBundle(run_dir=FIXTURES, agent_trace=trace)
        counts = aggregate_errors(bundle)
        assert all(isinstance(k, str) for k in counts)

    def test_deduplicates_top_level_and_validator_validation_issues(self):
        from benchmark.validation.models import SceneValidationResult, ValidationStatus, ValidatorResult

        issue = ValidationIssue(
            code="dimensions_mismatch",
            message="bad dimensions",
            severity=ValidationSeverity.ERROR,
            expected_path="expected_scene.objects[0].dimensions",
            actual_path="snapshot.objects[0].dimensions",
        )
        val = SceneValidationResult(
            task_id="t1",
            overall_status=ValidationStatus.FAILED,
            total_score=0.5,
            validators=[
                ValidatorResult(
                    name="transform_validator",
                    status=ValidationStatus.FAILED,
                    score=0.0,
                    issues=[issue],
                )
            ],
            issues=[issue],
            summary={},
        )
        bundle = RunArtifactBundle(run_dir=FIXTURES, validation_result=val)

        counts = aggregate_errors(bundle)

        assert counts == {ErrorCategory.SCENE_TRANSFORM_MISMATCH.value: 1}


# ---------------------------------------------------------------------------
# summarize_errors
# ---------------------------------------------------------------------------

class TestSummarizeErrors:
    def test_empty_list(self):
        assert summarize_errors([]) == {}

    def test_counts_by_category(self):
        from benchmark.analysis.models import ErrorRecord
        records = [
            ErrorRecord(run_id="r1", task_id="t1", step_index=0,
                        category=ErrorCategory.TOOL_RUNTIME_ERROR, message="a"),
            ErrorRecord(run_id="r1", task_id="t1", step_index=1,
                        category=ErrorCategory.TOOL_RUNTIME_ERROR, message="b"),
            ErrorRecord(run_id="r1", task_id="t1", step_index=2,
                        category=ErrorCategory.TOOL_DISABLED, message="c"),
        ]
        counts = summarize_errors(records)
        assert counts["tool_runtime_error"] == 2
        assert counts["tool_disabled"] == 1
