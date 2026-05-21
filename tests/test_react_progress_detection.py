"""Tests for ReAct progress detection, error classification, and trace metadata."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from benchmark.agent.llm import LlmResponse, MockLlmClient
from benchmark.agent.models import AgentConfig, AgentStepType, AgentStrategyName, LlmConfig
from benchmark.agent.strategies import ReactStrategy
from benchmark.agent.tool_context import AgentToolContext
from benchmark.agent.tool_executor import MockToolExecutor
from benchmark.validation.models import ValidationIssue, ValidationSeverity, ValidationStatus


def _config(max_steps: int = 10, **kwargs) -> AgentConfig:
    return AgentConfig(
        agent_id="test-agent",
        strategy=AgentStrategyName.REACT,
        llm=LlmConfig(provider="mock", model="mock"),
        mcp_profile="minimal",
        max_steps=max_steps,
        **kwargs,
    )


def _val_result(passed: bool = False, score: float = 0.5, issues: list | None = None) -> MagicMock:
    result = MagicMock()
    result.passed = passed
    result.total_score = score
    result.issues = issues or []
    result.overall_status = ValidationStatus.PASSED if passed else ValidationStatus.FAILED
    return result


def _issue(code: str = "object_missing") -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=f"Issue: {code}",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.objects[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )


# --- Max steps classification ---

def test_max_steps_classified_as_react_max_steps() -> None:
    """When max_steps is reached, error should be ReactMaxSteps."""
    action = '{"thought":"ok","action":{"tool":"bma_get_scene_snapshot","arguments":{}}}'
    llm = MockLlmClient([LlmResponse(content=action)] * 20)
    executor = MockToolExecutor(results={"bma_get_scene_snapshot": {"objects": []}})

    trace = ReactStrategy().run(
        # Use "export" with max_steps_by_category to control effective limit precisely
        {"id": "task-1", "prompt": "Do it", "category": "export"},
        _config(max_steps=3, detect_repeated_actions=False, max_steps_by_category={"export": 3}),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.error == "ReactMaxSteps"
    assert trace.metadata.get("react_error_type") == "ReactMaxSteps"
    assert trace.metadata.get("react_max_steps_count") == 1


# --- No progress detection ---

def test_no_progress_stops_with_react_no_progress() -> None:
    """When progress stalls for no_progress_limit steps, error should be ReactNoProgress."""
    stall_result = _val_result(passed=False, score=0.5, issues=[_issue()])

    action = '{"thought":"ok","action":{"tool":"bma_get_scene_snapshot","arguments":{}}}'
    llm = MockLlmClient([LlmResponse(content=action)] * 20)
    executor = MockToolExecutor(results={"bma_get_scene_snapshot": {"objects": []}})
    strategy = ReactStrategy()

    def validator_fn(path):
        return False, 0.5, stall_result

    strategy.scene_validator_fn = validator_fn

    trace = strategy.run(
        # Use "export" category (8 effective steps) so category max doesn't trigger first
        {"id": "task-1", "prompt": "Do it", "category": "export"},
        _config(
            max_steps=20,
            no_progress_limit=2,
            detect_no_progress=True,
            detect_repeated_actions=False,
        ),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.error == "ReactNoProgress"
    assert trace.metadata.get("react_error_type") == "ReactNoProgress"
    assert trace.metadata.get("react_no_progress_count", 0) >= 1


# --- Wasted steps ---

def test_repeated_action_becomes_wasted_step() -> None:
    """Repeated same action is counted as a wasted step."""
    action_str = '{"thought":"again","action":{"tool":"bma_get_scene_snapshot","arguments":{}}}'
    llm = MockLlmClient([LlmResponse(content=action_str)] * 10)
    executor = MockToolExecutor(results={"bma_get_scene_snapshot": {}})

    trace = ReactStrategy().run(
        {"id": "task-1", "prompt": "Do it", "category": "export"},
        _config(max_steps=20, detect_repeated_actions=True),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.metadata.get("react_wasted_steps", 0) >= 1
    assert trace.metadata.get("repeated_action_count", 0) >= 1


# --- Early stop on scene passed ---

def test_early_stop_when_scene_passes() -> None:
    """When scene passes, ReactStrategy should stop and mark success."""
    pass_result = _val_result(passed=True, score=1.0, issues=[])

    action = '{"thought":"ok","action":{"tool":"bma_get_scene_snapshot","arguments":{}}}'
    llm = MockLlmClient([LlmResponse(content=action)] * 20)
    executor = MockToolExecutor(results={"bma_get_scene_snapshot": {}})
    strategy = ReactStrategy()

    def validator_fn(path):
        return True, 1.0, pass_result

    strategy.scene_validator_fn = validator_fn

    trace = strategy.run(
        {"id": "task-1", "prompt": "Do it", "category": "export"},
        _config(max_steps=20, stop_after_scene_passed=True, detect_repeated_actions=False),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.success is True
    assert trace.final_message == "scene_passed_early_stop"
    assert len(trace.steps) < 10


# --- Trace metadata completeness ---

def test_trace_metadata_includes_react_aggregates() -> None:
    """Trace metadata must include all required ReAct aggregate fields."""
    action = '{"thought":"ok","action":{"tool":"bma_get_scene_snapshot","arguments":{}}}'
    llm = MockLlmClient([LlmResponse(content=action)] * 20)
    executor = MockToolExecutor(results={"bma_get_scene_snapshot": {}})

    trace = ReactStrategy().run(
        {"id": "task-1", "prompt": "Do it", "category": "export"},
        _config(max_steps=2, detect_repeated_actions=False, max_steps_by_category={"export": 2}),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    meta = trace.metadata
    assert "react_steps_total" in meta
    assert "react_repair_steps" in meta
    assert "react_wasted_steps" in meta
    assert "react_no_progress_count" in meta
    assert "react_blocked_export_count" in meta
    assert "react_max_steps_count" in meta
    assert "react_error_type" in meta
    assert "react_issue_resolution_rate" in meta
