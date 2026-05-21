"""Tests for ReAct loop guard conditions and progress detection."""
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


def _mock_val_result(
    passed: bool = False,
    score: float = 0.5,
    issues: list[ValidationIssue] | None = None,
) -> MagicMock:
    result = MagicMock()
    result.passed = passed
    result.total_score = score
    result.issues = issues or []
    result.overall_status = ValidationStatus.PASSED if passed else ValidationStatus.FAILED
    return result


def _blocking_issue(code: str = "object_missing") -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=f"Issue: {code}",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.objects[0]",
        actual_path=None,
        expected_value=None,
        actual_value=None,
    )


# --- Export block guard ---

def test_export_blocked_until_scene_valid() -> None:
    """Export call must be blocked when blocking issues are present.

    The LLM first inspects the scene (triggering validation which sets last_validation_result),
    then attempts to export — at that point the guard has a validation result to check.
    """
    blocking_issue = _blocking_issue("object_missing")
    val_result = _mock_val_result(passed=False, score=0.3, issues=[blocking_issue])

    # Step 1: inspect (triggers validation), Steps 2+: try to export (should be blocked)
    llm = MockLlmClient([
        LlmResponse(content='{"thought":"inspect first","action":{"tool":"bma_get_scene_snapshot","arguments":{}}}'),
        LlmResponse(content='{"thought":"export","action":{"tool":"bma_export_scene","arguments":{}}}'),
        LlmResponse(content='{"thought":"export again","action":{"tool":"bma_export_scene","arguments":{}}}'),
        LlmResponse(content='{"thought":"export again","action":{"tool":"bma_export_scene","arguments":{}}}'),
    ] * 3)
    executor = MockToolExecutor(results={
        "bma_get_scene_snapshot": {"objects": []},
        "bma_export_scene": {"exported": True},
    })
    strategy = ReactStrategy()

    def validator_fn(path):
        return False, 0.3, val_result

    strategy.scene_validator_fn = validator_fn

    trace = strategy.run(
        {"id": "export_task", "prompt": "Export", "category": "export"},
        _config(max_steps=8, detect_repeated_actions=False, stop_after_scene_passed=True, detect_no_progress=False),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="export_task"),
        Path("."),
    )

    # Export should have been blocked at least once (after validation ran on step 1)
    assert trace.metadata.get("react_blocked_export_count", 0) >= 1

    # bma_export_scene should not have been executed after the first validation set the result
    export_steps = [
        step for step in trace.steps
        if step.step_type == AgentStepType.TOOL_CALL and step.tool_name == "bma_export_scene"
    ]
    assert len(export_steps) == 0, "bma_export_scene must not execute while blocking issues exist"


# --- Premature finish guard ---

def test_premature_finish_blocked_when_validation_fails() -> None:
    """finish=true must be rejected when the scene has not passed validation."""
    val_result = _mock_val_result(passed=False, score=0.4, issues=[_blocking_issue()])

    # LLM immediately tries to finish
    llm = MockLlmClient([
        LlmResponse(content='{"thought":"done","finish":true}'),
        LlmResponse(content='{"thought":"done","finish":true}'),
        LlmResponse(content='{"thought":"still done","finish":true}'),
    ] * 5)
    executor = MockToolExecutor()
    strategy = ReactStrategy()

    def validator_fn(path):
        return False, 0.4, val_result

    strategy.scene_validator_fn = validator_fn

    trace = strategy.run(
        {"id": "task-1", "prompt": "Do it", "category": "geometry"},
        _config(max_steps=4, detect_repeated_actions=False, stop_after_scene_passed=True, detect_no_progress=False),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    # The finish was blocked and the loop hit max_steps or another stop condition
    assert trace.success is not True or trace.error == "ReactMaxSteps"
    # Check that premature finish was logged
    premature_finish_steps = [
        s for s in trace.steps
        if (s.metadata or {}).get("premature_finish_blocked")
    ]
    assert len(premature_finish_steps) >= 1


# --- Repeated action guard ---

def test_repeated_action_hint_then_error() -> None:
    """Guard allows 1 repair hint on first repeat, then stops on second repeat."""
    same_action = '{"thought":"again","action":{"tool":"bma_get_scene_snapshot","arguments":{}}}'
    llm = MockLlmClient([LlmResponse(content=same_action)] * 10)
    executor = MockToolExecutor(results={"bma_get_scene_snapshot": {"objects": []}})

    trace = ReactStrategy().run(
        # Use "export" category (8 effective max steps) so the loop doesn't hit category max
        # before the repeated-action guard fires twice
        {"id": "task-1", "prompt": "Do it", "category": "export"},
        _config(max_steps=20, detect_repeated_actions=True),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    # Should have stopped due to repeated action
    assert trace.metadata.get("repeated_action_count", 0) >= 1
    error_steps = [s for s in trace.steps if s.step_type == AgentStepType.ERROR]
    assert any(
        "ReactInvalidAction" in (s.error or "") or (s.metadata or {}).get("react_error_type") == "ReactInvalidAction"
        for s in error_steps
    )


# --- Duplicate object guard ---

def test_duplicate_object_creation_blocked() -> None:
    """Creating the same named object twice triggers guard, not actual creation."""
    create_action = '{"thought":"create","action":{"tool":"bma_create_object","arguments":{"name":"Cube","type":"MESH_CUBE"}}}'
    llm = MockLlmClient([LlmResponse(content=create_action)] * 8)
    executor = MockToolExecutor(results={"bma_create_object": {"name": "Cube"}})

    trace = ReactStrategy().run(
        {"id": "task-1", "prompt": "Create cube", "category": "geometry"},
        _config(max_steps=6, detect_duplicate_objects=True, detect_repeated_actions=False),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    # Second+ create attempts should have been blocked
    actual_creates = [
        s for s in trace.steps
        if s.step_type == AgentStepType.TOOL_CALL and s.tool_name == "bma_create_object"
    ]
    assert len(actual_creates) == 1, "Cube should only be created once"
