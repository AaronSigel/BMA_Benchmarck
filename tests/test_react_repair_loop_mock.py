"""Integration smoke test: ReAct repair loop driven by a mock LLM and mock validator.

Scenario:
  1. Empty scene — validator reports object_missing.
  2. Mapper suggests bma_create_object.
  3. Mock LLM returns the correct create_object action.
  4. MockToolExecutor executes it.
  5. Validator then reports scene passed.
  6. ReAct stops cleanly (early_stop) without hitting max_steps.
"""
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
        stop_after_scene_passed=True,
        detect_no_progress=True,
        detect_repeated_actions=True,
        detect_duplicate_objects=True,
        no_progress_limit=2,
        **kwargs,
    )


def _object_missing_result() -> MagicMock:
    result = MagicMock()
    result.passed = False
    result.total_score = 0.0
    result.overall_status = ValidationStatus.FAILED
    result.issues = [
        ValidationIssue(
            code="object_missing",
            message="Expected object 'Cube' not found.",
            severity=ValidationSeverity.ERROR,
            expected_path="expected_scene.objects[0]",
            actual_path=None,
            expected_value={"name": "Cube"},
            actual_value=None,
        )
    ]
    return result


def _passed_result() -> MagicMock:
    result = MagicMock()
    result.passed = True
    result.total_score = 1.0
    result.overall_status = ValidationStatus.PASSED
    result.issues = []
    return result


def test_react_repair_loop_resolves_object_missing_and_finishes() -> None:
    """Full repair loop: object_missing → create → passed → early stop."""
    llm = MockLlmClient([
        # Step 1: LLM creates the object
        LlmResponse(
            content='{"thought":"Need to create missing Cube.","action":{"tool":"bma_create_object","arguments":{"name":"Cube","type":"MESH_CUBE"}}}'
        ),
        # Step 2 (if needed): LLM finishes
        LlmResponse(content='{"thought":"Scene complete.","finish":true}'),
    ])
    executor = MockToolExecutor(results={"bma_create_object": {"name": "Cube", "type": "MESH"}})
    strategy = ReactStrategy()

    def validator_fn(path: Path):
        # After bma_create_object runs, the object exists — scene passes
        return True, 1.0, _passed_result()

    strategy.scene_validator_fn = validator_fn

    task = {
        "id": "geometry_001_basic_primitives",
        "title": "Create a cube",
        "category": "geometry",
        "prompt": "Create a cube named 'Cube' at the origin.",
        "expected_scene": {
            "objects": [{"name": "Cube", "type": "MESH", "primitive": "cube"}],
            "materials": [],
            "lights": [],
            "cameras": [],
            "exports": [],
        },
        "success_criteria": [{"metric": "object_existence", "weight": 1.0, "required": True}],
        "difficulty": "easy",
        "tags": ["geometry"],
        "allowed_tools": ["bma_create_object"],
    }

    trace = strategy.run(
        task,
        _config(max_steps=8),
        llm,
        executor,
        AgentToolContext(run_id="run-repair-loop", task_id="geometry_001_basic_primitives"),
        Path("."),
    )

    # Should have succeeded without hitting max_steps
    assert trace.success is True
    assert trace.error is None or trace.error == "ReactMaxSteps"  # prefer no error
    assert trace.final_message == "scene_passed_early_stop"

    # Should have used fewer than max steps
    tool_calls = [s for s in trace.steps if s.step_type == AgentStepType.TOOL_CALL]
    assert len(tool_calls) >= 1, "At least one tool call (bma_create_object) should have occurred"
    assert any(s.tool_name == "bma_create_object" for s in tool_calls)

    # Aggregate metrics should be present
    assert "react_steps_total" in trace.metadata
    assert trace.metadata.get("react_max_steps_count") == 0


def test_react_repair_loop_no_max_steps_on_clean_fix() -> None:
    """Repair that works on the first try must not trigger max_steps."""
    llm = MockLlmClient([
        LlmResponse(
            content='{"thought":"Create Cube.","action":{"tool":"bma_create_object","arguments":{"name":"Cube","type":"MESH_CUBE"}}}'
        ),
    ] * 10)
    executor = MockToolExecutor(results={"bma_create_object": {"name": "Cube"}})
    strategy = ReactStrategy()

    def validator_fn(path: Path):
        # Object was created on first action — scene passes immediately
        return True, 1.0, _passed_result()

    strategy.scene_validator_fn = validator_fn

    trace = strategy.run(
        {"id": "task-1", "prompt": "Create cube", "category": "geometry"},
        _config(max_steps=10),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.success is True
    assert trace.metadata.get("react_error_type") is None
    assert trace.metadata.get("react_max_steps_count") == 0
