"""Tests for ReAct early termination after scene validation passes."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from benchmark.agent.llm import LlmResponse, MockLlmClient
from benchmark.agent.models import AgentConfig, AgentStepType, AgentStrategyName, LlmConfig
from benchmark.agent.strategies.react import ReactStrategy
from benchmark.agent.tool_context import AgentToolContext
from benchmark.agent.tool_executor import MockToolExecutor
from benchmark.validation.models import ValidationIssue, ValidationSeverity, ValidationStatus


def _config(**kwargs) -> AgentConfig:
    params = {
        "agent_id": "test-agent",
        "strategy": AgentStrategyName.REACT,
        "llm": LlmConfig(provider="mock", model="mock"),
        "mcp_profile": "minimal",
        "max_steps": 5,
        "stop_after_scene_passed": True,
        "detect_no_progress": True,
        "detect_repeated_actions": True,
        "no_progress_limit": 2,
    }
    params.update(kwargs)
    return AgentConfig(**params)


def _failed_result() -> MagicMock:
    result = MagicMock()
    result.total_score = 0.0
    result.overall_status = ValidationStatus.FAILED
    result.issues = [
        ValidationIssue(
            code="object_missing",
            message="missing",
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
    result.total_score = 1.0
    result.overall_status = ValidationStatus.PASSED
    result.issues = []
    return result


def test_react_stops_immediately_after_passed_validation() -> None:
    strategy = ReactStrategy()
    strategy.initial_validation_result = _passed_result()
    llm = MockLlmClient([LlmResponse(content='{"thought":"noop","action":null,"finish":true}')] * 5)
    trace = strategy.run(
        {"id": "geometry_001", "category": "geometry", "prompt": "x", "allowed_tools": []},
        _config(),
        llm,
        MockToolExecutor(),
        AgentToolContext(run_id="run-early-stop", task_id="geometry_001"),
        Path("."),
    )
    assert trace.success is True
    assert trace.error is None
    assert trace.metadata.get("early_stop_reason") == "scene_passed_after_validation"
    assert trace.metadata.get("skipped_llm_after_passed") is True
    assert not [step for step in trace.steps if step.step_type == AgentStepType.LLM_CALL]


def test_invalid_action_after_passed_scene_is_warning_not_error() -> None:
    strategy = ReactStrategy()
    calls = 0

    def validator_fn(_path: Path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return False, 0.0, _failed_result()
        return True, 1.0, _passed_result()

    strategy.scene_validator_fn = validator_fn
    repeated = (
        '{"thought":"create","action":{"tool":"bma_create_object","arguments":{"name":"Cube","type":"MESH_CUBE"}},"finish":false}'
    )
    llm = MockLlmClient([LlmResponse(content=repeated)] * 4)
    trace = strategy.run(
        {
            "id": "geometry_001",
            "category": "geometry",
            "prompt": "create cube",
            "allowed_tools": ["bma_create_object"],
            "expected_scene": {"objects": [{"name": "Cube", "type": "MESH"}], "materials": [], "lights": [], "cameras": [], "exports": []},
        },
        _config(max_steps=4),
        llm,
        MockToolExecutor(results={"bma_create_object": {"name": "Cube"}}),
        AgentToolContext(run_id="run-passed-stop", task_id="geometry_001"),
        Path("."),
    )
    assert trace.success is True
    assert trace.error is None
    assert trace.metadata.get("react_error_type") is None
    assert trace.metadata.get("scene_passed_but_agent_error") in {False, None}


def test_max_steps_after_passed_scene_does_not_create_error_type() -> None:
    strategy = ReactStrategy()
    calls = 0

    def validator_fn(_path: Path):
        nonlocal calls
        calls += 1
        if calls <= 1:
            return False, 0.0, _failed_result()
        return True, 1.0, _passed_result()

    strategy.scene_validator_fn = validator_fn
    llm = MockLlmClient([
        LlmResponse(content='{"thought":"create","action":{"tool":"bma_create_object","arguments":{"name":"Cube","type":"MESH_CUBE"}},"finish":false}'),
        LlmResponse(content='{"thought":"repeat","action":{"tool":"bma_create_object","arguments":{"name":"Cube","type":"MESH_CUBE"}},"finish":false}'),
        LlmResponse(content='{"thought":"repeat again","action":{"tool":"bma_create_object","arguments":{"name":"Cube","type":"MESH_CUBE"}},"finish":false}'),
    ])
    trace = strategy.run(
        {
            "id": "geometry_001",
            "category": "geometry",
            "prompt": "create cube",
            "allowed_tools": ["bma_create_object"],
            "expected_scene": {"objects": [{"name": "Cube", "type": "MESH"}], "materials": [], "lights": [], "cameras": [], "exports": []},
        },
        _config(max_steps=3),
        llm,
        MockToolExecutor(results={"bma_create_object": {"name": "Cube"}}),
        AgentToolContext(run_id="run-no-max-error", task_id="geometry_001"),
        Path("."),
    )
    assert trace.success is True
    assert trace.error is None
    assert trace.metadata.get("react_error_type") is None
