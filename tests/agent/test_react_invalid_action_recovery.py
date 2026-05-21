from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from benchmark.agent.llm import LlmResponse, MockLlmClient
from benchmark.agent.models import AgentConfig, AgentStepType, AgentStrategyName, LlmConfig
from benchmark.agent.strategies.react import ReactStrategy
from benchmark.agent.tool_context import AgentToolContext
from benchmark.agent.tool_executor import MockToolExecutor
from benchmark.tasks.models import BenchmarkTask, DifficultyLevel, ExpectedLight, ExpectedScene, SuccessCriterion, TaskCategory, Vector3
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


def _failed_validation() -> MagicMock:
    result = MagicMock()
    result.total_score = 0.2
    result.overall_status = ValidationStatus.FAILED
    result.issues = [
        ValidationIssue(
            code="light_missing",
            message="missing light",
            severity=ValidationSeverity.ERROR,
            expected_path="expected_scene.lights[0]",
            actual_path=None,
            expected_value={"name": "KeyLight"},
            actual_value=None,
        )
    ]
    return result


def _lighting_task() -> BenchmarkTask:
    return BenchmarkTask(
        id="lighting_001",
        title="Lighting",
        category=TaskCategory.LIGHTING,
        difficulty=DifficultyLevel.EASY,
        prompt="Add light",
        tags=["lighting"],
        allowed_tools=[],
        expected_scene=ExpectedScene(
            lights=[ExpectedLight(name="KeyLight", type="AREA", location=Vector3(x=0, y=0, z=5))],
        ),
        success_criteria=[SuccessCriterion(metric="light_existence", weight=1.0)],
    )


def test_invalid_light_tool_recovered_by_issue_mapper() -> None:
    strategy = ReactStrategy()
    strategy.initial_validation_result = _failed_validation()
    task = _lighting_task()
    action_json = (
        '{"thought":"repeat","finish":false,'
        '"action":{"tool":"bma_set_light_properties","arguments":{"name":"KeyLight"}}}'
    )
    llm = MockLlmClient([LlmResponse(content=action_json)] * 4)
    trace = strategy.run(
        task.model_dump(mode="json"),
        _config(),
        llm,
        MockToolExecutor(),
        AgentToolContext(run_id="r1", task_id=task.id, artifacts_dir=Path("/tmp")),
        Path("/tmp"),
    )
    recovered = any(
        (step.metadata or {}).get("recovered_by_mapper")
        for step in trace.steps
    )
    assert recovered
    assert trace.error != "ReactInvalidAction"


def test_invalid_action_without_repair_becomes_agent_error() -> None:
    strategy = ReactStrategy()
    action_json = (
        '{"thought":"repeat","finish":false,'
        '"action":{"tool":"bma_set_transform","arguments":{"object_name":"Cube","location":[0,0,0]}}}'
    )
    llm = MockLlmClient([LlmResponse(content=action_json)] * 4)
    trace = strategy.run(
        {"id": "geometry_001", "category": "geometry", "prompt": "cube"},
        _config(detect_repeated_actions=True),
        llm,
        MockToolExecutor(),
        AgentToolContext(run_id="r2", task_id="geometry_001", artifacts_dir=Path("/tmp")),
        Path("/tmp"),
    )
    assert trace.metadata.get("react_error_type") == "ReactInvalidAction" or trace.error == "ReactInvalidAction"
