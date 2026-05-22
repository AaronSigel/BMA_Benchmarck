"""Export-focused ReAct post-pass termination tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from benchmark.agent.llm import LlmResponse, MockLlmClient
from benchmark.agent.models import AgentConfig, AgentStepType, AgentStrategyName, LlmConfig
from benchmark.agent.strategies.react import ReactStrategy
from benchmark.agent.tool_context import AgentToolContext
from benchmark.agent.tool_executor import MockToolExecutor
from benchmark.validation.models import ValidationStatus


def _config(**kwargs) -> AgentConfig:
    params = {
        "agent_id": "test-agent",
        "strategy": AgentStrategyName.REACT,
        "llm": LlmConfig(provider="mock", model="mock"),
        "mcp_profile": "minimal",
        "max_steps": 6,
        "stop_after_scene_passed": True,
        "detect_no_progress": True,
        "detect_repeated_actions": True,
        "no_progress_limit": 2,
        "max_steps_by_category": {"export": 6},
    }
    params.update(kwargs)
    return AgentConfig(**params)


def _passed_result() -> MagicMock:
    result = MagicMock()
    result.total_score = 1.0
    result.overall_status = ValidationStatus.PASSED
    result.issues = []
    return result


def test_react_stops_before_next_step_when_scene_passed_on_export() -> None:
    strategy = ReactStrategy()
    strategy.initial_validation_result = _passed_result()
    llm = MockLlmClient([LlmResponse(content='{"thought":"export","action":{"tool":"bma_export_scene","arguments":{}},"finish":false}')] * 5)
    trace = strategy.run(
        {
            "id": "export_002_glb_file",
            "category": "export",
            "prompt": "export glb",
            "allowed_tools": ["bma_export_scene"],
        },
        _config(),
        llm,
        MockToolExecutor(),
        AgentToolContext(run_id="run-export-stop", task_id="export_002_glb_file"),
        Path("."),
    )
    assert trace.success is True
    assert trace.error is None
    assert trace.metadata.get("early_stop_reason") == "scene_passed_after_validation"
    assert not [step for step in trace.steps if step.step_type == AgentStepType.LLM_CALL]


def test_react_max_steps_after_passed_scene_not_hard_failure() -> None:
    strategy = ReactStrategy()
    calls = 0

    def validator_fn(_path: Path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return False, 0.0, MagicMock(
                total_score=0.0,
                overall_status=ValidationStatus.FAILED,
                issues=[],
            )
        return True, 1.0, _passed_result()

    strategy.scene_validator_fn = validator_fn
    llm = MockLlmClient([
        LlmResponse(content='{"thought":"export","action":{"tool":"bma_export_scene","arguments":{"format":"glb"}},"finish":false}'),
        LlmResponse(content='{"thought":"again","action":{"tool":"bma_export_scene","arguments":{"format":"glb"}},"finish":false}'),
        LlmResponse(content='{"thought":"again2","action":{"tool":"bma_export_scene","arguments":{"format":"glb"}},"finish":false}'),
    ])
    trace = strategy.run(
        {
            "id": "export_002_glb_file",
            "category": "export",
            "prompt": "export glb",
            "allowed_tools": ["bma_export_scene"],
        },
        _config(max_steps=3),
        llm,
        MockToolExecutor(results={"bma_export_scene": {"ok": True}}),
        AgentToolContext(run_id="run-export-max", task_id="export_002_glb_file"),
        Path("."),
    )
    assert trace.success is True
    assert trace.error is None
    assert trace.metadata.get("react_error_type") is None


def test_react_invalid_action_after_passed_scene_is_soft_diagnostic() -> None:
    strategy = ReactStrategy()
    calls = 0

    def validator_fn(_path: Path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return False, 0.0, MagicMock(
                total_score=0.0,
                overall_status=ValidationStatus.FAILED,
                issues=[],
            )
        return True, 1.0, _passed_result()

    strategy.scene_validator_fn = validator_fn
    repeated = '{"thought":"export","action":{"tool":"bma_export_scene","arguments":{"format":"glb"}},"finish":false}'
    llm = MockLlmClient([LlmResponse(content=repeated)] * 4)
    trace = strategy.run(
        {
            "id": "export_001_blend_file",
            "category": "export",
            "prompt": "export blend",
            "allowed_tools": ["bma_export_scene"],
        },
        _config(max_steps=4),
        llm,
        MockToolExecutor(results={"bma_export_scene": {"ok": True}}),
        AgentToolContext(run_id="run-export-invalid", task_id="export_001_blend_file"),
        Path("."),
    )
    assert trace.success is True
    assert trace.error is None
    assert trace.metadata.get("react_error_type") is None
