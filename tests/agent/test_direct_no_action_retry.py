"""Tests for DirectNoAction retry and JSON action parsing."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from benchmark.agent.llm import LlmResponse, MockLlmClient
from benchmark.agent.models import AgentConfig, AgentStepType, AgentStrategyName, LlmConfig
from benchmark.agent.strategies.direct_tool_calling import DirectToolCallingStrategy, _extract_tool_calls
from benchmark.agent.tool_context import AgentToolContext
from benchmark.agent.tool_executor import MockToolExecutor


def _config() -> AgentConfig:
    return AgentConfig(
        agent_id="direct-test",
        strategy=AgentStrategyName.DIRECT_TOOL_CALLING,
        llm=LlmConfig(provider="mock", model="mock"),
        mcp_profile="minimal",
        max_steps=4,
    )


def test_direct_parses_json_action_from_content() -> None:
    response = LlmResponse(content='{"tool_name":"bma_create_object","arguments":{"name":"Cube","type":"MESH_CUBE"}}')
    calls = _extract_tool_calls(response)
    assert len(calls) == 1
    assert calls[0].name == "bma_create_object"


def test_direct_retries_once_on_empty_action() -> None:
    llm = MockLlmClient([
        LlmResponse(content=""),
        LlmResponse(content='{"tool_name":"bma_create_object","arguments":{"name":"Cube","type":"MESH_CUBE"}}'),
    ])
    trace = DirectToolCallingStrategy().run(
        {"id": "geometry_001", "prompt": "create cube", "allowed_tools": ["bma_create_object"]},
        _config(),
        llm,
        MockToolExecutor(results={"bma_create_object": {"name": "Cube"}}),
        AgentToolContext(run_id="direct-retry", task_id="geometry_001"),
        Path("."),
    )
    assert trace.success is True
    assert any(step.metadata.get("direct_no_action_retry") for step in trace.steps)


def test_direct_no_action_structured_error() -> None:
    llm = MockLlmClient([LlmResponse(content=""), LlmResponse(content="still nothing")])
    trace = DirectToolCallingStrategy().run(
        {"id": "geometry_001", "prompt": "create cube", "allowed_tools": ["bma_create_object"]},
        _config(),
        llm,
        MockToolExecutor(),
        AgentToolContext(run_id="direct-no-action", task_id="geometry_001"),
        Path("."),
    )
    assert trace.success is False
    assert trace.error == "DirectNoAction"
    assert trace.structured_error is not None
    assert trace.structured_error["error_type"] == "DirectNoAction"
    assert trace.steps[-1].step_type == AgentStepType.ERROR
