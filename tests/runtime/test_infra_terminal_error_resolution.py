"""Tests for infra-aware terminal error resolution."""

from __future__ import annotations

import datetime

from benchmark.agent.models import AgentStepType, AgentStrategyName, AgentTrace
from benchmark.agent.runtime import _resolve_terminal_error_from_trace
from benchmark.runner.error_classification import classify_failure, is_infra_parse_error


def _trace_with_infra_tool_failure(*, react_error_type: str | None = "ReactMaxSteps") -> AgentTrace:
    started = datetime.datetime.now(datetime.timezone.utc)
    trace = AgentTrace(
        run_id="run-infra",
        task_id="task",
        agent_id="agent",
        strategy=AgentStrategyName.REACT,
        started_at=started,
        success=False,
        error="ReactMaxSteps",
        metadata={"react_error_type": react_error_type},
    )
    return trace.add_step(
        AgentStepType.TOOL_CALL,
        tool_name="bma_assign_material",
        error="Empty response from Blender socket for tool 'bma_assign_material'",
        metadata={
            "tool_error_type": "EmptySocketResponse",
            "failure_stage": "socket_response",
        },
    )


def test_resolve_terminal_error_prefers_infra_tool_failure_over_react_max_steps() -> None:
    payload = _resolve_terminal_error_from_trace(_trace_with_infra_tool_failure())
    assert payload is not None
    assert payload["error_type"] == "EmptySocketResponse"
    assert payload["is_infra_failure"] is True
    assert payload.get("error_class") == "INFRA_ERROR"


def test_invalid_json_on_socket_stage_is_infra() -> None:
    result = classify_failure(
        error_type="InvalidJsonResponse",
        failure_stage="socket_response",
    )
    assert result.is_infra_failure is True
    assert is_infra_parse_error("InvalidJsonResponse", "socket_response") is True
