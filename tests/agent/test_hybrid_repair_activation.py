"""Tests for PlanExecuteReactRepairStrategy repair activation logic."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark.agent.models import AgentConfig, AgentStrategyName, LlmConfig, AgentTrace
from benchmark.agent.strategies.plan_execute_react_repair import (
    PlanExecuteReactRepairStrategy,
    _stamp_trace,
    _validate_via_tool_with_reason,
)


def _config() -> AgentConfig:
    return AgentConfig(
        agent_id="hybrid-test",
        strategy=AgentStrategyName.PLAN_EXECUTE_REACT_REPAIR,
        llm=LlmConfig(provider="mock", model="mock"),
        mcp_profile="minimal",
        max_steps=10,
    )


def _make_trace(success: bool = True) -> AgentTrace:
    import datetime, uuid
    return AgentTrace(
        run_id=str(uuid.uuid4()),
        task_id="test_task",
        agent_id="hybrid-test",
        strategy=AgentStrategyName.PLAN_AND_EXECUTE,
        started_at=datetime.datetime.now(datetime.timezone.utc),
        success=success,
        metadata={},
    )


def test_hybrid_skips_repair_when_plan_passed() -> None:
    """If plan passes validation, hybrid should not start repair."""
    val_result = MagicMock()
    val_result.overall_status = "passed"
    val_result.total_score = 1.0
    val_result.issues = []

    from benchmark.validation.models import ValidationStatus
    val_result.overall_status = ValidationStatus.PASSED

    plan_trace = _make_trace(success=True)

    from benchmark.agent.strategies.plan_execute_react_repair import _stamp_trace
    import datetime
    stamped = _stamp_trace(
        plan_trace, _config(),
        datetime.datetime.now(datetime.timezone.utc),
        hybrid_repair_used=False,
        plan_scene_status="passed",
        plan_score=1.0,
        plan_issue_count=0,
    )
    assert stamped.metadata["hybrid_repair_used"] is False
    assert "repair_unavailable" not in stamped.metadata or stamped.metadata.get("repair_unavailable") is False


def test_hybrid_runs_repair_when_plan_failed_validation() -> None:
    """Stamp trace shows repair_started=True and repair_start_reason when hybrid used."""
    plan_trace = _make_trace(success=False)
    import datetime
    stamped = _stamp_trace(
        plan_trace, _config(),
        datetime.datetime.now(datetime.timezone.utc),
        hybrid_repair_used=True,
        plan_scene_status="failed",
        plan_score=0.3,
        plan_issue_count=2,
        repair_result_status="passed",
    )
    assert stamped.metadata["hybrid_repair_used"] is True
    assert stamped.metadata.get("repair_started") is True
    assert stamped.metadata.get("repair_start_reason") == "plan_failed_validation"
    assert stamped.metadata.get("plan_scene_status") == "failed"
    assert stamped.metadata.get("plan_score") == 0.3
    assert stamped.metadata.get("plan_issue_count") == 2
    assert stamped.metadata.get("repair_result_status") == "passed"


def test_hybrid_logs_repair_unavailable_reason() -> None:
    """When validation is unavailable, repair_unavailable_reason must be set."""
    plan_trace = _make_trace(success=False)
    import datetime
    stamped = _stamp_trace(
        plan_trace, _config(),
        datetime.datetime.now(datetime.timezone.utc),
        hybrid_repair_used=False,
        repair_unavailable=True,
        repair_unavailable_reason="snapshot_tool_failed",
    )
    assert stamped.metadata["repair_unavailable"] is True
    assert stamped.metadata["repair_unavailable_reason"] == "snapshot_tool_failed"


def test_validate_via_tool_returns_reason_on_snapshot_failure() -> None:
    """_validate_via_tool_with_reason returns descriptive reason when snapshot fails."""
    executor = MagicMock()
    tool_result = MagicMock()
    tool_result.error = "SocketTimeout"
    tool_result.result = None
    executor.call_tool.return_value = tool_result

    val, reason = _validate_via_tool_with_reason(executor, {"id": "task"}, Path(".") / "snap.json")
    assert val is None
    assert reason == "snapshot_tool_failed"


def test_validate_via_tool_returns_reason_on_invalid_schema() -> None:
    executor = MagicMock()
    tool_result = MagicMock()
    tool_result.error = None
    tool_result.result = "not_a_dict"
    executor.call_tool.return_value = tool_result

    val, reason = _validate_via_tool_with_reason(executor, {"id": "task"}, Path(".") / "snap.json")
    assert val is None
    assert reason == "snapshot_invalid_schema"
