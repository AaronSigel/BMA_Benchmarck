"""Tests for hybrid repair error handling and outcome classification."""
from __future__ import annotations

import datetime
from unittest.mock import MagicMock

from benchmark.agent.models import AgentConfig, AgentStrategyName, AgentTrace, LlmConfig
from benchmark.agent.strategies.plan_execute_react_repair import (
    _merge_traces,
    _stamp_trace,
    _validate_via_tool_with_reason,
)
from benchmark.validation.models import ValidationStatus


def _config() -> AgentConfig:
    return AgentConfig(
        agent_id="hybrid-test",
        strategy=AgentStrategyName.PLAN_EXECUTE_REACT_REPAIR,
        llm=LlmConfig(provider="mock", model="mock"),
        mcp_profile="minimal",
        max_steps=10,
    )


def _trace(success: bool = True, error: str | None = None, metadata: dict | None = None) -> AgentTrace:
    return AgentTrace(
        run_id="run-hybrid",
        task_id="task",
        agent_id="hybrid-test",
        strategy=AgentStrategyName.PLAN_EXECUTE_REACT_REPAIR,
        started_at=datetime.datetime.now(datetime.timezone.utc),
        success=success,
        error=error,
        metadata=metadata or {},
    )


def _validation(status: ValidationStatus, score: float = 0.5, issues: int = 1) -> MagicMock:
    val = MagicMock()
    val.overall_status = status
    val.total_score = score
    val.issues = [MagicMock()] * issues
    return val


def test_repair_error_with_passed_scene_becomes_soft_pass() -> None:
    plan = _trace(success=False)
    react = _trace(success=False, error="ReactInvalidAction", metadata={"scene_passed_but_agent_error": True})
    merged = _merge_traces(
        plan,
        react,
        _config(),
        "run-hybrid",
        "task",
        datetime.datetime.now(datetime.timezone.utc),
        post_repair_val=_validation(ValidationStatus.PASSED, score=1.0, issues=0),
    )
    assert merged.success is True
    assert merged.error is None


def test_repair_error_with_failed_scene_becomes_failed_validation() -> None:
    plan = _trace(success=False)
    react = _trace(success=False, error="ReactNoProgress")
    merged = _merge_traces(
        plan,
        react,
        _config(),
        "run-hybrid",
        "task",
        datetime.datetime.now(datetime.timezone.utc),
        post_repair_val=_validation(ValidationStatus.FAILED, score=0.2, issues=2),
    )
    assert merged.success is False
    assert merged.error == "ReactNoProgress"


def test_repair_runtime_without_snapshot_becomes_runtime_error() -> None:
    plan = _trace(success=False, error="ToolTimeout")
    react = _trace(success=False, error="ReactMaxSteps")
    merged = _merge_traces(
        plan,
        react,
        _config(),
        "run-hybrid",
        "task",
        datetime.datetime.now(datetime.timezone.utc),
        post_repair_val=None,
    )
    assert merged.success is False
    assert merged.error == "ReactMaxSteps"


def test_hybrid_repair_improvement_metadata() -> None:
    stamped = _stamp_trace(
        _trace(),
        _config(),
        datetime.datetime.now(datetime.timezone.utc),
        hybrid_repair_used=True,
        repair_started=True,
        repair_start_reason="failed_validation",
        repair_result_status="improved",
        repair_score_before=0.3,
        repair_score_after=0.7,
        repair_improved_score=True,
        repair_reduced_issue_count=True,
        repair_error_type="ReactNoProgress",
        repair_scene_status_after_error="failed",
    )
    assert stamped.metadata["repair_improved_score"] is True
    assert stamped.metadata["repair_reduced_issue_count"] is True
    assert stamped.metadata["repair_error_type"] == "ReactNoProgress"


def test_validate_via_tool_preserves_snapshot_invalid_schema_reason() -> None:
    executor = MagicMock()
    tool_result = MagicMock()
    tool_result.error = None
    tool_result.result = "not_a_dict"
    executor.call_tool.return_value = tool_result
    val, reason = _validate_via_tool_with_reason(executor, {"id": "task"}, __import__("pathlib").Path("snap.json"))
    assert val is None
    assert reason == "snapshot_invalid_schema"
