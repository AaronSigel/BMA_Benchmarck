"""Tests for ReAct repair activation metadata and early-stop logic."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark.agent.models import AgentConfig, AgentStrategyName, LlmConfig
from benchmark.agent.strategies.react import (
    _build_step_context,
    _repair_activation_metadata,
)


def _config(max_steps: int = 5) -> AgentConfig:
    return AgentConfig(
        agent_id="test",
        strategy=AgentStrategyName.REACT,
        llm=LlmConfig(provider="mock", model="mock"),
        mcp_profile="minimal",
        max_steps=max_steps,
        stop_after_scene_passed=True,
        detect_no_progress=False,
    )


def _make_val_result(status: str = "failed", issue_codes: list[str] | None = None):
    issue_codes = issue_codes if issue_codes is not None else ["object_missing"]
    val = MagicMock()
    val.overall_status = status
    val.total_score = 0.0 if status == "failed" else 1.0
    issues = []
    for code in issue_codes:
        issue = MagicMock()
        issue.code = code
        issue.message = f"{code} issue"
        issue.severity = "error"
        issues.append(issue)
    val.issues = issues
    return val


class _Task:
    id = "geometry_001"
    category = "geometry"


def test_react_logs_fallback_reason_when_validation_missing() -> None:
    """When val_result is None, fallback_reason must be 'no_validation_result'."""
    ctx, repair, activation = _build_step_context(None, _Task(), None, [])
    assert activation["fallback_reason"] == "no_validation_result"
    assert activation["validation_result_available"] is False


def test_react_uses_mapper_when_validation_issue_available() -> None:
    """When val_result has issues and task_obj is present, mapper should be attempted."""
    val_result = _make_val_result(status="failed", issue_codes=["object_missing"])
    task_obj = MagicMock()
    task_obj.id = "geometry_001"

    # Patch at the issue_action_mapper module level (where _build_step_context imports from)
    with patch("benchmark.agent.strategies.issue_action_mapper.select_top_issue") as mock_top, \
         patch("benchmark.agent.strategies.issue_action_mapper.map_issue_to_repair") as mock_repair:
        mock_top.return_value = val_result.issues[0]
        mock_repair.return_value = None  # mapper returns None → fallback_reason

        ctx, repair_action, activation = _build_step_context(val_result, task_obj, None, [])

    assert activation["validation_result_available"] is True
    assert activation["issue_count"] == 1
    assert activation["top_issue_code"] == "object_missing"
    assert activation["task_obj_loaded"] is True


def test_repair_activation_metadata_no_issues() -> None:
    """With val_result present but no issues, fallback_reason should be 'no_issues'."""
    val_result = _make_val_result(status="passed", issue_codes=[])
    activation = _repair_activation_metadata(
        val_result, _Task(), None, None, [],
        deterministic_executed=False,
        fallback_reason="no_issues",
    )
    assert activation["fallback_reason"] == "no_issues"
    assert activation["issue_count"] == 0


def test_repair_activation_tool_not_available_in_profile() -> None:
    """When repair tool is not in tool_contracts, fallback should reflect that."""
    val_result = _make_val_result()
    task_obj = MagicMock()
    top_issue = val_result.issues[0]

    repair_action = MagicMock()
    repair_action.tool_name = "bma_some_unavailable_tool"
    repair_action.arguments_template = {"name": "Cube"}

    activation = _repair_activation_metadata(
        val_result, task_obj, top_issue, repair_action,
        [],  # empty tool_contracts → tool not available
        deterministic_executed=False,
        fallback_reason=None,
    )
    assert activation["repair_mapped"] is True
    assert activation["repair_tool"] == "bma_some_unavailable_tool"
    assert activation["fallback_reason"] == "tool_not_allowed_in_profile"
    assert activation["deterministic_repair_executed"] is False


def test_react_early_stops_after_passed_validation() -> None:
    """ReactStrategy should stop early when scene_validator_fn returns passed."""
    from benchmark.agent.llm.base import LlmResponse
    from benchmark.agent.strategies.react import ReactStrategy
    from benchmark.agent.tool_executor import MockToolExecutor

    llm = MagicMock()
    llm.complete.return_value = LlmResponse(
        content='{"thought":"done","action":null,"finish":true}'
    )

    strategy = ReactStrategy()

    call_count = 0

    def mock_validator(snap_path: Path):
        nonlocal call_count
        call_count += 1
        val = _make_val_result(status="passed", issue_codes=[])
        val.total_score = 1.0
        return True, 1.0, val

    strategy.scene_validator_fn = mock_validator

    trace = strategy.run(
        {"id": "geo_001", "category": "geometry"},
        _config(max_steps=10),
        llm,
        MockToolExecutor(),
        MagicMock(run_id="r1", task_id="geo_001"),
        Path("."),
    )

    assert trace.success is True
    assert call_count >= 1
