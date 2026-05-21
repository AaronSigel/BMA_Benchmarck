"""Tests for pass_type assignment semantics.

Key invariants:
- clean_pass is forbidden when error_type is set
- clean_pass is forbidden when scene_passed_but_agent_error is true
- readiness gate fails when export has InvalidToolResponse
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from benchmark.analysis.run_analysis import _classify_pass_type
from benchmark.metrics.export import _pass_type, _scene_passed_but_agent_error
from benchmark.analysis.report_bundle_validator import _evaluate_readiness_gates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_result(
    run_status: str = "passed",
    scene_status: str = "passed",
    agent_status: str = "completed",
    error_type: str | None = None,
    issue_counts: dict | None = None,
):
    from benchmark.runner.models import RunStatus, AgentStatus, SceneStatus

    result = MagicMock()
    result.run_status = RunStatus(run_status) if hasattr(RunStatus, run_status.lower()) else MagicMock(value=run_status)
    result.status = result.run_status
    result.scene_status = MagicMock(value=scene_status) if scene_status else None
    result.agent_status = MagicMock(value=agent_status) if agent_status else None

    structured_error = {}
    if error_type:
        structured_error["error_type"] = error_type

    validation = {}
    if issue_counts is not None:
        validation["issue_counts"] = issue_counts

    result.summary = {
        "structured_error": structured_error,
        "validation": validation,
    }
    result.error = error_type or ""
    result.total_score = 1.0
    result.overall_status = "passed"
    return result


# ---------------------------------------------------------------------------
# test_clean_pass_never_has_error_type
# ---------------------------------------------------------------------------

def test_clean_pass_never_has_error_type() -> None:
    """When error_type is set, pass_type must not be clean_pass."""
    result = _run_result(
        run_status="passed",
        scene_status="passed",
        agent_status="completed",
        error_type="InvalidJsonResponse",
        issue_counts=None,
    )
    pt = _pass_type(result)
    assert pt != "clean_pass", f"clean_pass must not coexist with error_type, got {pt!r}"
    assert pt == "soft_pass"


def test_clean_pass_allowed_when_no_error_type() -> None:
    result = _run_result(
        run_status="passed",
        scene_status="passed",
        agent_status="completed",
        error_type=None,
        issue_counts=None,
    )
    pt = _pass_type(result)
    assert pt == "clean_pass"


def test_scene_passed_agent_error_is_not_clean_pass() -> None:
    """When scene passed but agent had an error status, pass_type must be soft_pass."""
    result = _run_result(
        run_status="passed",
        scene_status="passed",
        agent_status="max_steps_reached",
        error_type=None,
        issue_counts=None,
    )
    pt = _pass_type(result)
    assert pt != "clean_pass", f"got {pt!r}"
    assert pt == "soft_pass"


def test_runtime_error_on_agent_run_error() -> None:
    result = _run_result(run_status="error", scene_status="not_available")
    assert _pass_type(result) == "runtime_error"


def test_failed_validation_on_scene_failed() -> None:
    result = _run_result(run_status="passed", scene_status="failed")
    assert _pass_type(result) == "failed_validation"


def test_soft_pass_when_issue_counts_present() -> None:
    result = _run_result(
        run_status="passed",
        scene_status="passed",
        agent_status="completed",
        issue_counts={"dimension_mismatch": 1},
    )
    assert _pass_type(result) == "soft_pass"


# ---------------------------------------------------------------------------
# test_readiness_gate_fails_on_export_invalid_tool_response
# ---------------------------------------------------------------------------

def _csv_row(task_id: str, pass_type: str, error_type: str = "", strategy: str = "direct") -> dict:
    return {
        "task_id": task_id,
        "pass_type": pass_type,
        "error_type": error_type,
        "strategy": strategy,
        "validation_issues": "",
        "agent_issues": "",
        "error": "",
    }


def test_readiness_gate_fails_on_export_invalid_tool_response() -> None:
    gates = {"invalid_tool_response_export_max": 0}
    rows = [
        _csv_row("export_001_blend_file", "runtime_error", "InvalidJsonResponse"),
        _csv_row("export_002_glb_file", "runtime_error", "EmptySocketResponse"),
        _csv_row("geometry_001_basic_primitives", "clean_pass", ""),
    ]
    result = _evaluate_readiness_gates(gates, rows)
    assert result["readiness_ok"] is False
    names = [fg["name"] for fg in result["failed_gates"]]
    assert "invalid_tool_response_export_max" in names


def test_readiness_gate_passes_when_export_ok() -> None:
    gates = {"invalid_tool_response_export_max": 0, "clean_pass_with_error_type_max": 0}
    rows = [
        _csv_row("export_001_blend_file", "clean_pass", ""),
        _csv_row("export_002_glb_file", "soft_pass", ""),
        _csv_row("geometry_001_basic_primitives", "clean_pass", ""),
    ]
    result = _evaluate_readiness_gates(gates, rows)
    assert result["readiness_ok"] is True
    assert result["failed_gates"] == []


def test_clean_pass_with_error_type_gate() -> None:
    gates = {"clean_pass_with_error_type_max": 0}
    rows = [
        _csv_row("geometry_001", "clean_pass", "ReactMaxSteps"),
    ]
    result = _evaluate_readiness_gates(gates, rows)
    assert result["readiness_ok"] is False
    assert any(fg["name"] == "clean_pass_with_error_type_max" for fg in result["failed_gates"])


def test_run_analysis_classify_forbids_clean_pass_with_error_type() -> None:
    pt = _classify_pass_type("passed", "passed", "completed", [], error_type="ReactInvalidAction")
    assert pt == "soft_pass"


def test_readiness_gate_fails_on_zero_react_repair_steps() -> None:
    gates = {"require_react_repair_steps_on_validation_issues": True}
    rows = [
        {
            **_csv_row("geometry_001", "failed_validation", strategy="react"),
            "validation_issues": "object_missing:1",
            "react_repair_steps": "0",
        }
    ]
    result = _evaluate_readiness_gates(gates, rows)
    assert result["readiness_ok"] is False
    assert any(fg["name"] == "require_react_repair_steps_on_validation_issues" for fg in result["failed_gates"])
