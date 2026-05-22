"""Tests for invalid_tool_response_export_max readiness gate semantics."""

from __future__ import annotations

from benchmark.analysis.report_bundle_validator import _evaluate_readiness_gates


def _csv_row(
    task_id: str,
    pass_type: str,
    error_type: str = "",
    *,
    strategy: str = "direct",
    is_infra_failure: str = "",
    run_id: str = "run_1",
    artifact_dir: str = "/tmp/run_1",
) -> dict[str, str]:
    return {
        "run_id": run_id,
        "task_id": task_id,
        "task_category": "export" if task_id.startswith("export_") else "geometry",
        "pass_type": pass_type,
        "error_type": error_type,
        "error_class": "INFRA_ERROR" if is_infra_failure.lower() == "true" else "",
        "is_infra_failure": is_infra_failure,
        "strategy": strategy,
        "agent_id": "test_agent",
        "mcp_profile": "minimal",
        "validation_issues": "",
        "agent_issues": "",
        "artifact_dir": artifact_dir,
        "error": "",
    }


def test_invalid_tool_response_gate_reports_affected_runs() -> None:
    gates = {"invalid_tool_response_export_max": 0}
    rows = [
        _csv_row(
            "export_001_blend_file",
            "runtime_error",
            "InvalidJsonResponse",
            run_id="run_a",
            artifact_dir="/tmp/run_a",
        ),
        _csv_row("export_002_glb_file", "clean_pass", ""),
    ]
    result = _evaluate_readiness_gates(gates, rows)
    assert result["readiness_ok"] is False
    failed = next(fg for fg in result["failed_gates"] if fg["name"] == "invalid_tool_response_export_max")
    assert failed["actual"] == 1
    assert len(failed["affected_runs"]) == 1
    assert failed["affected_runs"][0]["run_id"] == "run_a"
    assert failed["affected_runs"][0]["error_type"] == "InvalidJsonResponse"
    assert failed["affected_runs"][0]["source_file"] == "/tmp/run_a/run_result.json"


def test_invalid_tool_response_gate_does_not_trigger_without_run_level_evidence() -> None:
    gates = {"invalid_tool_response_export_max": 0}
    rows = [
        _csv_row("export_001_blend_file", "runtime_error", "EmptySocketResponse", is_infra_failure="true"),
        _csv_row("export_002_glb_file", "soft_pass", "ToolError"),
        _csv_row("geometry_001_basic_primitives", "clean_pass", ""),
    ]
    result = _evaluate_readiness_gates(gates, rows)
    assert result["readiness_ok"] is True
    assert result["failed_gates"] == []


def test_readiness_summary_consistent_with_summary_csv() -> None:
    gates = {
        "invalid_tool_response_export_max": 0,
        "empty_socket_response_count_max": 1,
    }
    rows = [
        _csv_row("export_001_blend_file", "runtime_error", "EmptySocketResponse", run_id="run_infra"),
        _csv_row("export_002_glb_file", "runtime_error", "InvalidToolResponse", run_id="run_invalid"),
    ]
    result = _evaluate_readiness_gates(gates, rows)
    assert result["readiness_ok"] is False
    names = {fg["name"] for fg in result["failed_gates"]}
    assert "invalid_tool_response_export_max" in names
    assert "empty_socket_response_count_max" not in names
    invalid_gate = next(fg for fg in result["failed_gates"] if fg["name"] == "invalid_tool_response_export_max")
    assert invalid_gate["affected_runs"][0]["error_type"] == "InvalidToolResponse"
    assert invalid_gate["affected_runs"][0]["run_id"] == "run_invalid"
