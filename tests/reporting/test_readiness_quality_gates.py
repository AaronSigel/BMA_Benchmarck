"""Tests for readiness quality gates vs structural bundle validation."""
from __future__ import annotations

from benchmark.analysis.report_bundle_validator import (
    _evaluate_readiness_gates,
    _finish_result,
    validate_report_bundle_result,
)


def _row(task_id: str, pass_type: str, **extra: str) -> dict[str, str]:
    return {
        "task_id": task_id,
        "pass_type": pass_type,
        "strategy": extra.get("strategy", "react"),
        "error_type": extra.get("error_type", ""),
        "validation_issues": extra.get("validation_issues", ""),
        "react_repair_steps": extra.get("react_repair_steps", "0"),
        "hybrid_repair_used": extra.get("hybrid_repair_used", "false"),
        "repair_unavailable_reason": extra.get("repair_unavailable_reason", ""),
    }


def test_readiness_fails_when_lighting_below_threshold() -> None:
    rows = [
        _row("lighting_001_area_light", "failed_validation"),
        _row("lighting_002_sun_light", "failed_validation"),
        _row("lighting_003_three_point_lighting", "clean_pass"),
        _row("geometry_001_basic_primitives", "clean_pass"),
    ]
    result = _evaluate_readiness_gates({"lighting_success_rate_min": 0.70}, rows)
    assert result["readiness_ok"] is False
    assert any(gate["name"] == "lighting_success_rate_min" for gate in result["failed_gates"])


def test_readiness_fails_when_export_below_threshold() -> None:
    rows = [
        _row("export_001_blend_file", "clean_pass"),
        _row("export_002_glb_file", "failed_validation"),
        _row("export_002_glb_file", "failed_validation"),
    ]
    result = _evaluate_readiness_gates({"export_success_rate_min": 0.75}, rows)
    assert result["readiness_ok"] is False
    assert any(gate["name"] == "export_success_rate_min" for gate in result["failed_gates"])


def test_structural_validity_can_pass_when_readiness_fails(tmp_path) -> None:
    result = _finish_result(
        tmp_path,
        [{"name": "bundle_directory", "status": "passed"}],
        write_result=False,
        gate_result={
            "readiness_ok": False,
            "failed_gates": [
                {
                    "name": "lighting_success_rate_min",
                    "expected": 0.70,
                    "actual": 0.0,
                    "severity": "blocking",
                }
            ],
            "warning_gates": [],
            "gates_checked": ["lighting_success_rate_min"],
        },
    )
    assert result["structural_validity"] == "passed"
    assert result["readiness_ok"] is False


def test_failed_gates_include_actual_expected_severity() -> None:
    rows = [_row("geometry_001_basic_primitives", "failed_validation")]
    result = _evaluate_readiness_gates({"geometry_success_rate_min": 0.80}, rows)
    gate = next(item for item in result["failed_gates"] if item["name"] == "geometry_success_rate_min")
    assert gate["expected"] == 0.80
    assert gate["actual"] == 0.0
    assert gate["severity"] == "blocking"
