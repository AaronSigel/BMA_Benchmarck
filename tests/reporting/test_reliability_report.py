from __future__ import annotations

import json

from benchmark.analysis.comparison import analyze_run_results, count_terminal_infra_from_runs
from benchmark.analysis.models import ExperimentAnalysisResult, ExperimentSummary, RunAnalysisResult
from benchmark.analysis.report_builder import _model_failures_after_infra_table, _reliability_table
from benchmark.analysis.report_bundle_validator import (
    _evaluate_readiness_gates,
    _model_failure_rate,
    _soft_success_diagnostic_rate,
)


def test_report_contains_infra_error_rate() -> None:
    analysis = ExperimentAnalysisResult(
        experiment_id="exp",
        runs=[],
        summary=ExperimentSummary(total_runs=2, infra_error_rate=0.5),
    )
    rows = _reliability_table(analysis, [])
    assert any(row[0] == "infra_error_rate" and row[1] != "" for row in rows)


def test_report_contains_model_failure_rate_excluding_infra() -> None:
    runs = [
        RunAnalysisResult(
            run_id="r1",
            task_id="t1",
            agent_id="a1",
            strategy="react",
            metrics={"is_model_failure": True, "is_infra_failure": False, "structured_error_type": "ReactNoProgress"},
        )
    ]
    rows = _model_failures_after_infra_table(runs)
    assert any(row[0] == "model_failure_rate_excluding_infra" for row in rows)


def test_readiness_fails_on_infra_error_rate() -> None:
    rows = [
        {"pass_type": "runtime_error", "is_infra_failure": "true", "error_type": "EmptySocketResponse"},
        {"pass_type": "runtime_error", "is_infra_failure": "true", "error_type": "ToolTimeout"},
    ]
    result = _evaluate_readiness_gates({"infra_error_rate_max": 0.05}, rows)
    assert result["readiness_ok"] is False
    assert any(g["name"] == "infra_error_rate_max" for g in result["failed_gates"])


def _soft_diagnostic_row(**extra: str) -> dict[str, str]:
    return {
        "task_id": "export_001_blend_file",
        "pass_type": "soft_pass",
        "scene_status": "passed",
        "strategy": "react",
        "error_type": "ReactMaxSteps",
        "is_model_failure": "true",
        "is_infra_failure": "false",
        "error_class": "SOFT_SUCCESS_DIAGNOSTIC",
        "diagnostic_only": "true",
        **extra,
    }


def test_model_failure_rate_excludes_soft_diagnostic() -> None:
    rows = [
        _soft_diagnostic_row(),
        {
            "task_id": "geometry_001_basic_primitives",
            "pass_type": "failed_validation",
            "scene_status": "failed",
            "error_type": "ReactNoProgress",
            "is_model_failure": "true",
            "is_infra_failure": "false",
        },
    ]
    assert _model_failure_rate(rows) == 0.5
    assert _soft_success_diagnostic_rate(rows) == 0.5


def test_readiness_model_failure_gate_ignores_soft_diagnostic() -> None:
    rows = [_soft_diagnostic_row() for _ in range(20)] + [
        {
            "task_id": "geometry_001_basic_primitives",
            "pass_type": "clean_pass",
            "scene_status": "passed",
            "error_type": "",
            "is_model_failure": "false",
            "is_infra_failure": "false",
        }
        for _ in range(80)
    ]
    result = _evaluate_readiness_gates({"model_failure_rate_max": 0.20}, rows)
    assert result["readiness_ok"] is True


def test_socket_timeout_is_warning_when_infra_rate_low() -> None:
    rows = [
        {
            "pass_type": "runtime_error",
            "is_infra_failure": "true",
            "error_type": "ToolTimeout",
        },
    ] + [
        {
            "pass_type": "clean_pass",
            "is_infra_failure": "false",
            "error_type": "",
        }
        for _ in range(139)
    ]
    result = _evaluate_readiness_gates(
        {
            "socket_timeout_count_max": 0,
            "infra_error_rate_max": 0.05,
        },
        rows,
    )
    assert result["readiness_ok"] is True
    assert any(g["name"] == "socket_timeout_count_max" for g in result["warning_gates"])
    assert not any(g["name"] == "socket_timeout_count_max" for g in result["failed_gates"])


def test_socket_timeout_blocks_when_infra_rate_high() -> None:
    rows = [
        {"pass_type": "runtime_error", "is_infra_failure": "true", "error_type": "ToolTimeout"},
        {"pass_type": "runtime_error", "is_infra_failure": "true", "error_type": "EmptySocketResponse"},
        {"pass_type": "clean_pass", "is_infra_failure": "false", "error_type": ""},
    ]
    result = _evaluate_readiness_gates(
        {
            "socket_timeout_count_max": 0,
            "infra_error_rate_max": 0.05,
        },
        rows,
    )
    assert result["readiness_ok"] is False
    assert any(g["name"] == "socket_timeout_count_max" for g in result["failed_gates"])


def test_build_summary_infra_counts_match_readiness_csv() -> None:
    runs = [
        RunAnalysisResult(
            run_id="r1",
            task_id="export_001_blend_file",
            agent_id="a1",
            strategy="react",
            metrics={"structured_error_type": "ToolTimeout", "is_infra_failure": True},
        ),
        RunAnalysisResult(
            run_id="r2",
            task_id="export_002_glb_file",
            agent_id="a2",
            strategy="react",
            metrics={"structured_error_type": "EmptySocketResponse", "is_infra_failure": True},
        ),
        RunAnalysisResult(
            run_id="r3",
            task_id="geometry_001_basic_primitives",
            agent_id="a3",
            strategy="direct_tool_calling",
            metrics={"structured_error_type": "EmptySocketResponse", "is_infra_failure": True},
        ),
    ]
    analysis = analyze_run_results(runs, experiment_id="exp")
    assert analysis.summary.infra_socket_timeouts == 1
    assert analysis.summary.infra_empty_socket_responses == 2

    csv_rows = [
        {"error_type": "ToolTimeout"},
        {"error_type": "EmptySocketResponse"},
        {"error_type": "EmptySocketResponse"},
    ]
    terminal = count_terminal_infra_from_runs(runs)
    assert terminal["socket_timeout_count"] == analysis.summary.infra_socket_timeouts
    assert terminal["empty_socket_response_count"] == analysis.summary.infra_empty_socket_responses

    gate_result = _evaluate_readiness_gates(
        {"socket_timeout_count_max": 0, "empty_socket_response_count_max": 0},
        csv_rows,
    )
    timeout_gate = next(g for g in gate_result["warning_gates"] if g["name"] == "socket_timeout_count_max")
    empty_gate = next(g for g in gate_result["warning_gates"] if g["name"] == "empty_socket_response_count_max")
    assert timeout_gate["actual"] == analysis.summary.infra_socket_timeouts
    assert empty_gate["actual"] == analysis.summary.infra_empty_socket_responses


def test_build_summary_infra_counts_match_readiness_csv() -> None:
    runs = [
        RunAnalysisResult(
            run_id="r1",
            task_id="export_001_blend_file",
            agent_id="a1",
            strategy="react",
            metrics={"structured_error_type": "ToolTimeout", "is_infra_failure": True},
        ),
        RunAnalysisResult(
            run_id="r2",
            task_id="export_002_glb_file",
            agent_id="a2",
            strategy="react",
            metrics={"structured_error_type": "EmptySocketResponse", "is_infra_failure": True},
        ),
        RunAnalysisResult(
            run_id="r3",
            task_id="geometry_001_basic_primitives",
            agent_id="a3",
            strategy="direct_tool_calling",
            metrics={"structured_error_type": "EmptySocketResponse", "is_infra_failure": True},
        ),
    ]
    analysis = analyze_run_results(runs, experiment_id="exp")
    assert analysis.summary.infra_socket_timeouts == 1
    assert analysis.summary.infra_empty_socket_responses == 2

    csv_rows = [
        {"error_type": "ToolTimeout"},
        {"error_type": "EmptySocketResponse"},
        {"error_type": "EmptySocketResponse"},
    ]
    terminal = count_terminal_infra_from_runs(runs)
    assert terminal["socket_timeout_count"] == analysis.summary.infra_socket_timeouts
    assert terminal["empty_socket_response_count"] == analysis.summary.infra_empty_socket_responses

    gate_result = _evaluate_readiness_gates(
        {"socket_timeout_count_max": 0, "empty_socket_response_count_max": 0},
        csv_rows,
    )
    timeout_gate = next(g for g in gate_result["warning_gates"] if g["name"] == "socket_timeout_count_max")
    empty_gate = next(g for g in gate_result["warning_gates"] if g["name"] == "empty_socket_response_count_max")
    assert timeout_gate["actual"] == analysis.summary.infra_socket_timeouts
    assert empty_gate["actual"] == analysis.summary.infra_empty_socket_responses
