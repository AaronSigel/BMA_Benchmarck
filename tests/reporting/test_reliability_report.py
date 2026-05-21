from __future__ import annotations

from benchmark.analysis.models import ExperimentAnalysisResult, ExperimentSummary, RunAnalysisResult
from benchmark.analysis.report_builder import _model_failures_after_infra_table, _reliability_table
from benchmark.analysis.report_bundle_validator import _evaluate_readiness_gates


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
