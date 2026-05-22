from benchmark.experiments.acceptance_policy import evaluate_acceptance_policy


def _row(**kwargs: str) -> dict[str, str]:
    base = {
        "pass_type": "clean_pass",
        "error_type": "",
        "is_infra_failure": "false",
        "task_id": "geometry_001_basic_primitives",
    }
    base.update(kwargs)
    return base


def test_accept_for_report_when_gates_pass() -> None:
    policy = {
        "accept_for_report_if": {
            "structural_validity": "passed",
            "readiness_ok": True,
            "unclassified_error": 0,
            "reported_success_rate_min": 0.80,
        }
    }
    rows = [_row() for _ in range(10)]
    result = evaluate_acceptance_policy(
        policy,
        rows=rows,
        gate_result={"readiness_ok": True, "failed_gates": []},
        structural_validity="passed",
        planned_runs=10,
        expected_runs=10,
    )
    assert result["decision_level"] == "accept_for_report"


def test_rerun_required_on_unclassified_errors() -> None:
    policy = {
        "rerun_required_if": {"unclassified_error_gt": 0},
        "accept_for_report_if": {"structural_validity": "passed", "readiness_ok": True},
    }
    rows = [_row(error_type="UnclassifiedError")]
    result = evaluate_acceptance_policy(
        policy,
        rows=rows,
        gate_result={"readiness_ok": True, "failed_gates": []},
        structural_validity="passed",
        planned_runs=1,
        expected_runs=1,
    )
    assert result["decision_level"] == "rerun_required"
    assert "unclassified_error" in result["reasons"]
