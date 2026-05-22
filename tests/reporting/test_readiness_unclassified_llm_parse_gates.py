from benchmark.analysis.report_bundle_validator import _evaluate_readiness_gates


def _row(**kwargs: str) -> dict[str, str]:
    base = {
        "pass_type": "runtime_error",
        "task_id": "geometry_001_basic_primitives",
        "task_category": "geometry",
        "strategy": "react",
        "is_infra_failure": "false",
    }
    base.update(kwargs)
    return base


def test_unclassified_error_max_gate() -> None:
    rows = [_row(error_type="UnclassifiedError")]
    result = _evaluate_readiness_gates({"unclassified_error_max": 0}, rows)
    assert result["readiness_ok"] is False
    assert any(g["name"] == "unclassified_error_max" for g in result["failed_gates"])


def test_llm_parse_error_rate_max_gate() -> None:
    rows = [_row(error_type="LlmParseError") for _ in range(5)] + [
        _row(error_type="", pass_type="clean_pass") for _ in range(5)
    ]
    result = _evaluate_readiness_gates({"llm_parse_error_rate_max": 0.02}, rows)
    assert result["readiness_ok"] is False
    assert any(g["name"] == "llm_parse_error_rate_max" for g in result["failed_gates"])
