from benchmark.experiments.third_repetition import evaluate_third_repetition_policy


def _row(repetition: int, pass_type: str = "clean_pass", **extra: str) -> dict[str, str]:
    base = {
        "repetition": str(repetition),
        "pass_type": pass_type,
        "task_id": "geometry_001_basic_primitives",
        "strategy": "react",
        "model_id": "google/gemini-2.5-flash-lite",
        "is_infra_failure": "false",
    }
    base.update(extra)
    return base


def test_recommends_third_repetition_on_large_success_delta() -> None:
    policy = {
        "enabled": True,
        "run_full_third_repetition": False,
        "trigger_conditions": {
            "model_reported_success_delta_between_repetitions_gt": 0.10,
        },
    }
    rows = [_row(1, "clean_pass") for _ in range(10)] + [_row(2, "runtime_error") for _ in range(10)]
    result = evaluate_third_repetition_policy(policy, rows)
    assert result["evaluated"] is True
    assert result["recommend_third_repetition"] is True
