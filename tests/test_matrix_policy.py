from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.matrix import load_matrix
from benchmark.experiments.matrix_policy import (
    apply_generation_profile,
    order_runs,
    resolve_matrix_policy,
    validate_matrix_policy,
)
from benchmark.agent.models import LlmConfig, LlmProvider
from benchmark.runner.models import RunConfig, ExecutionMode
from pathlib import Path


def test_resolve_matrix_policy_includes_generation_profile() -> None:
    matrix = load_matrix("configs/matrices/final_multimodel_openrouter_v1.yaml")
    policy = resolve_matrix_policy(matrix)
    assert policy["matrix_id"] == "final_multimodel_openrouter_v1"
    assert policy["generation_profile"]["max_tokens"] == 6144
    assert "strategy_limits" in policy
    assert "direct_tool_calling" in policy["strategy_limits"]


def test_apply_generation_profile_overrides_max_tokens() -> None:
    llm = LlmConfig(provider=LlmProvider.OPENROUTER, model="test", max_tokens=2048)
    updated = apply_generation_profile(llm, {"apply_to_all_models": True, "max_tokens": 6144, "temperature": 0.2})
    assert updated.max_tokens == 6144


def test_order_runs_stratified_interleaved_is_deterministic() -> None:
    runs = [
        RunConfig(
            run_id=f"run-{i}",
            task_id=f"task_{i % 2}",
            execution_mode=ExecutionMode.AGENT_MCP,
            artifacts_dir=Path("."),
            output_dir=Path("."),
            metadata={
                "model_id": f"model-{i % 2}",
                "repetition": (i % 2) + 1,
                "agent_id": "agent",
                "mcp_profile": "minimal",
            },
        )
        for i in range(6)
    ]
    policy = {"mode": "stratified_interleaved", "seed": 42, "stratify_by": ["model_id", "repetition"]}
    ordered_a = order_runs(runs, policy)
    ordered_b = order_runs(runs, policy)
    assert [r.run_id for r in ordered_a] == [r.run_id for r in ordered_b]
    assert len(ordered_a) == 6


def test_validate_matrix_policy_expected_runs() -> None:
    matrix = load_matrix("configs/matrices/final_multimodel_openrouter_v1.yaml")
    policy = resolve_matrix_policy(matrix)
    config = generate_experiment_config(matrix)
    issues = validate_matrix_policy(policy, planned_runs=len(config.runs))
    assert issues == []
