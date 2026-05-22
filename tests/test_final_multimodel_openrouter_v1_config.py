from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.matrix import load_matrix
from benchmark.runner.models import ExecutionMode


def test_final_multimodel_generates_3600_runs() -> None:
    matrix = load_matrix("configs/matrices/final_multimodel_openrouter_v1.yaml")
    config = generate_experiment_config(matrix)
    assert len(config.runs) == 3600
    assert config.experiment_id == "final_multimodel_openrouter_v1"
    assert all(run.execution_mode is ExecutionMode.AGENT_MCP for run in config.runs)


def test_final_multimodel_propagates_matrix_policy_to_runs() -> None:
    matrix = load_matrix("configs/matrices/final_multimodel_openrouter_v1.yaml")
    config = generate_experiment_config(matrix)
    run = config.runs[0]
    assert "matrix_policy" in run.metadata
    assert run.metadata["generation_profile"]["max_tokens"] == 6144
    assert "readiness_gates" in run.metadata
    assert config.metadata["matrix_policy"]["matrix_id"] == "final_multimodel_openrouter_v1"


def test_final_multimodel_strategy_limits_include_all_agents() -> None:
    policy = load_matrix("configs/matrices/final_multimodel_openrouter_v1.yaml").metadata["strategy_limits"]
    assert "direct_tool_calling" in policy
    assert "plan_and_execute" in policy
    assert "react" in policy
    assert "plan_execute_react_repair" in policy
