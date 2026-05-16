from pathlib import Path

import yaml

from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.matrix import load_matrix
from benchmark.runner.models import ExecutionMode


def _load_agent_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_api_models_matrix_is_opt_in_and_documents_cost_risk() -> None:
    matrix = load_matrix("configs/matrices/api_models_matrix.yaml")

    assert matrix.metadata["opt_in"] is True
    assert matrix.metadata["requires_api_keys"] is True
    assert matrix.metadata["pytest_marker"] == "api_e2e"
    assert "paid external APIs" in matrix.metadata["cost_warning"]


def test_api_models_matrix_covers_required_providers_and_strategies() -> None:
    matrix = load_matrix("configs/matrices/api_models_matrix.yaml")

    assert matrix.metadata["providers"] == ["openrouter", "openai_compatible", "anthropic"]
    assert matrix.metadata["strategies"] == [
        "direct_tool_calling",
        "react",
        "plan_and_execute",
    ]
    assert matrix.agents.ids == [
        "direct_openrouter",
        "direct_openai_compatible",
        "react_openrouter",
        "react_anthropic",
        "plan_execute_openrouter",
    ]


def test_api_models_matrix_uses_safe_mcp_profiles() -> None:
    matrix = load_matrix("configs/matrices/api_models_matrix.yaml")

    assert matrix.mcp_profiles == ["no_python", "inspection_enabled"]
    assert "python_enabled" not in matrix.mcp_profiles
    assert "full" not in matrix.mcp_profiles


def test_api_models_matrix_agent_keys_are_env_only() -> None:
    config = generate_experiment_config(load_matrix("configs/matrices/api_models_matrix.yaml"))
    agent_paths = {run.agent_config_path for run in config.runs}

    assert agent_paths
    for path in agent_paths:
        assert path is not None
        agent = _load_agent_yaml(path)
        llm = agent.get("llm", {})
        assert "api_key" not in llm
        assert llm.get("api_key_env", "").endswith("_API_KEY")


def test_api_models_matrix_generates_reproducible_run_configs_without_running_apis() -> None:
    matrix = load_matrix("configs/matrices/api_models_matrix.yaml")

    first = generate_experiment_config(matrix)
    second = generate_experiment_config(matrix)

    expected_runs = 5 * 5 * 2 * 1 * 1
    assert len(first.runs) == expected_runs
    assert first == second
    assert len({run.run_id for run in first.runs}) == expected_runs
    assert all(run.execution_mode is ExecutionMode.AGENT_MCP for run in first.runs)
    assert first.runs[0].run_id == (
        "api_models_matrix__geometry_001_basic_primitives__direct_openrouter__no_python__r1"
    )
