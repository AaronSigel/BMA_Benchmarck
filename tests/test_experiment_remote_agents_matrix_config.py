from pathlib import Path

import yaml

from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.matrix import load_matrix
from benchmark.runner.models import ExecutionMode


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_remote_agents_matrix_is_opt_in() -> None:
    matrix = load_matrix("configs/matrices/remote_agents_matrix.yaml")

    assert matrix.metadata["opt_in"] is True
    assert matrix.metadata["requires_remote_agent"] is True
    assert matrix.metadata["pytest_marker"] == "remote_agent_e2e"
    assert matrix.agents.include_remote_agents is True


def test_remote_agents_matrix_contains_expected_agents() -> None:
    matrix = load_matrix("configs/matrices/remote_agents_matrix.yaml")

    assert matrix.agents.ids == [
        "remote_agent_codex",
        "remote_agent_claude",
        "generic_http",
        "generic_command",
    ]
    assert matrix.metadata["remote_agent_providers"] == [
        "codex",
        "claude_code",
        "generic_http",
        "generic_command",
    ]


def test_remote_agent_configs_are_valid_placeholders_without_inline_secrets() -> None:
    config = generate_experiment_config(load_matrix("configs/matrices/remote_agents_matrix.yaml"))

    assert {run.agent_config_path for run in config.runs} == {
        Path("configs/agents/remote_agent_codex.yaml"),
        Path("configs/agents/remote_agent_claude.yaml"),
        Path("configs/agents/generic_http.yaml"),
        Path("configs/agents/generic_command.yaml"),
    }
    for path in {run.agent_config_path for run in config.runs}:
        assert path is not None
        agent = _load_yaml(path)
        remote_agent = agent["remote_agent"]
        assert agent["strategy"] == "remote_agent"
        assert "api_key" not in remote_agent


def test_remote_agents_matrix_generates_configs_without_running_agents() -> None:
    config = generate_experiment_config(load_matrix("configs/matrices/remote_agents_matrix.yaml"))

    assert len(config.runs) == 4
    assert [run.run_id for run in config.runs] == [
        "remote_agents_matrix__geometry_001_basic_primitives__remote_agent_codex__minimal__r1",
        "remote_agents_matrix__geometry_001_basic_primitives__remote_agent_claude__minimal__r1",
        "remote_agents_matrix__geometry_001_basic_primitives__generic_http__minimal__r1",
        "remote_agents_matrix__geometry_001_basic_primitives__generic_command__minimal__r1",
    ]
    assert all(run.execution_mode is ExecutionMode.REMOTE_AGENT for run in config.runs)
    assert len({run.output_dir for run in config.runs}) == 4
