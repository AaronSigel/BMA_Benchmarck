from pathlib import Path

import pytest
import yaml

from benchmark.experiments.matrix import (
    ExperimentMatrixError,
    load_agent_pool,
    select_agents,
    select_agents_by_ids,
    select_agents_by_strategy,
)
from benchmark.experiments.models import ExperimentMatrix


def write_agent_config(
    directory: Path,
    filename: str,
    agent_id: str,
    strategy: str,
) -> None:
    (directory / filename).write_text(
        yaml.safe_dump(
            {
                "agent_id": agent_id,
                "strategy": strategy,
                "mcp_profile": "minimal",
                "llm": {"provider": "mock", "model": "mock"},
            }
        ),
        encoding="utf-8",
    )


def make_agent_pool(tmp_path: Path) -> dict[str, dict]:
    write_agent_config(tmp_path, "01-direct.yaml", "direct_agent", "direct_tool_calling")
    write_agent_config(tmp_path, "02-react.yaml", "react_agent", "react")
    write_agent_config(tmp_path, "03-plan.yaml", "plan_agent", "plan_and_execute")
    write_agent_config(tmp_path, "04-mock.yaml", "mock_agent", "direct_tool_calling")
    write_agent_config(tmp_path, "05-remote.yaml", "remote_agent_codex", "remote_agent")
    return load_agent_pool(tmp_path)


def test_load_agent_pool_reads_yaml_configs_in_stable_order(tmp_path: Path) -> None:
    pool = make_agent_pool(tmp_path)

    assert list(pool) == [
        "direct_agent",
        "react_agent",
        "plan_agent",
        "mock_agent",
        "remote_agent_codex",
    ]
    assert pool["direct_agent"]["config_path"] == tmp_path / "01-direct.yaml"


def test_select_agents_by_strategy_supports_direct_react_and_plan(tmp_path: Path) -> None:
    pool = make_agent_pool(tmp_path)

    selected = select_agents_by_strategy(
        pool,
        ["direct_tool_calling", "react", "plan_and_execute"],
    )

    assert [agent["agent_id"] for agent in selected] == [
        "direct_agent",
        "react_agent",
        "plan_agent",
        "mock_agent",
    ]


def test_select_agents_by_ids_can_select_mock_agent(tmp_path: Path) -> None:
    selected = select_agents_by_ids(make_agent_pool(tmp_path), ["mock_agent"])

    assert [agent["agent_id"] for agent in selected] == ["mock_agent"]


def test_select_agents_excludes_remote_agents_by_default(tmp_path: Path) -> None:
    matrix = ExperimentMatrix(matrix_id="smoke")

    selected = select_agents(matrix, make_agent_pool(tmp_path))

    assert "remote_agent_codex" not in [agent["agent_id"] for agent in selected]


def test_select_agents_can_include_remote_agents_opt_in(tmp_path: Path) -> None:
    matrix = ExperimentMatrix(
        matrix_id="remote",
        agents={"include_remote_agents": True},
    )

    selected = select_agents(matrix, make_agent_pool(tmp_path))

    assert "remote_agent_codex" in [agent["agent_id"] for agent in selected]


def test_select_agents_combines_id_and_strategy_as_intersection(tmp_path: Path) -> None:
    matrix = ExperimentMatrix(
        matrix_id="react_only",
        agents={
            "ids": ["direct_agent", "react_agent"],
            "strategies": ["react"],
        },
    )

    selected = select_agents(matrix, make_agent_pool(tmp_path))

    assert [agent["agent_id"] for agent in selected] == ["react_agent"]


def test_missing_agent_id_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ExperimentMatrixError, match="missing_agent"):
        select_agents_by_ids(make_agent_pool(tmp_path), ["missing_agent"])


def test_empty_agent_selection_is_configuration_error(tmp_path: Path) -> None:
    matrix = ExperimentMatrix(
        matrix_id="empty",
        agents={"strategies": ["remote_agent"]},
    )

    with pytest.raises(ExperimentMatrixError, match="agents"):
        select_agents(matrix, make_agent_pool(tmp_path))


def test_load_agent_pool_rejects_missing_agent_id(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text(
        yaml.safe_dump({"strategy": "direct_tool_calling"}),
        encoding="utf-8",
    )

    with pytest.raises(ExperimentMatrixError, match="agent_id"):
        load_agent_pool(tmp_path)


def test_builtin_agent_pool_supports_expected_stage_8_selections() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "agents"
    pool = load_agent_pool(config_dir)

    strategies = select_agents_by_strategy(
        pool,
        ["direct_tool_calling", "react", "plan_and_execute"],
    )
    mock = select_agents_by_ids(pool, ["mock_agent"])
    smoke = select_agents(ExperimentMatrix(matrix_id="smoke"), pool)

    assert {"direct_openrouter", "react_openrouter", "plan_execute_openrouter"} <= {
        agent["agent_id"] for agent in strategies
    }
    assert [agent["agent_id"] for agent in mock] == ["mock_agent"]
    assert all(agent["strategy"] != "remote_agent" for agent in smoke)
