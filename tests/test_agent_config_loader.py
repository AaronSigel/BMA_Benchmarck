from pathlib import Path

import pytest
import yaml

from benchmark.agent.config_loader import (
    dump_agent_config,
    load_agent_config,
    load_agent_configs_from_dir,
)
from benchmark.agent.errors import AgentConfigError
from benchmark.agent.models import AgentConfig, AgentStrategyName, LlmConfig


def test_load_agent_config_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent_id": "agent-1",
                "strategy": "react",
                "mcp_profile": "minimal",
                "llm": {"provider": "mock", "model": "mock-model"},
            }
        ),
        encoding="utf-8",
    )

    config = load_agent_config(config_path)

    assert config.agent_id == "agent-1"
    assert config.strategy == AgentStrategyName.REACT
    assert config.llm == LlmConfig(provider="mock", model="mock-model")


def test_dump_agent_config_to_yaml(tmp_path: Path) -> None:
    config = AgentConfig(
        agent_id="agent-1",
        strategy=AgentStrategyName.DIRECT_TOOL_CALLING,
        llm=LlmConfig(provider="mock", model="mock-model"),
    )
    config_path = tmp_path / "nested" / "agent.yaml"

    dump_agent_config(config, config_path)
    loaded = load_agent_config(config_path)

    assert loaded == config


def test_load_agent_configs_from_dir_loads_yaml_files_in_name_order(tmp_path: Path) -> None:
    (tmp_path / "b.yaml").write_text(
        yaml.safe_dump(
            {
                "agent_id": "agent-b",
                "strategy": "react",
                "llm": {"provider": "mock", "model": "mock-b"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "a.yml").write_text(
        yaml.safe_dump(
            {
                "agent_id": "agent-a",
                "strategy": "direct_tool_calling",
                "llm": {"provider": "mock", "model": "mock-a"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "ignored.txt").write_text("not yaml", encoding="utf-8")

    configs = load_agent_configs_from_dir(tmp_path)

    assert [config.agent_id for config in configs] == ["agent-a", "agent-b"]


def test_load_agent_config_validation_error_includes_path(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent_id": "agent-1",
                "strategy": "react",
                "llm": {"provider": "ollama", "model": "llama"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AgentConfigError) as error:
        load_agent_config(config_path)

    assert str(config_path) in str(error.value)


def test_load_agent_config_requires_yaml_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- not\n- mapping\n", encoding="utf-8")

    with pytest.raises(AgentConfigError, match="must contain a YAML mapping"):
        load_agent_config(config_path)


def test_load_agent_configs_from_dir_requires_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    file_path = tmp_path / "agent.yaml"
    file_path.write_text("agent_id: agent\n", encoding="utf-8")

    with pytest.raises(AgentConfigError, match="does not exist"):
        load_agent_configs_from_dir(missing)
    with pytest.raises(AgentConfigError, match="not a directory"):
        load_agent_configs_from_dir(file_path)


def test_builtin_agent_configs_are_valid() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "agents"
    config_paths = sorted(config_dir.glob("*.yaml"))

    assert {path.name for path in config_paths} == {
        "mock_agent.yaml",
        "direct_openrouter.yaml",
        "react_openrouter.yaml",
        "plan_execute_openrouter.yaml",
        "direct_openai_compatible.yaml",
        "react_anthropic.yaml",
        "remote_agent_codex.yaml",
        "remote_agent_claude.yaml",
        "generic_http.yaml",
        "generic_command.yaml",
        "pilot_plan_openrouter_gemini_flash_lite.yaml",
    }
    assert not list(config_dir.glob("*ollama*.yaml"))

    configs = load_agent_configs_from_dir(config_dir)
    by_id = {config.agent_id: config for config in configs}

    assert by_id["mock_agent"].llm is not None
    assert by_id["mock_agent"].llm.api_key_env is None

    for config in configs:
        if config.llm is not None:
            assert config.llm.api_key_env is None or config.llm.api_key_env.endswith("_API_KEY")
            assert "api_key" not in config.llm.model_dump()
        if config.remote_agent is not None:
            assert config.llm is None
            dumped = config.remote_agent.model_dump()
            assert "model" not in dumped
            assert "api_key" not in dumped
