from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from benchmark.agent.errors import AgentConfigError
from benchmark.agent.models import AgentConfig


def load_agent_config(path: Path | str) -> AgentConfig:
    data = _load_yaml_mapping(path, "agent config")
    try:
        return AgentConfig.model_validate(data)
    except ValidationError as error:
        raise AgentConfigError(f"Invalid agent config in {Path(path)}: {error}") from error


def dump_agent_config(config: AgentConfig, path: Path | str) -> None:
    _dump_model(config, path, "agent config")


def load_agent_configs_from_dir(path: Path | str) -> list[AgentConfig]:
    config_dir = Path(path)
    if not config_dir.exists():
        raise AgentConfigError(f"Agent config directory does not exist: {config_dir}")
    if not config_dir.is_dir():
        raise AgentConfigError(f"Agent config path is not a directory: {config_dir}")

    configs: list[AgentConfig] = []
    for config_path in sorted(
        item for item in config_dir.iterdir() if item.suffix in {".yaml", ".yml"}
    ):
        configs.append(load_agent_config(config_path))
    return configs


def _load_yaml_mapping(path: Path | str, label: str) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except OSError as error:
        raise AgentConfigError(f"Failed to read {label} {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise AgentConfigError(f"Failed to parse YAML {label} {config_path}: {error}") from error

    if not isinstance(data, dict):
        raise AgentConfigError(
            f"{label.capitalize()} {config_path} must contain a YAML mapping at the top level"
        )
    return data


def _dump_model(config: BaseModel, path: Path | str, label: str) -> None:
    config_path = Path(path)
    data = config.model_dump(mode="json", exclude_none=True)
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
    except OSError as error:
        raise AgentConfigError(f"Failed to write {label} {config_path}: {error}") from error
