from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from benchmark.runner.errors import RunnerConfigError
from benchmark.runner.models import ExperimentConfig, RunConfig


def load_run_config(path: Path | str) -> RunConfig:
    data = _load_yaml_mapping(path, "run config")
    try:
        return RunConfig.model_validate(data)
    except ValidationError as error:
        raise RunnerConfigError(f"Invalid run config in {Path(path)}: {error}") from error


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    data = _load_yaml_mapping(path, "experiment config")
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as error:
        raise RunnerConfigError(f"Invalid experiment config in {Path(path)}: {error}") from error


def dump_run_config(config: RunConfig, path: Path | str) -> None:
    _dump_model(config, path, "run config")


def dump_experiment_config(config: ExperimentConfig, path: Path | str) -> None:
    _dump_model(config, path, "experiment config")


def _load_yaml_mapping(path: Path | str, label: str) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except OSError as error:
        raise RunnerConfigError(f"Failed to read {label} {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise RunnerConfigError(f"Failed to parse YAML {label} {config_path}: {error}") from error

    if not isinstance(data, dict):
        raise RunnerConfigError(
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
        raise RunnerConfigError(f"Failed to write {label} {config_path}: {error}") from error
