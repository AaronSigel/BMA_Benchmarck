from pathlib import Path

import yaml
from pydantic import ValidationError

from benchmark.tasks.models import BenchmarkTask


class TaskLoadError(ValueError):
    """Raised when a benchmark task YAML file cannot be loaded."""


def load_task(path: Path | str) -> BenchmarkTask:
    task_path = Path(path)

    try:
        with task_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except OSError as error:
        raise TaskLoadError(f"Failed to read task file {task_path}: {error}") from error
    except yaml.YAMLError as error:
        raise TaskLoadError(f"Failed to parse YAML task file {task_path}: {error}") from error

    if not isinstance(data, dict):
        raise TaskLoadError(f"Task file {task_path} must contain a YAML mapping at the top level")

    try:
        return BenchmarkTask.model_validate(data)
    except ValidationError as error:
        raise TaskLoadError(f"Invalid benchmark task in {task_path}: {error}") from error


def load_tasks_from_dir(directory: Path | str, recursive: bool = True) -> list[BenchmarkTask]:
    tasks_dir = Path(directory)
    pattern = "**/*" if recursive else "*"
    yaml_paths = sorted(
        path
        for path in tasks_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )

    return [load_task(path) for path in yaml_paths]


def dump_task(task: BenchmarkTask, path: Path | str) -> None:
    task_path = Path(path)
    data = task.model_dump(mode="json", exclude_none=True)
    try:
        task_path.parent.mkdir(parents=True, exist_ok=True)
        with task_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
    except OSError as error:
        raise TaskLoadError(f"Failed to write task file {task_path}: {error}") from error
