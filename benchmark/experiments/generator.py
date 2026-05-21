from __future__ import annotations

import re
from pathlib import Path

import yaml

from benchmark.experiments.models import ExperimentMatrix
from benchmark.experiments.matrix import (
    ExperimentMatrixError,
    load_agent_pool,
    load_mcp_profile_pool,
    select_agents,
    select_mcp_profiles,
    select_tasks,
)
from benchmark.runner.models import ExecutionMode, ExperimentConfig, RunConfig
from benchmark.tasks.loader import load_task
from benchmark.tasks.registry import TaskRegistry


def generate_experiment_config(matrix: ExperimentMatrix) -> ExperimentConfig:
    task_registry, task_paths = _load_task_registry_with_paths(_tasks_root(matrix))
    agent_pool = load_agent_pool(matrix.agents.config_root)
    mcp_profile_pool = load_mcp_profile_pool(_mcp_config_root(matrix))

    tasks = select_tasks(matrix, task_registry)
    agents = select_agents(matrix, agent_pool)
    mcp_profiles = select_mcp_profiles(matrix, mcp_profile_pool)
    models = _select_models(matrix)
    execution_modes = matrix.execution_modes or [ExecutionMode.AGENT_MCP]

    runs: list[RunConfig] = []
    include_model_in_run_id = len(models) > 1
    for task in tasks:
        for agent in agents:
            for mcp_profile in mcp_profiles:
                for model_id in models:
                    for repetition in range(1, matrix.repetitions + 1):
                        run_id = _run_id(
                            matrix_id=matrix.matrix_id,
                            task_id=task.id,
                            agent_id=agent["agent_id"],
                            mcp_profile=mcp_profile["profile"],
                            repetition=repetition,
                            model_id=model_id if include_model_in_run_id else None,
                        )
                        output_dir = matrix.output_root / run_id
                        for execution_mode in execution_modes:
                            mode_run_id = _mode_run_id(run_id, execution_mode, len(execution_modes))
                            mode_output_dir = (
                                output_dir
                                if mode_run_id == run_id
                                else matrix.output_root / mode_run_id
                            )
                            runs.append(
                                RunConfig(
                                    run_id=mode_run_id,
                                    task_id=task.id,
                                    execution_mode=execution_mode,
                                    task_path=task_paths[task.id],
                                    snapshot_path=_snapshot_path(matrix, task.id),
                                    artifacts_dir=_artifacts_dir(matrix, mode_output_dir),
                                    output_dir=mode_output_dir,
                                    mcp_config_path=mcp_profile["config_path"],
                                    mcp_profile=mcp_profile["profile"],
                                    agent_config_path=agent["config_path"],
                                    agent_output_dir=mode_output_dir / "agent",
                                    metadata={
                                        "matrix_id": matrix.matrix_id,
                                        "agent_id": agent["agent_id"],
                                        "agent_strategy": agent["strategy"],
                                        "mcp_profile": mcp_profile["profile"],
                                        "model_id": model_id,
                                        "repetition": repetition,
                                        "strategy_limits": matrix.metadata.get("strategy_limits", {}),
                                    },
                                )
                            )

    return ExperimentConfig(
        experiment_id=matrix.matrix_id,
        runs=runs,
        metadata={
            "matrix_id": matrix.matrix_id,
            "title": matrix.title,
            "description": matrix.description,
        },
    )


def _load_task_registry_with_paths(root: Path) -> tuple[TaskRegistry, dict[str, Path]]:
    if not root.exists():
        raise ExperimentMatrixError(f"Task directory does not exist: {root}")
    paths = sorted(
        path
        for path in root.glob("**/*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )
    tasks = []
    task_paths: dict[str, Path] = {}
    for path in paths:
        if _is_known_non_task_yaml(path):
            continue
        task = load_task(path)
        tasks.append(task)
        task_paths[task.id] = path
    return TaskRegistry(tasks), task_paths


def _select_models(matrix: ExperimentMatrix) -> list[str]:
    if matrix.models.ids:
        return list(matrix.models.ids)
    if matrix.models.providers:
        return list(matrix.models.providers)
    return ["default"]


def _run_id(
    *,
    matrix_id: str,
    task_id: str,
    agent_id: str,
    mcp_profile: str,
    repetition: int,
    model_id: str | None = None,
) -> str:
    parts = [matrix_id, task_id, agent_id, mcp_profile]
    if model_id is not None:
        parts.append(_slug(model_id))
    parts.append(f"r{repetition}")
    return "__".join(parts)


def _mode_run_id(run_id: str, execution_mode: ExecutionMode, mode_count: int) -> str:
    if mode_count == 1:
        return run_id
    return f"{run_id}__{execution_mode.value}"


def _tasks_root(matrix: ExperimentMatrix) -> Path:
    return Path(matrix.metadata.get("tasks_root", "tasks"))


def _mcp_config_root(matrix: ExperimentMatrix) -> Path:
    return Path(matrix.metadata.get("mcp_config_root", "configs/mcp"))


def _snapshot_path(matrix: ExperimentMatrix, task_id: str | None = None) -> Path | None:
    if task_id:
        by_task = matrix.metadata.get("snapshot_path_by_task")
        if isinstance(by_task, dict) and task_id in by_task:
            value = by_task[task_id]
            return Path(value) if value else None
    value = matrix.metadata.get("snapshot_path")
    return Path(value) if value else None


def _artifacts_dir(matrix: ExperimentMatrix, output_dir: Path) -> Path:
    value = matrix.metadata.get("artifacts_dir")
    return Path(value) if value else output_dir


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "model"


def _is_known_non_task_yaml(path: Path) -> bool:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(data, dict):
        return False
    return "matrix_id" in data or "agent_id" in data
