from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from benchmark.experiments.models import EnvironmentRequirement, ExperimentMatrix, ReadinessCheckResult
from benchmark.experiments.matrix import (
    ExperimentMatrixError,
    load_agent_pool,
    load_mcp_profile_pool,
    select_agents,
    select_mcp_profiles,
    select_tasks,
)
from benchmark.runner.models import ExecutionMode, ExperimentConfig
from benchmark.tasks.loader import load_task
from benchmark.tasks.registry import TaskRegistry

_SOCKET_TIMEOUT_SEC = 0.2


def check_matrix_readiness(matrix: ExperimentMatrix) -> ReadinessCheckResult:
    errors: list[str] = []
    warnings: list[str] = []
    requirements: list[EnvironmentRequirement] = []

    task_registry = _load_task_registry(_metadata_path(matrix.metadata, "tasks_root", "tasks"), errors)
    if task_registry is not None:
        _append_selection_check(
            "tasks_found",
            requirements,
            errors,
            lambda: select_tasks(matrix, task_registry),
        )

    agent_pool = _load_pool("agent_configs_found", requirements, errors, lambda: load_agent_pool(matrix.agents.config_root))
    selected_agents: list[dict[str, Any]] = []
    if agent_pool is not None:
        selected_agents = _append_selection_check(
            "agent_configs_found",
            requirements,
            errors,
            lambda: select_agents(matrix, agent_pool),
        )
        strict = bool(matrix.metadata.get("strict_readiness", False))
        _check_api_keys(selected_agents, warnings, errors, requirements, strict=strict)
        _check_remote_agents(selected_agents, warnings, errors, requirements, strict=strict)

    mcp_pool = _load_pool(
        "mcp_configs_found",
        requirements,
        errors,
        lambda: load_mcp_profile_pool(_metadata_path(matrix.metadata, "mcp_config_root", "configs/mcp")),
    )
    selected_mcp_profiles: list[dict[str, Any]] = []
    if mcp_pool is not None:
        selected_mcp_profiles = _append_selection_check(
            "mcp_configs_found",
            requirements,
            errors,
            lambda: select_mcp_profiles(matrix, mcp_pool),
        )

    if matrix.report_config_path is not None and not Path(matrix.report_config_path).exists():
        errors.append(f"report_config_path does not exist: {matrix.report_config_path}")
    elif matrix.report_config_path is not None:
        requirements.append(
            EnvironmentRequirement(
                name="report_config_found",
                required=True,
                description=str(matrix.report_config_path),
            )
        )

    _check_writable_dir(matrix.output_root, "output_root_writable", errors, requirements)

    modes = matrix.execution_modes
    if _requires_blender(modes):
        _check_blender_available(errors, requirements)
    if _requires_mcp(modes):
        _check_mcp_connectivity(selected_mcp_profiles, errors, requirements)

    return ReadinessCheckResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        requirements=requirements,
        metadata={"matrix_id": matrix.matrix_id},
    )


def check_experiment_readiness(config: ExperimentConfig) -> ReadinessCheckResult:
    errors: list[str] = []
    warnings: list[str] = []
    requirements: list[EnvironmentRequirement] = []

    for run in config.runs:
        if run.task_path is not None and not run.task_path.exists():
            errors.append(f"task_path does not exist for {run.run_id}: {run.task_path}")
        if run.snapshot_path is not None and not run.snapshot_path.exists():
            errors.append(f"snapshot_path does not exist for {run.run_id}: {run.snapshot_path}")
        if run.agent_config_path is not None:
            if not run.agent_config_path.exists():
                errors.append(f"agent_config_path does not exist for {run.run_id}: {run.agent_config_path}")
            else:
                _check_api_keys(
                    [_load_yaml_mapping(run.agent_config_path)],
                    warnings,
                    errors,
                    requirements,
                    strict=False,
                )
        if run.mcp_config_path is not None and not run.mcp_config_path.exists():
            errors.append(f"mcp_config_path does not exist for {run.run_id}: {run.mcp_config_path}")
        _check_writable_dir(run.output_dir, f"output_dir_writable:{run.run_id}", errors, requirements)

    modes = [run.execution_mode for run in config.runs]
    if _requires_blender(modes):
        _check_blender_available(errors, requirements)
    if _requires_mcp(modes):
        mcp_profiles = [
            _load_yaml_mapping(run.mcp_config_path)
            for run in config.runs
            if run.mcp_config_path is not None and run.mcp_config_path.exists()
        ]
        _check_mcp_connectivity(mcp_profiles, errors, requirements)

    return ReadinessCheckResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        requirements=requirements,
        metadata={"experiment_id": config.experiment_id, "runs": len(config.runs)},
    )


def write_readiness_result(result: ReadinessCheckResult, path: Path | str) -> None:
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def readiness_result_to_json(result: ReadinessCheckResult) -> str:
    return json.dumps(result.model_dump(mode="json"), indent=2)


def _load_task_registry(root: Path, errors: list[str]) -> TaskRegistry | None:
    try:
        tasks = []
        for path in sorted(
            item
            for item in root.glob("**/*")
            if item.is_file() and item.suffix.lower() in {".yaml", ".yml"}
        ):
            if _is_known_non_task_yaml(path):
                continue
            tasks.append(load_task(path))
        return TaskRegistry(tasks)
    except Exception as error:
        errors.append(f"tasks not found or invalid in {root}: {error}")
        return None


def _load_pool(
    name: str,
    requirements: list[EnvironmentRequirement],
    errors: list[str],
    loader,
):
    try:
        pool = loader()
    except ExperimentMatrixError as error:
        errors.append(str(error))
        return None
    requirements.append(EnvironmentRequirement(name=name, required=True))
    return pool


def _append_selection_check(
    name: str,
    requirements: list[EnvironmentRequirement],
    errors: list[str],
    selector,
):
    try:
        selected = selector()
    except ExperimentMatrixError as error:
        errors.append(str(error))
        return []
    requirements.append(
        EnvironmentRequirement(
            name=name,
            required=True,
            metadata={"count": len(selected)},
        )
    )
    return selected


def _check_api_keys(
    agents: list[dict[str, Any]],
    warnings: list[str],
    errors: list[str],
    requirements: list[EnvironmentRequirement],
    *,
    strict: bool,
) -> None:
    for agent in agents:
        llm = agent.get("llm")
        if not isinstance(llm, dict):
            continue
        api_key_env = llm.get("api_key_env")
        if not api_key_env:
            continue
        requirements.append(
            EnvironmentRequirement(
                name="api_key_env",
                required=False,
                env_var=api_key_env,
                metadata={"agent_id": agent.get("agent_id")},
            )
        )
        if not os.environ.get(api_key_env):
            message = f"API key env var is not set for {agent.get('agent_id')}: {api_key_env}"
            if strict:
                errors.append(message)
            else:
                warnings.append(message)


def _check_remote_agents(
    agents: list[dict[str, Any]],
    warnings: list[str],
    errors: list[str],
    requirements: list[EnvironmentRequirement],
    *,
    strict: bool,
) -> None:
    for agent in agents:
        if agent.get("strategy") != "remote_agent":
            continue
        remote_agent = agent.get("remote_agent", {})
        provider = remote_agent.get("provider") if isinstance(remote_agent, dict) else None
        requirements.append(
            EnvironmentRequirement(
                name="remote_agent_configured",
                required=True,
                metadata={"agent_id": agent.get("agent_id"), "provider": provider},
            )
        )
        api_key_env = remote_agent.get("api_key_env") if isinstance(remote_agent, dict) else None
        if api_key_env and not os.environ.get(api_key_env):
            message = f"Remote agent API key env var is not set for {agent.get('agent_id')}: {api_key_env}"
            if strict:
                errors.append(message)
            else:
                warnings.append(message)
        warnings.append(f"Remote agent readiness is opt-in and requires external runtime: {agent.get('agent_id')}")


def _check_writable_dir(
    path: Path,
    name: str,
    errors: list[str],
    requirements: list[EnvironmentRequirement],
) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".readiness_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        errors.append(f"{name} is not writable: {path}: {error}")
    else:
        requirements.append(EnvironmentRequirement(name=name, required=True, description=str(path)))


def _check_blender_available(
    errors: list[str],
    requirements: list[EnvironmentRequirement],
) -> None:
    from benchmark.blender.config import find_blender_executable

    blender_bin = find_blender_executable()
    requirements.append(EnvironmentRequirement(name="blender_available", required=True))
    if blender_bin is None:
        errors.append("Blender executable not found. Set BMA_BLENDER_BIN or add blender to PATH.")


def _check_mcp_connectivity(
    mcp_profiles: list[dict[str, Any]],
    errors: list[str],
    requirements: list[EnvironmentRequirement],
) -> None:
    from benchmark.mcp.connection_check import check_blender_socket
    from benchmark.mcp.errors import BlenderSocketUnavailableError

    seen: set[tuple[str, int]] = set()
    for profile in mcp_profiles:
        host = str(profile.get("blender_host", "localhost"))
        port = int(profile.get("blender_port", 9876))
        key = (host, port)
        if key in seen:
            continue
        seen.add(key)
        requirements.append(
            EnvironmentRequirement(
                name="blender_socket_available",
                required=True,
                metadata={"host": host, "port": port},
            )
        )
        try:
            check_blender_socket(host, port, timeout_sec=_SOCKET_TIMEOUT_SEC)
        except BlenderSocketUnavailableError as error:
            errors.append(str(error))


def _requires_blender(modes: list[ExecutionMode]) -> bool:
    return any(mode in _BLENDER_REQUIRED_MODES for mode in modes)


def _requires_mcp(modes: list[ExecutionMode]) -> bool:
    return any(mode in _MCP_REQUIRED_MODES for mode in modes)


_BLENDER_REQUIRED_MODES = {
    ExecutionMode.BLENDER_SMOKE,
    ExecutionMode.MCP_SMOKE,
    ExecutionMode.MCP_EXTERNAL,
    ExecutionMode.AGENT_MCP,
    ExecutionMode.REMOTE_AGENT,
}

_MCP_REQUIRED_MODES = {
    ExecutionMode.MCP_SMOKE,
    ExecutionMode.MCP_EXTERNAL,
    ExecutionMode.AGENT_MCP,
    ExecutionMode.REMOTE_AGENT,
}


def _metadata_path(metadata: dict[str, Any], key: str, default: str) -> Path:
    return Path(metadata.get(key, default))


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data if isinstance(data, dict) else {}


def _is_known_non_task_yaml(path: Path) -> bool:
    data = _load_yaml_mapping(path)
    return "matrix_id" in data or "agent_id" in data
