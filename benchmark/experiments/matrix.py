from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from benchmark.experiments.models import ExperimentMatrix
from benchmark.tasks.models import BenchmarkTask, DifficultyLevel, TaskCategory
from benchmark.tasks.registry import TaskRegistry


SUPPORTED_MCP_PROFILES: tuple[str, ...] = (
    "minimal",
    "no_python",
    "inspection_enabled",
    "python_enabled",
    "full",
)

DEFAULT_MCP_PROFILES: tuple[str, ...] = (
    "minimal",
    "no_python",
    "inspection_enabled",
)


class ExperimentMatrixError(ValueError):
    """Raised when an experiment matrix cannot be loaded or resolved."""


def load_matrix(path: Path | str) -> ExperimentMatrix:
    matrix_path = Path(path)
    try:
        data = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ExperimentMatrixError(f"Failed to read matrix config {matrix_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ExperimentMatrixError(f"Failed to parse matrix config {matrix_path}: {error}") from error
    if not isinstance(data, dict):
        raise ExperimentMatrixError(f"Matrix config {matrix_path} must contain a YAML mapping")
    return ExperimentMatrix.model_validate(data)


def dump_matrix(matrix: ExperimentMatrix, path: Path | str) -> None:
    matrix_path = Path(path)
    payload: dict[str, Any] = matrix.model_dump(mode="json", exclude_none=True)
    try:
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except OSError as error:
        raise ExperimentMatrixError(f"Failed to write matrix config {matrix_path}: {error}") from error


def ensure_non_empty_selection(items: list[Any], label: str) -> list[Any]:
    if not items:
        raise ExperimentMatrixError(f"Matrix selection for {label} is empty")
    return items


def select_tasks_by_ids(task_registry: TaskRegistry, ids: list[str]) -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []
    for task_id in ids:
        try:
            tasks.append(task_registry.get(task_id))
        except KeyError as error:
            raise ExperimentMatrixError(f"Unknown task_id in matrix: {task_id}") from error
    return ensure_non_empty_selection(tasks, "tasks")


def select_tasks_by_category(
    task_registry: TaskRegistry,
    category: TaskCategory | str,
) -> list[BenchmarkTask]:
    try:
        tasks = task_registry.filter_by_category(category)
    except ValueError as error:
        raise ExperimentMatrixError(f"Unknown task category in matrix: {category}") from error
    return ensure_non_empty_selection(tasks, f"tasks.category={_enum_value(category)}")


def select_tasks_by_difficulty(
    task_registry: TaskRegistry,
    difficulty: DifficultyLevel | str,
) -> list[BenchmarkTask]:
    try:
        tasks = task_registry.filter_by_difficulty(difficulty)
    except ValueError as error:
        raise ExperimentMatrixError(f"Unknown task difficulty in matrix: {difficulty}") from error
    return ensure_non_empty_selection(tasks, f"tasks.difficulty={_enum_value(difficulty)}")


def select_tasks_by_tags(task_registry: TaskRegistry, tags: list[str]) -> list[BenchmarkTask]:
    selected: dict[str, BenchmarkTask] = {}
    for tag in tags:
        for task in task_registry.filter_by_tag(tag):
            selected[task.id] = task
    return ensure_non_empty_selection(
        _tasks_in_registry_order(task_registry, set(selected)),
        f"tasks.tags={','.join(tags)}",
    )


def select_tasks(matrix: ExperimentMatrix, task_registry: TaskRegistry) -> list[BenchmarkTask]:
    selector = matrix.tasks
    selected_ids: set[str] | None = None

    if selector.ids:
        selected_ids = _intersect_ids(
            selected_ids,
            (task.id for task in select_tasks_by_ids(task_registry, selector.ids)),
        )

    if selector.categories:
        category_ids: set[str] = set()
        for category in selector.categories:
            category_ids.update(task.id for task in select_tasks_by_category(task_registry, category))
        selected_ids = _intersect_ids(selected_ids, category_ids)

    if selector.difficulties:
        difficulty_ids: set[str] = set()
        for difficulty in selector.difficulties:
            difficulty_ids.update(task.id for task in select_tasks_by_difficulty(task_registry, difficulty))
        selected_ids = _intersect_ids(selected_ids, difficulty_ids)

    if selector.tags:
        selected_ids = _intersect_ids(
            selected_ids,
            (task.id for task in select_tasks_by_tags(task_registry, selector.tags)),
        )

    if selected_ids is None:
        selected_ids = {task.id for task in task_registry.list_all()}

    if selector.ids:
        tasks = [task_registry.get(task_id) for task_id in selector.ids if task_id in selected_ids]
    else:
        tasks = _tasks_in_registry_order(task_registry, selected_ids)
    return ensure_non_empty_selection(tasks, "tasks")


def load_agent_pool(path: Path | str) -> dict[str, dict[str, Any]]:
    config_dir = Path(path)
    if not config_dir.exists():
        raise ExperimentMatrixError(f"Agent config directory does not exist: {config_dir}")
    if not config_dir.is_dir():
        raise ExperimentMatrixError(f"Agent config path is not a directory: {config_dir}")

    agent_pool: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for config_path in sorted(
        item for item in config_dir.iterdir() if item.suffix in {".yaml", ".yml"}
    ):
        data = _load_yaml_mapping(config_path, "agent config")
        if _is_known_non_agent_yaml(data):
            continue
        agent_id = data.get("agent_id")
        strategy = data.get("strategy")
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ExperimentMatrixError(f"Agent config {config_path} must define non-empty agent_id")
        if not isinstance(strategy, str) or not strategy.strip():
            raise ExperimentMatrixError(f"Agent config {config_path} must define non-empty strategy")
        if agent_id in agent_pool:
            duplicate_ids.append(agent_id)
        data = dict(data)
        data["config_path"] = config_path
        agent_pool[agent_id] = data

    if duplicate_ids:
        duplicates = ", ".join(sorted(set(duplicate_ids)))
        raise ExperimentMatrixError(f"Duplicate agent ids in pool: {duplicates}")
    return agent_pool


def select_agents_by_ids(
    agent_pool: dict[str, dict[str, Any]],
    ids: list[str],
) -> list[dict[str, Any]]:
    agents: list[dict[str, Any]] = []
    for agent_id in ids:
        try:
            agents.append(agent_pool[agent_id])
        except KeyError as error:
            raise ExperimentMatrixError(f"Unknown agent_id in matrix: {agent_id}") from error
    return ensure_non_empty_selection(agents, "agents")


def select_agents_by_strategy(
    agent_pool: dict[str, dict[str, Any]],
    strategies: list[str],
) -> list[dict[str, Any]]:
    from benchmark.agent.strategies.registry import STRATEGY_REGISTRY

    unknown = sorted(set(strategies) - set(STRATEGY_REGISTRY.names()))
    if unknown:
        raise ExperimentMatrixError(
            f"Unknown agent strategy in matrix: {', '.join(unknown)}. "
            f"Available: {', '.join(STRATEGY_REGISTRY.names())}"
        )
    strategy_set = set(strategies)
    agents = [agent for agent in agent_pool.values() if agent.get("strategy") in strategy_set]
    return ensure_non_empty_selection(
        agents,
        f"agents.strategy={','.join(strategies)}",
    )


def select_agents(
    matrix: ExperimentMatrix,
    agent_pool: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selector = matrix.agents
    selected_ids: set[str] | None = None

    if selector.ids:
        selected_ids = _intersect_ids(
            selected_ids,
            (agent["agent_id"] for agent in select_agents_by_ids(agent_pool, selector.ids)),
        )

    if selector.strategies:
        selected_ids = _intersect_ids(
            selected_ids,
            (
                agent["agent_id"]
                for agent in select_agents_by_strategy(agent_pool, selector.strategies)
            ),
        )

    if selected_ids is None:
        selected_ids = set(agent_pool)

    if selector.ids:
        agents = [agent_pool[agent_id] for agent_id in selector.ids if agent_id in selected_ids]
    else:
        agents = [agent for agent_id, agent in agent_pool.items() if agent_id in selected_ids]
    if not selector.include_remote_agents:
        agents = [agent for agent in agents if agent.get("strategy") != "remote_agent"]
    return ensure_non_empty_selection(agents, "agents")


def load_mcp_profile_pool(path: Path | str) -> dict[str, dict[str, Any]]:
    config_dir = Path(path)
    if not config_dir.exists():
        raise ExperimentMatrixError(f"MCP config directory does not exist: {config_dir}")
    if not config_dir.is_dir():
        raise ExperimentMatrixError(f"MCP config path is not a directory: {config_dir}")

    profile_pool: dict[str, dict[str, Any]] = {}
    duplicate_profiles: list[str] = []
    for config_path in sorted(
        item for item in config_dir.iterdir() if item.suffix in {".yaml", ".yml"}
    ):
        data = _load_yaml_mapping(config_path, "MCP config")
        profile = data.get("profile", config_path.stem)
        if not isinstance(profile, str) or not profile.strip():
            raise ExperimentMatrixError(f"MCP config {config_path} must define non-empty profile")
        profile = profile.strip().lower()
        _validate_supported_mcp_profile(profile)
        if profile in profile_pool:
            duplicate_profiles.append(profile)
        data = dict(data)
        data["profile"] = profile
        data["config_path"] = config_path
        profile_pool[profile] = data

    if duplicate_profiles:
        duplicates = ", ".join(sorted(set(duplicate_profiles)))
        raise ExperimentMatrixError(f"Duplicate MCP profiles in pool: {duplicates}")
    return profile_pool


def select_mcp_profiles_by_names(
    profile_pool: dict[str, dict[str, Any]],
    profiles: list[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for profile in profiles:
        profile_name = profile.strip().lower()
        _validate_supported_mcp_profile(profile_name)
        try:
            selected.append(profile_pool[profile_name])
        except KeyError as error:
            raise ExperimentMatrixError(f"MCP profile config not found: {profile_name}") from error
    return ensure_non_empty_selection(selected, "mcp_profiles")


def select_mcp_profiles(
    matrix: ExperimentMatrix,
    profile_pool: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    profiles = matrix.mcp_profiles or list(DEFAULT_MCP_PROFILES)
    return select_mcp_profiles_by_names(profile_pool, profiles)


def _intersect_ids(current: set[str] | None, incoming: Any) -> set[str]:
    incoming_ids = set(incoming)
    if current is None:
        return incoming_ids
    return current & incoming_ids


def _tasks_in_registry_order(task_registry: TaskRegistry, task_ids: set[str]) -> list[BenchmarkTask]:
    return [task for task in task_registry.list_all() if task.id in task_ids]


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _validate_supported_mcp_profile(profile: str) -> None:
    if profile not in SUPPORTED_MCP_PROFILES:
        supported = ", ".join(SUPPORTED_MCP_PROFILES)
        raise ExperimentMatrixError(f"Unsupported MCP profile: {profile}. Supported: {supported}")


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except OSError as error:
        raise ExperimentMatrixError(f"Failed to read {label} {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ExperimentMatrixError(f"Failed to parse YAML {label} {path}: {error}") from error

    if not isinstance(data, dict):
        raise ExperimentMatrixError(f"{label.capitalize()} {path} must contain a YAML mapping")
    return data


def _is_known_non_agent_yaml(data: dict[str, Any]) -> bool:
    return "matrix_id" in data or "schema_version" in data
