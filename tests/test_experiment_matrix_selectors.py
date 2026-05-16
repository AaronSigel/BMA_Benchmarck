from pathlib import Path

import pytest
import yaml

from benchmark.experiments.matrix import (
    ExperimentMatrixError,
    load_agent_pool,
    load_matrix,
    load_mcp_profile_pool,
    select_agents_by_strategy,
    select_mcp_profiles,
    select_tasks,
    select_tasks_by_category,
    select_tasks_by_difficulty,
    select_tasks_by_ids,
)
from benchmark.experiments.models import ExperimentMatrix
from benchmark.tasks.models import (
    BenchmarkTask,
    DifficultyLevel,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
)
from benchmark.tasks.registry import TaskRegistry


def make_task(
    task_id: str,
    category: TaskCategory = TaskCategory.GEOMETRY,
    difficulty: DifficultyLevel = DifficultyLevel.EASY,
) -> BenchmarkTask:
    return BenchmarkTask(
        id=task_id,
        title=f"Task {task_id}",
        category=category,
        difficulty=difficulty,
        prompt=f"Run {task_id}.",
        tags=[],
        allowed_tools=[],
        expected_scene=ExpectedScene(),
        success_criteria=[SuccessCriterion(metric="complete", weight=1.0)],
    )


def make_task_registry() -> TaskRegistry:
    return TaskRegistry(
        [
            make_task("geometry_easy", TaskCategory.GEOMETRY, DifficultyLevel.EASY),
            make_task("geometry_medium", TaskCategory.GEOMETRY, DifficultyLevel.MEDIUM),
            make_task("materials_easy", TaskCategory.MATERIALS, DifficultyLevel.EASY),
        ]
    )


def write_agent_config(directory: Path, filename: str, agent_id: str, strategy: str) -> None:
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


def write_mcp_config(directory: Path, profile: str) -> None:
    (directory / f"{profile}.yaml").write_text(
        yaml.safe_dump({"profile": profile, "command": "uvx", "args": ["blender-mcp"]}),
        encoding="utf-8",
    )


def test_load_matrix_yaml(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "matrix_id": "selector_matrix",
                "tasks": {"ids": ["geometry_easy"]},
                "agents": {"strategies": ["react"]},
                "mcp_profiles": ["minimal"],
            }
        ),
        encoding="utf-8",
    )

    matrix = load_matrix(matrix_path)

    assert matrix.matrix_id == "selector_matrix"
    assert matrix.tasks.ids == ["geometry_easy"]
    assert matrix.agents.strategies == ["react"]


def test_select_tasks_by_id_category_and_difficulty() -> None:
    registry = make_task_registry()

    assert [task.id for task in select_tasks_by_ids(registry, ["materials_easy"])] == [
        "materials_easy"
    ]
    assert [task.id for task in select_tasks_by_category(registry, "geometry")] == [
        "geometry_easy",
        "geometry_medium",
    ]
    assert [task.id for task in select_tasks_by_difficulty(registry, "easy")] == [
        "geometry_easy",
        "materials_easy",
    ]


def test_select_tasks_empty_selection_is_error() -> None:
    with pytest.raises(ExperimentMatrixError, match="tasks.category=camera"):
        select_tasks(
            ExperimentMatrix(matrix_id="empty", tasks={"categories": ["camera"]}),
            make_task_registry(),
        )


def test_select_agents_by_strategy(tmp_path: Path) -> None:
    write_agent_config(tmp_path, "direct.yaml", "direct", "direct_tool_calling")
    write_agent_config(tmp_path, "react.yaml", "react", "react")
    write_agent_config(tmp_path, "plan.yaml", "plan", "plan_and_execute")
    pool = load_agent_pool(tmp_path)

    selected = select_agents_by_strategy(pool, ["react", "plan_and_execute"])

    assert [agent["agent_id"] for agent in selected] == ["plan", "react"]


def test_select_mcp_profiles(tmp_path: Path) -> None:
    write_mcp_config(tmp_path, "minimal")
    write_mcp_config(tmp_path, "no_python")
    pool = load_mcp_profile_pool(tmp_path)

    selected = select_mcp_profiles(
        ExperimentMatrix(matrix_id="mcp", mcp_profiles=["minimal", "no_python"]),
        pool,
    )

    assert [profile["profile"] for profile in selected] == ["minimal", "no_python"]


def test_select_mcp_profiles_empty_selection_is_error(tmp_path: Path) -> None:
    write_mcp_config(tmp_path, "minimal")
    pool = load_mcp_profile_pool(tmp_path)

    with pytest.raises(ExperimentMatrixError, match="no_python"):
        select_mcp_profiles(
            ExperimentMatrix(matrix_id="missing_mcp", mcp_profiles=["no_python"]),
            pool,
        )
