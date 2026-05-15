from pathlib import Path

import pytest

from benchmark.tasks.loader import dump_task
from benchmark.tasks.models import BenchmarkTask, DifficultyLevel, ExpectedScene, SuccessCriterion, TaskCategory
from benchmark.tasks.registry import TaskRegistry


def make_task(
    task_id: str,
    category: TaskCategory = TaskCategory.GEOMETRY,
    difficulty: DifficultyLevel = DifficultyLevel.EASY,
    tags: list[str] | None = None,
) -> BenchmarkTask:
    return BenchmarkTask(
        id=task_id,
        title=f"Task {task_id}",
        category=category,
        difficulty=difficulty,
        prompt=f"Run benchmark task {task_id}.",
        tags=tags or [],
        allowed_tools=[],
        expected_scene=ExpectedScene(),
        success_criteria=[SuccessCriterion(metric="complete", weight=1.0)],
    )


def test_get_returns_task_by_id() -> None:
    task = make_task("geometry-basic")
    registry = TaskRegistry([task])

    assert registry.get("geometry-basic") == task


def test_get_unknown_task_raises_key_error() -> None:
    registry = TaskRegistry([])

    with pytest.raises(KeyError, match="missing"):
        registry.get("missing")


def test_list_all_returns_tasks_in_stable_order() -> None:
    tasks = [make_task("first"), make_task("second")]
    registry = TaskRegistry(tasks)

    listed_tasks = registry.list_all()

    assert [task.id for task in listed_tasks] == ["first", "second"]
    assert listed_tasks is not tasks


def test_filter_by_category_accepts_enum_and_string() -> None:
    registry = TaskRegistry(
        [
            make_task("geometry-task", category=TaskCategory.GEOMETRY),
            make_task("materials-task", category=TaskCategory.MATERIALS),
        ]
    )

    assert [task.id for task in registry.filter_by_category(TaskCategory.GEOMETRY)] == ["geometry-task"]
    assert [task.id for task in registry.filter_by_category("materials")] == ["materials-task"]


def test_filter_by_difficulty_accepts_enum_and_string() -> None:
    registry = TaskRegistry(
        [
            make_task("easy-task", difficulty=DifficultyLevel.EASY),
            make_task("hard-task", difficulty=DifficultyLevel.HARD),
        ]
    )

    assert [task.id for task in registry.filter_by_difficulty(DifficultyLevel.EASY)] == ["easy-task"]
    assert [task.id for task in registry.filter_by_difficulty("hard")] == ["hard-task"]


def test_filter_by_tag_returns_matching_tasks_without_mutating_registry() -> None:
    registry = TaskRegistry(
        [
            make_task("cube-task", tags=["geometry", "cube"]),
            make_task("sphere-task", tags=["geometry", "sphere"]),
            make_task("material-task", tags=["material"]),
        ]
    )

    matches = registry.filter_by_tag("geometry")
    matches.clear()

    assert [task.id for task in registry.filter_by_tag("geometry")] == ["cube-task", "sphere-task"]
    assert [task.id for task in registry.list_all()] == ["cube-task", "sphere-task", "material-task"]


def test_duplicate_ids_raise_error() -> None:
    with pytest.raises(ValueError, match="duplicate-task"):
        TaskRegistry([make_task("duplicate-task"), make_task("duplicate-task")])


def test_from_directory_creates_registry_from_yaml_files(tmp_path: Path) -> None:
    first = make_task("first-task", tags=["first"])
    second = make_task("second-task", category=TaskCategory.CAMERA, difficulty=DifficultyLevel.MEDIUM)
    dump_task(first, tmp_path / "geometry" / "first.yaml")
    dump_task(second, tmp_path / "camera" / "second.yml")

    registry = TaskRegistry.from_directory(tmp_path)

    assert [task.id for task in registry.list_all()] == ["second-task", "first-task"]
    assert registry.get("first-task") == first
    assert registry.filter_by_category("camera") == [second]

