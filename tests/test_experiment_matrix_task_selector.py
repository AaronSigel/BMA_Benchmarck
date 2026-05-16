import pytest

from benchmark.experiments.matrix import (
    ExperimentMatrixError,
    select_tasks,
    select_tasks_by_category,
    select_tasks_by_difficulty,
    select_tasks_by_ids,
    select_tasks_by_tags,
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


def make_registry() -> TaskRegistry:
    return TaskRegistry(
        [
            make_task(
                "geometry_easy",
                category=TaskCategory.GEOMETRY,
                difficulty=DifficultyLevel.EASY,
                tags=["geometry", "smoke"],
            ),
            make_task(
                "geometry_medium",
                category=TaskCategory.GEOMETRY,
                difficulty=DifficultyLevel.MEDIUM,
                tags=["geometry"],
            ),
            make_task(
                "materials_easy",
                category=TaskCategory.MATERIALS,
                difficulty=DifficultyLevel.EASY,
                tags=["materials", "smoke"],
            ),
        ]
    )


def test_select_tasks_by_category_can_select_only_geometry_tasks() -> None:
    selected = select_tasks_by_category(make_registry(), "geometry")

    assert [task.id for task in selected] == ["geometry_easy", "geometry_medium"]
    assert {task.category for task in selected} == {TaskCategory.GEOMETRY}


def test_select_tasks_by_difficulty_can_select_easy_tasks() -> None:
    selected = select_tasks_by_difficulty(make_registry(), DifficultyLevel.EASY)

    assert [task.id for task in selected] == ["geometry_easy", "materials_easy"]
    assert {task.difficulty for task in selected} == {DifficultyLevel.EASY}


def test_select_tasks_by_ids_can_select_specific_task_ids() -> None:
    selected = select_tasks_by_ids(make_registry(), ["materials_easy", "geometry_easy"])

    assert [task.id for task in selected] == ["materials_easy", "geometry_easy"]


def test_select_tasks_by_tags_matches_any_requested_tag_in_registry_order() -> None:
    selected = select_tasks_by_tags(make_registry(), ["smoke"])

    assert [task.id for task in selected] == ["geometry_easy", "materials_easy"]


def test_select_tasks_intersects_selector_dimensions() -> None:
    matrix = ExperimentMatrix(
        matrix_id="geometry_easy_matrix",
        tasks={
            "categories": ["geometry"],
            "difficulties": ["easy"],
            "tags": ["smoke"],
        },
    )

    selected = select_tasks(matrix, make_registry())

    assert [task.id for task in selected] == ["geometry_easy"]


def test_select_tasks_without_selector_returns_all_tasks() -> None:
    selected = select_tasks(ExperimentMatrix(matrix_id="all_tasks"), make_registry())

    assert [task.id for task in selected] == [
        "geometry_easy",
        "geometry_medium",
        "materials_easy",
    ]


def test_empty_selection_is_configuration_error() -> None:
    matrix = ExperimentMatrix(
        matrix_id="empty_matrix",
        tasks={"categories": ["camera"]},
    )

    with pytest.raises(ExperimentMatrixError, match="tasks.category=camera"):
        select_tasks(matrix, make_registry())


def test_unknown_task_id_is_configuration_error() -> None:
    with pytest.raises(ExperimentMatrixError, match="missing_task"):
        select_tasks_by_ids(make_registry(), ["missing_task"])
