import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.tasks.loader import dump_task, load_tasks_from_dir
from benchmark.tasks.models import (
    BenchmarkTask,
    DifficultyLevel,
    ExpectedCamera,
    ExpectedLight,
    ExpectedObject,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
)
from benchmark.tasks.validator import validate_task, validate_task_set


TASKS_DIR = Path(__file__).resolve().parents[1] / "tasks"


@pytest.fixture(scope="module")
def all_task_files() -> list[BenchmarkTask]:
    return load_tasks_from_dir(TASKS_DIR)


def make_task(
    task_id: str = "geometry_001",
    category: TaskCategory = TaskCategory.GEOMETRY,
    expected_scene: ExpectedScene | None = None,
    allowed_tools: list[str] | None = None,
    success_criteria: list[SuccessCriterion] | None = None,
    tags: list[str] | None = None,
) -> BenchmarkTask:
    return BenchmarkTask(
        id=task_id,
        title=f"Task {task_id}",
        category=category,
        difficulty=DifficultyLevel.EASY,
        prompt=f"Execute task {task_id}.",
        tags=tags if tags is not None else ["smoke"],
        allowed_tools=allowed_tools if allowed_tools is not None else ["tool.run"],
        expected_scene=expected_scene
        if expected_scene is not None
        else ExpectedScene(objects=[ExpectedObject(type="mesh")]),
        success_criteria=success_criteria
        if success_criteria is not None
        else [SuccessCriterion(metric="complete", weight=1.0)],
    )


def test_validator_finds_task_without_allowed_tools() -> None:
    task = make_task(allowed_tools=[])

    warnings = validate_task(task)

    assert any("allowed tool" in warning for warning in warnings)


def test_validator_finds_task_without_success_criteria() -> None:
    task = make_task(success_criteria=[])

    warnings = validate_task(task)

    assert any("success criterion" in warning for warning in warnings)


def test_validator_warns_when_geometry_task_has_no_expected_objects() -> None:
    task = make_task(expected_scene=ExpectedScene())

    warnings = validate_task(task)

    assert any("expected_scene.objects" in warning for warning in warnings)


def test_validate_task_set_reports_duplicate_ids() -> None:
    tasks = [make_task(task_id="geometry_001"), make_task(task_id="geometry_001")]

    warnings = validate_task_set(tasks)

    assert any("Duplicate benchmark task ids" in warning for warning in warnings)


def test_category_content_checks_accept_matching_scene_content() -> None:
    tasks = [
        make_task(
            task_id="lighting_001",
            category=TaskCategory.LIGHTING,
            expected_scene=ExpectedScene(lights=[ExpectedLight(type="area")]),
        ),
        make_task(
            task_id="camera_001",
            category=TaskCategory.CAMERA,
            expected_scene=ExpectedScene(cameras=[ExpectedCamera(name="Camera")]),
        ),
        make_task(
            task_id="materials_001",
            category=TaskCategory.MATERIALS,
            expected_scene=ExpectedScene(objects=[ExpectedObject(type="mesh", material="Red")]),
        ),
    ]

    warnings = validate_task_set(tasks)

    assert warnings == []


def test_cli_runs_with_tasks_directory() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "benchmark.tasks.validator", "tasks/"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Found" in result.stdout


def test_all_yaml_files_from_tasks_directory_pass_validation() -> None:
    tasks = load_tasks_from_dir("tasks/")

    warnings = validate_task_set(tasks)

    assert warnings == []


def test_cli_prints_warnings_for_invalid_logical_task(tmp_path: Path) -> None:
    task = make_task(task_id="geometry_001", allowed_tools=[])
    dump_task(task, tmp_path / "geometry" / "task.yaml")

    result = subprocess.run(
        [sys.executable, "-m", "benchmark.tasks.validator", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "allowed tool" in result.stdout


def test_all_task_files_are_valid(all_task_files: list[BenchmarkTask]) -> None:
    assert all_task_files
    assert all(isinstance(task, BenchmarkTask) for task in all_task_files)


def test_task_ids_are_unique(all_task_files: list[BenchmarkTask]) -> None:
    task_ids = [task.id for task in all_task_files]

    assert len(task_ids) == len(set(task_ids))


def test_task_id_matches_category(all_task_files: list[BenchmarkTask]) -> None:
    for task in all_task_files:
        assert task.id.startswith(f"{task.category.value}_")


def test_each_task_has_success_criteria(all_task_files: list[BenchmarkTask]) -> None:
    for task in all_task_files:
        assert task.success_criteria


def test_each_task_has_allowed_tools(all_task_files: list[BenchmarkTask]) -> None:
    for task in all_task_files:
        assert task.allowed_tools


def test_expected_scene_matches_category(all_task_files: list[BenchmarkTask]) -> None:
    for task in all_task_files:
        scene = task.expected_scene
        match task.category:
            case TaskCategory.GEOMETRY:
                assert scene.objects
            case TaskCategory.MATERIALS:
                assert scene.materials
            case TaskCategory.LIGHTING:
                assert scene.lights
            case TaskCategory.CAMERA:
                assert scene.cameras
            case TaskCategory.EXPORT:
                assert scene.exports

