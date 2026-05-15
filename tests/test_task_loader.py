from pathlib import Path

import pytest

from benchmark.tasks.loader import TaskLoadError, dump_task, load_task, load_tasks_from_dir
from benchmark.tasks.models import (
    BenchmarkTask,
    ColorRGBA,
    DifficultyLevel,
    ExpectedMaterial,
    ExpectedObject,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
    Vector3,
)


VALID_TASK_YAML = """
id: geometry-loader-cube
title: Loader cube task
category: geometry
difficulty: easy
prompt: Create a cube at the origin.
tags:
  - geometry
allowed_tools:
  - mesh.create_cube
expected_scene:
  objects:
    - name: Cube
      type: mesh
      primitive: cube
      location:
        x: 0
        y: 0
        z: 0
  materials: []
  lights: []
  cameras: []
  exports: []
success_criteria:
  - metric: object_exists
    weight: 1.0
"""


def make_valid_task() -> BenchmarkTask:
    return BenchmarkTask(
        id="dump-roundtrip-cube",
        title="Dump roundtrip cube",
        category=TaskCategory.GEOMETRY,
        difficulty=DifficultyLevel.EASY,
        prompt="Create a blue cube at the origin.",
        tags=["geometry", "dump"],
        allowed_tools=["mesh.create_cube", "material.create"],
        expected_scene=ExpectedScene(
            objects=[
                ExpectedObject(
                    name="Cube",
                    type="mesh",
                    primitive="cube",
                    location=Vector3(x=0.0, y=0.0, z=0.0),
                    material="Blue",
                )
            ],
            materials=[
                ExpectedMaterial(
                    name="Blue",
                    base_color=ColorRGBA(r=0.0, g=0.0, b=1.0),
                )
            ],
        ),
        success_criteria=[SuccessCriterion(metric="object_exists", weight=1.0)],
    )


def write_task(path: Path, task_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(VALID_TASK_YAML.replace("geometry-loader-cube", task_id), encoding="utf-8")


def test_load_task_loads_valid_yaml(tmp_path: Path) -> None:
    task_path = tmp_path / "task.yaml"
    task_path.write_text(VALID_TASK_YAML, encoding="utf-8")

    task = load_task(task_path)

    assert task.id == "geometry-loader-cube"
    assert task.category is TaskCategory.GEOMETRY
    assert task.expected_scene.objects[0].primitive == "cube"


def test_load_task_fails_on_invalid_yaml(tmp_path: Path) -> None:
    task_path = tmp_path / "broken.yaml"
    task_path.write_text("id: [not valid", encoding="utf-8")

    with pytest.raises(TaskLoadError, match=str(task_path)):
        load_task(task_path)


def test_load_task_fails_on_invalid_task_data(tmp_path: Path) -> None:
    task_path = tmp_path / "invalid.yaml"
    task_path.write_text(VALID_TASK_YAML.replace("prompt: Create a cube at the origin.", "prompt: ''"), encoding="utf-8")

    with pytest.raises(TaskLoadError, match=str(task_path)):
        load_task(task_path)


def test_load_tasks_from_dir_loads_nested_yaml_files(tmp_path: Path) -> None:
    write_task(tmp_path / "geometry" / "cube.yaml", "geometry-cube")
    write_task(tmp_path / "materials" / "blue.yml", "materials-blue")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    tasks = load_tasks_from_dir(tmp_path)

    assert [task.id for task in tasks] == ["geometry-cube", "materials-blue"]


def test_load_tasks_from_dir_respects_non_recursive_mode(tmp_path: Path) -> None:
    write_task(tmp_path / "root.yaml", "root-task")
    write_task(tmp_path / "nested" / "nested.yaml", "nested-task")

    tasks = load_tasks_from_dir(tmp_path, recursive=False)

    assert [task.id for task in tasks] == ["root-task"]


def test_dump_task_saves_yaml_that_can_be_loaded_again(tmp_path: Path) -> None:
    task = make_valid_task()
    task_path = tmp_path / "generated" / "task.yaml"

    dump_task(task, task_path)
    loaded_task = load_task(task_path)

    assert loaded_task == task


def test_load_tasks_from_dir_error_contains_problem_file_path(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.yaml"
    invalid_path = tmp_path / "nested" / "invalid.yaml"
    write_task(valid_path, "valid-task")
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_path.write_text(VALID_TASK_YAML.replace("category: geometry", "category: animation"), encoding="utf-8")

    with pytest.raises(TaskLoadError, match=str(invalid_path)):
        load_tasks_from_dir(tmp_path)

