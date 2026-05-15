from pathlib import Path

from benchmark.tasks.loader import load_tasks_from_dir
from benchmark.tasks.models import BenchmarkTask, DifficultyLevel, TaskCategory


class TaskRegistry:
    def __init__(self, tasks: list[BenchmarkTask]) -> None:
        self._tasks = list(tasks)
        self._tasks_by_id: dict[str, BenchmarkTask] = {}
        self.validate_unique_ids()

    def get(self, task_id: str) -> BenchmarkTask:
        try:
            return self._tasks_by_id[task_id]
        except KeyError as error:
            raise KeyError(f"Benchmark task not found: {task_id}") from error

    def list_all(self) -> list[BenchmarkTask]:
        return list(self._tasks)

    def filter_by_category(self, category: TaskCategory | str) -> list[BenchmarkTask]:
        task_category = TaskCategory(category)
        return [task for task in self._tasks if task.category is task_category]

    def filter_by_difficulty(self, difficulty: DifficultyLevel | str) -> list[BenchmarkTask]:
        difficulty_level = DifficultyLevel(difficulty)
        return [task for task in self._tasks if task.difficulty is difficulty_level]

    def filter_by_tag(self, tag: str) -> list[BenchmarkTask]:
        return [task for task in self._tasks if tag in task.tags]

    @classmethod
    def from_directory(cls, directory: Path | str) -> "TaskRegistry":
        return cls(load_tasks_from_dir(directory))

    def validate_unique_ids(self) -> None:
        tasks_by_id: dict[str, BenchmarkTask] = {}
        duplicate_ids: list[str] = []

        for task in self._tasks:
            if task.id in tasks_by_id:
                duplicate_ids.append(task.id)
            else:
                tasks_by_id[task.id] = task

        if duplicate_ids:
            duplicates = ", ".join(sorted(set(duplicate_ids)))
            raise ValueError(f"Duplicate benchmark task ids: {duplicates}")

        self._tasks_by_id = tasks_by_id

