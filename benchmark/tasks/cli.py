import argparse
from pathlib import Path

import yaml

from benchmark.tasks.loader import TaskLoadError, load_tasks_from_dir
from benchmark.tasks.models import BenchmarkTask
from benchmark.tasks.registry import TaskRegistry
from benchmark.tasks.validator import validate_task_set


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        return _list_tasks(args.tasks_dir)
    if args.command == "show":
        return _show_task(args.task_id, args.tasks_dir)
    if args.command == "validate":
        return _validate_tasks(args.tasks_dir)

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browse and validate benchmark tasks.")
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List benchmark tasks.")
    list_parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"))

    show_parser = subparsers.add_parser("show", help="Show one benchmark task.")
    show_parser.add_argument("task_id")
    show_parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"))

    validate_parser = subparsers.add_parser("validate", help="Validate benchmark tasks.")
    validate_parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"))

    return parser


def _list_tasks(tasks_dir: Path) -> int:
    try:
        registry = _load_registry(tasks_dir)
    except (TaskLoadError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1

    tasks = registry.list_all()
    if not tasks:
        print(f"No tasks found in {tasks_dir}")
        return 0

    print(f"{'id':<38} {'category':<10} {'difficulty':<10} title")
    print("-" * 80)
    for task in tasks:
        print(f"{task.id:<38} {task.category.value:<10} {task.difficulty.value:<10} {task.title}")

    return 0


def _show_task(task_id: str, tasks_dir: Path) -> int:
    try:
        registry = _load_registry(tasks_dir)
        task = registry.get(task_id)
    except (TaskLoadError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    except KeyError:
        print(f"ERROR: unknown task id '{task_id}'")
        return 1

    print(_format_task(task))
    return 0


def _validate_tasks(tasks_dir: Path) -> int:
    try:
        tasks = load_tasks_from_dir(tasks_dir)
    except TaskLoadError as error:
        print(f"ERROR: {error}")
        return 1

    warnings = validate_task_set(tasks)
    print(f"Found {len(tasks)} task(s) in {tasks_dir}")

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("No warnings")

    return 0


def _load_registry(tasks_dir: Path) -> TaskRegistry:
    return TaskRegistry(load_tasks_from_dir(tasks_dir))


def _format_task(task: BenchmarkTask) -> str:
    lines = [
        f"id: {task.id}",
        f"title: {task.title}",
        f"category: {task.category.value}",
        f"difficulty: {task.difficulty.value}",
        "prompt:",
        _indent(task.prompt.strip()),
        "allowed_tools:",
        _format_yaml(task.allowed_tools),
        "expected_scene:",
        _format_yaml(task.expected_scene.model_dump(mode="json", exclude_none=True)),
        "success_criteria:",
        _format_yaml([criterion.model_dump(mode="json") for criterion in task.success_criteria]),
    ]
    return "\n".join(lines)


def _format_yaml(value: object) -> str:
    dumped = yaml.safe_dump(value, sort_keys=False, allow_unicode=True).rstrip()
    return _indent(dumped)


def _indent(value: str) -> str:
    return "\n".join(f"  {line}" for line in value.splitlines())


if __name__ == "__main__":
    raise SystemExit(main())

