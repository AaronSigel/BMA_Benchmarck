import argparse
from pathlib import Path

from benchmark.tasks.loader import TaskLoadError, load_tasks_from_dir
from benchmark.tasks.models import BenchmarkTask, TaskCategory
from benchmark.tasks.registry import TaskRegistry
from benchmark.tasks.tool_catalog import TOOL_CATALOG


def validate_task(task: BenchmarkTask) -> list[str]:
    warnings: list[str] = []

    if not task.success_criteria:
        warnings.append(f"{task.id}: task must define at least one success criterion")

    if not task.allowed_tools:
        warnings.append(f"{task.id}: task must define at least one allowed tool")
    else:
        allowed_tools = set(TOOL_CATALOG[task.category.value])
        unknown_tools = sorted(set(task.allowed_tools) - allowed_tools)
        if unknown_tools:
            warnings.append(
                f"{task.id}: task contains tools not allowed for category "
                f"{task.category.value}: {', '.join(unknown_tools)}"
            )

    warnings.extend(_validate_expected_scene_matches_category(task))

    total_weight = sum(criterion.weight for criterion in task.success_criteria)
    if task.success_criteria and abs(total_weight - 1.0) > 0.001:
        warnings.append(f"{task.id}: success criteria weights sum to {total_weight:.3f}, expected close to 1.0")

    empty_tags = [tag for tag in task.tags if not tag.strip()]
    if empty_tags:
        warnings.append(f"{task.id}: task contains empty tags")

    expected_prefix = f"{task.category.value}_"
    if not task.id.startswith(expected_prefix):
        warnings.append(f"{task.id}: task id should start with '{expected_prefix}'")

    return warnings


def validate_task_set(tasks: list[BenchmarkTask]) -> list[str]:
    warnings: list[str] = []

    try:
        TaskRegistry(tasks)
    except ValueError as error:
        warnings.append(str(error))

    for task in tasks:
        warnings.extend(validate_task(task))

    return warnings


def _validate_expected_scene_matches_category(task: BenchmarkTask) -> list[str]:
    scene = task.expected_scene

    if task.category is TaskCategory.GEOMETRY and not scene.objects:
        return [f"{task.id}: geometry task must define expected_scene.objects"]

    if task.category is TaskCategory.MATERIALS:
        has_materials = bool(scene.materials)
        has_objects_with_materials = any(expected_object.material for expected_object in scene.objects)
        if not has_materials and not has_objects_with_materials:
            return [f"{task.id}: materials task must define materials or objects with material"]

    if task.category is TaskCategory.LIGHTING and not scene.lights:
        return [f"{task.id}: lighting task must define expected_scene.lights"]

    if task.category is TaskCategory.CAMERA and not scene.cameras:
        return [f"{task.id}: camera task must define expected_scene.cameras"]

    if task.category is TaskCategory.EXPORT and not scene.exports:
        return [f"{task.id}: export task must define expected_scene.exports"]

    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark task YAML files.")
    parser.add_argument("directory", type=Path, help="Directory with benchmark task YAML files.")
    args = parser.parse_args(argv)

    try:
        tasks = load_tasks_from_dir(args.directory)
    except TaskLoadError as error:
        print(f"ERROR: {error}")
        return 1

    warnings = validate_task_set(tasks)
    print(f"Found {len(tasks)} task(s) in {args.directory}")

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("No warnings")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
