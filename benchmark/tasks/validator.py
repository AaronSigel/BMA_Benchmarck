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
    warnings.extend(_validate_tool_consistency(task))
    warnings.extend(_validate_criteria_coverage(task))

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


def _validate_tool_consistency(task: BenchmarkTask) -> list[str]:
    """Warn when allowed_tools are missing tools required by expected_scene."""
    warnings: list[str] = []
    scene = task.expected_scene
    tools = set(task.allowed_tools)

    objects_need_transform = any(
        obj.location is not None
        or obj.rotation is not None
        or obj.scale is not None
        or obj.dimensions is not None
        for obj in scene.objects
    )
    if objects_need_transform and "set_transform" not in tools:
        warnings.append(
            f"{task.id}: expected_scene has objects with location/rotation/scale/dimensions "
            "but allowed_tools does not include set_transform"
        )

    objects_have_material = any(obj.material for obj in scene.objects)
    if (scene.materials or objects_have_material) and "assign_material" not in tools:
        warnings.append(
            f"{task.id}: expected_scene has materials but allowed_tools does not include assign_material"
        )

    materials_have_params = any(
        m.roughness is not None or m.metallic is not None or m.base_color is not None
        for m in scene.materials
    )
    if materials_have_params and "set_material_properties" not in tools:
        warnings.append(
            f"{task.id}: expected_scene has materials with numeric parameters "
            "but allowed_tools does not include set_material_properties"
        )

    if scene.lights and "create_light" not in tools:
        warnings.append(
            f"{task.id}: expected_scene has lights but allowed_tools does not include create_light"
        )

    lights_have_transform = any(
        light.location is not None or light.rotation is not None for light in scene.lights
    )
    if lights_have_transform and "set_transform" not in tools:
        warnings.append(
            f"{task.id}: expected_scene has lights with location/rotation "
            "but allowed_tools does not include set_transform"
        )

    lights_have_energy = any(light.energy is not None for light in scene.lights)
    if lights_have_energy and "set_light_properties" not in tools:
        warnings.append(
            f"{task.id}: expected_scene has lights with energy "
            "but allowed_tools does not include set_light_properties"
        )

    if scene.cameras and "create_camera" not in tools and "set_camera" not in tools:
        warnings.append(
            f"{task.id}: expected_scene has cameras but allowed_tools does not include "
            "create_camera or set_camera"
        )

    if scene.exports and "export_scene" not in tools:
        warnings.append(
            f"{task.id}: expected_scene has exports but allowed_tools does not include export_scene"
        )

    return warnings


def _validate_criteria_coverage(task: BenchmarkTask) -> list[str]:
    """Warn when success_criteria omit metrics for the task's primary category group."""
    warnings: list[str] = []
    scene = task.expected_scene
    metrics = {c.metric for c in task.success_criteria}
    cat = task.category

    object_metrics = {"object_existence", "geometry_accuracy", "object_placement"}
    material_metrics = {"material_accuracy", "parameter_correctness"}
    light_metrics = {"light_existence", "lighting_correctness"}
    camera_metrics = {"camera_existence", "camera_correctness", "target_visibility"}
    export_metrics = {"export_validity"}

    if cat is TaskCategory.GEOMETRY and scene.objects and not metrics & object_metrics:
        warnings.append(
            f"{task.id}: geometry task expected_scene.objects is non-empty but success_criteria "
            "do not reference any object metric (object_existence, geometry_accuracy, object_placement)"
        )

    if cat is TaskCategory.MATERIALS and scene.materials and not metrics & material_metrics:
        warnings.append(
            f"{task.id}: materials task expected_scene.materials is non-empty but success_criteria "
            "do not reference any material metric (material_accuracy, parameter_correctness)"
        )

    if cat is TaskCategory.LIGHTING and scene.lights and not metrics & light_metrics:
        warnings.append(
            f"{task.id}: lighting task expected_scene.lights is non-empty but success_criteria "
            "do not reference any lighting metric (light_existence, lighting_correctness)"
        )

    if cat is TaskCategory.CAMERA and scene.cameras and not metrics & camera_metrics:
        warnings.append(
            f"{task.id}: camera task expected_scene.cameras is non-empty but success_criteria "
            "do not reference any camera metric (camera_existence, camera_correctness, target_visibility)"
        )

    if cat is TaskCategory.EXPORT and scene.exports and not metrics & export_metrics:
        warnings.append(
            f"{task.id}: export task expected_scene.exports is non-empty but success_criteria "
            "do not reference export_validity"
        )

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
