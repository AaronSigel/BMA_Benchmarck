"""Maps ValidationIssue codes to structured RepairAction suggestions.

The mapper extracts pre-filled arguments from the task spec so the LLM only
needs to confirm or fill in any remaining gaps — it does not decide from scratch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from benchmark.blender.models import SceneSnapshot
from benchmark.tasks.models import BenchmarkTask
from benchmark.validation.models import ValidationIssue, ValidationSeverity

_INDEX_RE = re.compile(r"\[(\d+)\]")

# Lower number = higher priority (fix first)
_ISSUE_PRIORITY: dict[str, int] = {
    "object_missing": 0,
    "object_missing_for_transform": 1,
    "material_missing": 2,
    "object_material_missing": 2,
    "light_missing": 3,
    "camera_missing": 4,
    "dimensions_mismatch": 5,
    "transform_mismatch": 5,
    "location_mismatch": 5,
    "rotation_mismatch": 5,
    "scale_mismatch": 5,
    "material_color_mismatch": 6,
    "material_roughness_mismatch": 6,
    "material_metallic_mismatch": 6,
    "light_rotation_mismatch": 7,
    "light_energy_mismatch": 7,
    "light_type_mismatch": 7,
    "camera_rotation_mismatch": 8,
    "camera_position_mismatch": 8,
    "active_camera_mismatch": 9,
    "export_missing": 10,
    "export_import_missing": 11,
}

# Issue codes that block export until they are resolved
EXPORT_BLOCKING_CODES = frozenset({
    "object_missing",
    "object_missing_for_transform",
    "material_missing",
    "object_material_missing",
    "light_missing",
    "camera_missing",
})


@dataclass
class RepairAction:
    tool_name: str
    arguments_template: dict[str, Any]
    issue_code: str
    description: str
    requires_prior_step: RepairAction | None = None


def select_top_issue(issues: list[ValidationIssue]) -> ValidationIssue | None:
    """Return the highest-priority actionable issue from the list."""
    errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
    candidates = errors if errors else issues
    actionable = [i for i in candidates if i.code in _ISSUE_PRIORITY]
    if not actionable:
        actionable = candidates
    if not actionable:
        return None
    return min(actionable, key=lambda i: _ISSUE_PRIORITY.get(i.code, 99))


def has_export_blocking_issues(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    """Return ERROR-severity issues that must be resolved before export."""
    return [
        i for i in issues
        if i.code in EXPORT_BLOCKING_CODES and i.severity == ValidationSeverity.ERROR
    ]


def map_issue_to_repair(
    issue: ValidationIssue,
    task: BenchmarkTask,
    snapshot: SceneSnapshot | None = None,
) -> RepairAction | None:
    """Map a ValidationIssue to a suggested RepairAction with pre-filled arguments."""
    code = issue.code
    if code == "object_missing":
        return _repair_object_missing(issue, task)
    if code == "object_missing_for_transform":
        return _repair_object_missing_for_transform(issue, task)
    if code in {"transform_mismatch", "dimensions_mismatch", "location_mismatch", "rotation_mismatch", "scale_mismatch"}:
        return _repair_transform(issue, task)
    if code in {"material_missing", "object_material_missing"}:
        return _repair_material_missing(issue, task)
    if code in {"material_color_mismatch", "material_roughness_mismatch", "material_metallic_mismatch"}:
        return _repair_material_mismatch(issue, task)
    if code == "light_missing":
        return _repair_light_missing(issue, task)
    if code in {"light_rotation_mismatch", "light_energy_mismatch", "light_type_mismatch"}:
        return _repair_light_mismatch(issue, task)
    if code == "camera_missing":
        return _repair_camera_missing(issue, task)
    if code in {"camera_rotation_mismatch", "camera_position_mismatch"}:
        return _repair_camera_mismatch(issue, task)
    if code == "active_camera_mismatch":
        return _repair_active_camera(issue, task)
    if code in {"export_missing", "export_import_missing"}:
        return _repair_export(issue, task)
    return None


def _extract_index(path: str | None) -> int | None:
    if not path:
        return None
    m = _INDEX_RE.search(path)
    return int(m.group(1)) if m else None


def _vec3_list(v: Any) -> list[float] | None:
    if v is None:
        return None
    return [v.x, v.y, v.z]


def _repair_object_missing(issue: ValidationIssue, task: BenchmarkTask) -> RepairAction:
    idx = _extract_index(issue.expected_path)
    objects = task.expected_scene.objects
    obj = objects[idx] if idx is not None and idx < len(objects) else None

    _PRIM_MAP = {
        "cube": "MESH_CUBE", "sphere": "MESH_SPHERE",
        "cylinder": "MESH_CYLINDER", "cone": "MESH_CONE", "plane": "MESH_PLANE",
    }
    args: dict[str, Any] = {}
    if obj:
        if obj.name:
            args["name"] = obj.name
        args["type"] = _PRIM_MAP.get(obj.primitive or "", "MESH_CUBE")
        loc = _vec3_list(obj.location)
        if loc:
            args["location"] = loc
        dims = _vec3_list(obj.dimensions)
        if dims:
            args["dimensions"] = dims
        elif obj.scale:
            args["scale"] = _vec3_list(obj.scale)

    return RepairAction(
        tool_name="bma_create_object",
        arguments_template=args,
        issue_code=issue.code,
        description=f"Create missing object '{args.get('name', 'object')}'",
    )


def _repair_object_missing_for_transform(issue: ValidationIssue, task: BenchmarkTask) -> RepairAction:
    create = _repair_object_missing(issue, task)
    idx = _extract_index(issue.expected_path)
    objects = task.expected_scene.objects
    obj = objects[idx] if idx is not None and idx < len(objects) else None
    if obj is None:
        return create

    t_args: dict[str, Any] = {}
    if obj.name:
        t_args["object_name"] = obj.name
    loc = _vec3_list(obj.location)
    if loc:
        t_args["location"] = loc
    dims = _vec3_list(obj.dimensions)
    if dims:
        t_args["dimensions"] = dims
    elif obj.scale:
        t_args["scale"] = _vec3_list(obj.scale)
    if obj.rotation:
        t_args["rotation"] = _vec3_list(obj.rotation)

    set_transform = RepairAction(
        tool_name="bma_set_transform",
        arguments_template=t_args,
        issue_code=issue.code,
        description=f"Set transform on '{obj.name}' after creating it",
    )
    return RepairAction(
        tool_name="bma_create_object",
        arguments_template=create.arguments_template,
        issue_code=issue.code,
        description=create.description,
        requires_prior_step=set_transform,
    )


def _repair_transform(issue: ValidationIssue, task: BenchmarkTask) -> RepairAction:
    idx = _extract_index(issue.expected_path)
    objects = task.expected_scene.objects
    obj = objects[idx] if idx is not None and idx < len(objects) else None

    args: dict[str, Any] = {}
    if obj:
        if obj.name:
            args["object_name"] = obj.name
        loc = _vec3_list(obj.location)
        if loc:
            args["location"] = loc
        dims = _vec3_list(obj.dimensions)
        if dims:
            args["dimensions"] = dims
        elif obj.scale:
            args["scale"] = _vec3_list(obj.scale)
        if obj.rotation:
            args["rotation"] = _vec3_list(obj.rotation)
    elif isinstance(issue.actual_value, str):
        args["object_name"] = issue.actual_value

    return RepairAction(
        tool_name="bma_set_transform",
        arguments_template=args,
        issue_code=issue.code,
        description=f"Fix transform mismatch on '{args.get('object_name', 'object')}'",
    )


def _repair_material_missing(issue: ValidationIssue, task: BenchmarkTask) -> RepairAction:
    path = issue.expected_path or ""
    mat = None
    if "materials[" in path:
        idx = _extract_index(path)
        materials = task.expected_scene.materials
        mat = materials[idx] if idx is not None and idx < len(materials) else None

    obj = None
    if ".material" in path:
        obj_part = path.split(".material")[0]
        obj_idx = _extract_index(obj_part)
        objects = task.expected_scene.objects
        obj = objects[obj_idx] if obj_idx is not None and obj_idx < len(objects) else None

    args: dict[str, Any] = {}
    if obj and obj.name:
        args["object_name"] = obj.name
    if mat:
        args["material_name"] = mat.name
        if mat.base_color:
            args["base_color"] = [mat.base_color.r, mat.base_color.g, mat.base_color.b, mat.base_color.a]
        if mat.roughness is not None:
            args["roughness"] = mat.roughness
        if mat.metallic is not None:
            args["metallic"] = mat.metallic
    elif isinstance(issue.expected_value, str):
        args["material_name"] = issue.expected_value

    return RepairAction(
        tool_name="bma_assign_material",
        arguments_template=args,
        issue_code=issue.code,
        description=f"Assign material to '{args.get('object_name', 'object')}'",
    )


def _repair_material_mismatch(issue: ValidationIssue, task: BenchmarkTask) -> RepairAction:
    idx = _extract_index(issue.expected_path)
    materials = task.expected_scene.materials
    mat = materials[idx] if idx is not None and idx < len(materials) else None

    args: dict[str, Any] = {}
    if mat:
        args["material_name"] = mat.name
        if mat.base_color:
            args["base_color"] = [mat.base_color.r, mat.base_color.g, mat.base_color.b, mat.base_color.a]
        if mat.roughness is not None:
            args["roughness"] = mat.roughness
        if mat.metallic is not None:
            args["metallic"] = mat.metallic

    return RepairAction(
        tool_name="bma_assign_material",
        arguments_template=args,
        issue_code=issue.code,
        description=f"Fix material mismatch on '{mat.name if mat else 'material'}'",
    )


def _repair_light_missing(issue: ValidationIssue, task: BenchmarkTask) -> RepairAction:
    idx = _extract_index(issue.expected_path)
    lights = task.expected_scene.lights
    light = lights[idx] if idx is not None and idx < len(lights) else None

    args: dict[str, Any] = {"type": "AREA"}
    if light:
        args["type"] = light.type
        if light.name:
            args["name"] = light.name
        loc = _vec3_list(light.location)
        if loc:
            args["location"] = loc
        if light.target:
            args["target"] = _vec3_list(light.target)
        elif light.rotation:
            args["rotation"] = _vec3_list(light.rotation)
        if light.energy is not None:
            args["energy"] = light.energy

    return RepairAction(
        tool_name="bma_create_light",
        arguments_template=args,
        issue_code=issue.code,
        description=f"Create missing light '{args.get('name', 'light')}'",
    )


def _repair_light_mismatch(issue: ValidationIssue, task: BenchmarkTask) -> RepairAction:
    # Recreate the light with corrected parameters
    idx = _extract_index(issue.expected_path)
    lights = task.expected_scene.lights
    light = lights[idx] if idx is not None and idx < len(lights) else None

    args: dict[str, Any] = {"type": "AREA"}
    if light:
        args["type"] = light.type
        if light.name:
            args["name"] = light.name
        loc = _vec3_list(light.location)
        if loc:
            args["location"] = loc
        if light.target:
            args["target"] = _vec3_list(light.target)
        elif light.rotation:
            args["rotation"] = _vec3_list(light.rotation)
        if light.energy is not None:
            args["energy"] = light.energy

    return RepairAction(
        tool_name="bma_create_light",
        arguments_template=args,
        issue_code=issue.code,
        description=f"Fix light '{args.get('name', 'light')}' mismatch",
    )


def _repair_camera_missing(issue: ValidationIssue, task: BenchmarkTask) -> RepairAction:
    idx = _extract_index(issue.expected_path)
    cameras = task.expected_scene.cameras
    cam = cameras[idx] if idx is not None and idx < len(cameras) else None

    args: dict[str, Any] = {}
    tool = "bma_create_camera"
    if cam:
        if cam.name:
            args["name"] = cam.name
        loc = _vec3_list(cam.location)
        if loc:
            args["location"] = loc
        if cam.target:
            args["target"] = _vec3_list(cam.target)
            tool = "bma_create_camera_look_at"
        elif cam.rotation:
            args["rotation"] = _vec3_list(cam.rotation)
        if cam.focal_length is not None:
            args["focal_length"] = cam.focal_length

    return RepairAction(
        tool_name=tool,
        arguments_template=args,
        issue_code=issue.code,
        description=f"Create missing camera '{args.get('name', 'camera')}'",
    )


def _repair_camera_mismatch(issue: ValidationIssue, task: BenchmarkTask) -> RepairAction:
    return _repair_camera_missing(issue, task)


def _repair_active_camera(issue: ValidationIssue, task: BenchmarkTask) -> RepairAction:
    cameras = task.expected_scene.cameras
    active = next((c for c in cameras if c.require_active), cameras[0] if cameras else None)
    args: dict[str, Any] = {}
    tool = "bma_create_camera_look_at"
    if active:
        if active.name:
            args["name"] = active.name
        loc = _vec3_list(active.location)
        if loc:
            args["location"] = loc
        if active.target:
            args["target"] = _vec3_list(active.target)
        elif active.rotation:
            args["rotation"] = _vec3_list(active.rotation)
            tool = "bma_create_camera"

    return RepairAction(
        tool_name=tool,
        arguments_template=args,
        issue_code=issue.code,
        description=f"Set active camera to '{args.get('name', 'camera')}'",
    )


def _repair_export(issue: ValidationIssue, task: BenchmarkTask) -> RepairAction:
    exports = task.expected_scene.exports
    export = exports[0] if exports else None
    args: dict[str, Any] = {"format": "glb"}
    if export:
        args["format"] = export.format
        if export.filename:
            args["filename"] = export.filename

    return RepairAction(
        tool_name="bma_export_scene",
        arguments_template=args,
        issue_code=issue.code,
        description="Export scene to required format",
    )


def build_task_checklist(task: BenchmarkTask) -> dict[str, Any]:
    """Build a structured checklist from the task spec for first-step injection."""
    scene = task.expected_scene
    return {
        "required_objects": [
            {"name": o.name, "primitive": o.primitive}
            for o in scene.objects
            if o.name
        ],
        "required_materials": [m.name for m in scene.materials],
        "required_lights": [
            {"name": l.name, "type": l.type}
            for l in scene.lights
            if l.name
        ],
        "required_cameras": [c.name for c in scene.cameras if c.name],
        "required_exports": [
            {"format": e.format, "filename": e.filename}
            for e in scene.exports
        ],
    }
