"""Стабильные идентификаторы проверок для отчётов."""

from __future__ import annotations

from benchmark.validation.models import ValidationCheckRow

DISPLAY_CHECK_IDS: dict[tuple[str, str, str | None], str] = {
    ("object_validator", "object exists", "object"): "object.exists",
    ("object_validator", "object type", "type"): "object.type",
    ("object_validator", "primitive hint", "primitive"): "primitive.type",
    ("transform_validator", "location", "location"): "location",
    ("transform_validator", "rotation", "rotation"): "rotation",
    ("transform_validator", "scale", "scale"): "scale",
    ("transform_validator", "dimensions", "dimensions"): "dimensions",
    ("material_validator", "material exists", "material"): "material.exists",
    ("material_validator", "assignment", "material"): "material.assignment",
    ("material_validator", "base_color", "base_color"): "material.base_color",
    ("material_validator", "roughness", "roughness"): "material.roughness",
    ("material_validator", "metallic", "metallic"): "material.metallic",
    ("export_validator", "file exists", "path"): "export.file_exists",
    ("export_validator", "file size", "file_size_bytes"): "export.file_non_empty",
    ("glb_import_back_validator", "import success", None): "import_back.success",
    ("glb_import_back_validator", "mesh count", None): "import_back.object_count",
    ("glb_import_back_validator", "expected names", "object_names"): "import_back.required_objects",
    ("glb_import_back_validator", "duplicate names", None): "import_back.duplicate_names",
    ("glb_import_back_validator", "material presence", None): "import_back.materials_preserved",
}


def display_check_id(row: ValidationCheckRow | dict) -> str:
    if isinstance(row, ValidationCheckRow):
        key = (row.validator_name, row.check_name, row.field)
        if key in DISPLAY_CHECK_IDS:
            return DISPLAY_CHECK_IDS[key]
        field = row.field
        if field:
            return f"{row.check_name}.{field}"
        return row.check_name
    validator_name = str(row.get("validator_name") or "")
    check_name = str(row.get("check_name") or "")
    field = row.get("field")
    key = (validator_name, check_name, field)
    if key in DISPLAY_CHECK_IDS:
        return DISPLAY_CHECK_IDS[key]
    if field:
        return f"{check_name}.{field}"
    return check_name


def display_object_ref(row: ValidationCheckRow | dict) -> str:
    if isinstance(row, ValidationCheckRow):
        entity = row.entity_ref
        validator = row.validator_name
    else:
        entity = row.get("entity_ref")
        validator = row.get("validator_name")
    if entity:
        return str(entity)
    if validator == "export_validator":
        return "export"
    if validator == "glb_import_back_validator":
        return "import_back"
    return "scene"
