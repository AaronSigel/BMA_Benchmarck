from __future__ import annotations

from dataclasses import dataclass

from benchmark.tasks.models import ExpectedCamera, ExpectedLight, ExpectedMaterial, ExpectedObject


@dataclass(frozen=True)
class ValidatorFieldSpec:
    validator_name: str
    criterion_name: str
    checked_entity: str
    checked_field: str
    expected_section: str
    expected_attr: str | None
    actual_path_template: str
    scoring_rule: str
    issue_codes: tuple[str, ...]
    default_tolerance: float | None = None
    limitation: str | None = None


DEFAULT_OBJECT_TOLERANCE = ExpectedObject.model_fields["tolerance"].default
DEFAULT_MATERIAL_TOLERANCE = ExpectedMaterial.model_fields["tolerance"].default
DEFAULT_LIGHT_TOLERANCE = ExpectedLight.model_fields["tolerance"].default
DEFAULT_LIGHT_DIRECTION_TOLERANCE_DEG = ExpectedLight.model_fields["direction_tolerance_deg"].default
DEFAULT_CAMERA_TOLERANCE = ExpectedCamera.model_fields["tolerance"].default
DEFAULT_CAMERA_DIRECTION_TOLERANCE_DEG = ExpectedCamera.model_fields["direction_tolerance_deg"].default

PASS_THRESHOLD = 0.85
WARNING_THRESHOLD = 0.60

VALIDATOR_LIMITATIONS = {
    "object_validator": "Matches objects by name/type heuristics; does not prove semantic intent beyond expected snapshot fields.",
    "transform_validator": "Compares snapshot transforms and dimensions with numeric tolerance; cannot detect visually equivalent alternate modeling.",
    "material_validator": "Checks material parameters exposed in the snapshot and assignment slots; shader graphs are not fully reconstructed.",
    "light_validator": "Checks light type, transform/direction and energy; rendered illumination quality is approximated by snapshot values.",
    "camera_validator": "Checks camera transform, focal length, active flag and target direction; composition aesthetics are not fully assessed.",
    "export_validator": "Checks expected export file existence and non-empty size; does not import or inspect file contents.",
    "glb_import_back_validator": "Imports GLB when Blender is available and compares a snapshot subset; skipped for non-GLB exports.",
}

VALIDATOR_FIELD_SPECS: tuple[ValidatorFieldSpec, ...] = (
    ValidatorFieldSpec("object_validator", "object exists", "objects", "object", "objects", None, "snapshot.objects", "presence", ("object_missing",)),
    ValidatorFieldSpec("object_validator", "object type", "objects", "type", "objects", "type", "snapshot.objects[{index}].type", "exact_match", ("object_type_mismatch",)),
    ValidatorFieldSpec("object_validator", "primitive hint", "objects", "primitive", "objects", "primitive", "snapshot.objects[{index}].primitive_hint", "normalized_name_match", ("primitive_mismatch",)),
    ValidatorFieldSpec("transform_validator", "location", "objects", "location", "objects", "location", "snapshot.objects[{index}].location", "vector_tolerance", ("location_mismatch",), DEFAULT_OBJECT_TOLERANCE),
    ValidatorFieldSpec("transform_validator", "rotation", "objects", "rotation", "objects", "rotation", "snapshot.objects[{index}].rotation_euler", "vector_tolerance_degrees_to_radians", ("rotation_mismatch",), DEFAULT_OBJECT_TOLERANCE),
    ValidatorFieldSpec("transform_validator", "scale", "objects", "scale", "objects", "scale", "snapshot.objects[{index}].scale", "vector_tolerance", ("scale_mismatch",), DEFAULT_OBJECT_TOLERANCE),
    ValidatorFieldSpec("transform_validator", "dimensions", "objects", "dimensions", "objects", "dimensions", "snapshot.objects[{index}].dimensions", "vector_tolerance", ("dimensions_mismatch",), DEFAULT_OBJECT_TOLERANCE),
    ValidatorFieldSpec("material_validator", "material exists", "materials", "material", "materials", None, "snapshot.materials", "presence", ("material_missing",)),
    ValidatorFieldSpec("material_validator", "base_color", "materials", "base_color", "materials", "base_color", "snapshot.materials[{index}].base_color", "color_channel_tolerance", ("material_color_mismatch",), DEFAULT_MATERIAL_TOLERANCE),
    ValidatorFieldSpec("material_validator", "roughness", "materials", "roughness", "materials", "roughness", "snapshot.materials[{index}].roughness", "numeric_tolerance", ("material_roughness_mismatch",), DEFAULT_MATERIAL_TOLERANCE),
    ValidatorFieldSpec("material_validator", "metallic", "materials", "metallic", "materials", "metallic", "snapshot.materials[{index}].metallic", "numeric_tolerance", ("material_metallic_mismatch",), DEFAULT_MATERIAL_TOLERANCE),
    ValidatorFieldSpec("material_validator", "assignment", "objects", "material", "objects", "material", "snapshot.objects[{index}].material_slots", "slot_name_similarity", ("object_material_missing",)),
    ValidatorFieldSpec("light_validator", "light exists", "lights", "light", "lights", None, "snapshot.lights", "presence", ("light_missing",)),
    ValidatorFieldSpec("light_validator", "type", "lights", "type", "lights", "type", "snapshot.lights[{index}].type", "exact_match", ("light_type_mismatch",)),
    ValidatorFieldSpec("light_validator", "location", "lights", "location", "lights", "location", "snapshot.lights[{index}].location", "vector_tolerance", ("light_location_mismatch",), DEFAULT_LIGHT_TOLERANCE),
    ValidatorFieldSpec("light_validator", "rotation", "lights", "rotation", "lights", "rotation", "snapshot.lights[{index}].rotation_euler", "vector_or_direction_tolerance", ("light_rotation_mismatch", "light_direction_mismatch"), DEFAULT_LIGHT_TOLERANCE),
    ValidatorFieldSpec("light_validator", "energy", "lights", "energy", "lights", "energy", "snapshot.lights[{index}].energy", "relative_energy_tolerance_min_1w", ("light_energy_mismatch",), DEFAULT_LIGHT_TOLERANCE),
    ValidatorFieldSpec("camera_validator", "camera exists", "cameras", "camera", "cameras", None, "snapshot.cameras", "presence", ("camera_missing",)),
    ValidatorFieldSpec("camera_validator", "location", "cameras", "location", "cameras", "location", "snapshot.cameras[{index}].location", "vector_tolerance", ("camera_location_mismatch",), DEFAULT_CAMERA_TOLERANCE),
    ValidatorFieldSpec("camera_validator", "rotation", "cameras", "rotation", "cameras", "rotation", "snapshot.cameras[{index}].rotation_euler", "vector_tolerance_degrees_to_radians", ("camera_rotation_mismatch",), DEFAULT_CAMERA_TOLERANCE),
    ValidatorFieldSpec("camera_validator", "lens", "cameras", "focal_length", "cameras", "focal_length", "snapshot.cameras[{index}].lens", "numeric_tolerance", ("camera_focal_length_mismatch",), DEFAULT_CAMERA_TOLERANCE),
    ValidatorFieldSpec("camera_validator", "active camera", "cameras", "require_active", "cameras", "require_active", "snapshot.cameras[{index}].is_active", "boolean_match", ("active_camera_mismatch",)),
    ValidatorFieldSpec("camera_validator", "target direction", "cameras", "target", "cameras", "target", "snapshot.cameras[{index}].rotation_euler", "angular_tolerance_degrees", ("camera_direction_mismatch",), DEFAULT_CAMERA_DIRECTION_TOLERANCE_DEG),
    ValidatorFieldSpec("export_validator", "file exists", "exports", "path", "exports", None, "artifacts_dir/<expected export path>", "file_presence", ("export_missing", "export_format_unsupported")),
    ValidatorFieldSpec("export_validator", "file size", "exports", "file_size_bytes", "exports", None, "artifacts_dir/<expected export path>.stat().st_size", "positive_file_size", ("export_empty_file",)),
    ValidatorFieldSpec("glb_import_back_validator", "import success", "exports", "glb", "exports", "format", "imported_snapshot", "blender_import_success", ("export_import_failed", "export_import_missing", "export_import_file_too_small")),
    ValidatorFieldSpec("glb_import_back_validator", "mesh count", "exports", "object_count", "exports", "format", "imported_snapshot.objects", "count_match", ("export_import_mesh_count_mismatch",)),
    ValidatorFieldSpec("glb_import_back_validator", "expected names", "exports", "object_names", "exports", "format", "imported_snapshot.objects", "name_match", ("export_import_object_missing",)),
    ValidatorFieldSpec("glb_import_back_validator", "duplicate names", "exports", "object_names", "exports", "format", "imported_snapshot.objects", "duplicate_count_zero", ("export_import_duplicate_names",)),
    ValidatorFieldSpec("glb_import_back_validator", "material presence", "exports", "materials", "exports", "format", "imported_snapshot.materials", "material_presence_and_params", ("export_import_material_lost_after_export", "export_import_material_parameters_mismatch")),
)
