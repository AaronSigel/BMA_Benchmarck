"""Tests for GLB import-back diagnostics and material rename tolerance."""
from __future__ import annotations

from pathlib import Path

from benchmark.blender.models import ColorRGBA, MaterialSnapshot, ObjectSnapshot, SceneSnapshot, Vector3
from benchmark.tasks.models import BenchmarkTask, ExpectedMaterial, ExpectedObject, ExpectedScene
from benchmark.validation.validators.export_validator import STANDARD_EXPORT_CANDIDATES
from benchmark.validation.validators.glb_import_back_validator import (
    GlbImportBackValidator,
    _match_material_by_parameters,
)


def _snapshot_with_material(name: str, color: ColorRGBA) -> SceneSnapshot:
    return SceneSnapshot(
        scene_name="Test",
        objects=[
            ObjectSnapshot(
                name="Cube",
                type="MESH",
                primitive_hint="cube",
                location=Vector3(x=0, y=0, z=0),
                rotation_euler=Vector3(x=0, y=0, z=0),
                scale=Vector3(x=1, y=1, z=1),
                dimensions=Vector3(x=1, y=1, z=1),
                material_slots=[name],
                parent=None,
                collection_names=["Collection"],
                vertex_count=8,
                polygon_count=6,
            )
        ],
        materials=[
            MaterialSnapshot(
                name=name,
                base_color=color,
                roughness=0.5,
                metallic=0.0,
                use_nodes=True,
            )
        ],
        lights=[],
        cameras=[],
        collections=["Collection"],
        render_settings={
            "engine": "BLENDER_EEVEE",
            "resolution_x": 1920,
            "resolution_y": 1080,
            "frame_start": 1,
            "frame_end": 1,
            "frame_current": 1,
        },
        frame_current=1,
        blender_version="4.2.0",
        created_at="2026-01-01T00:00:00Z",
    )


def test_glb_material_rename_not_fatal_if_parameters_match() -> None:
    from benchmark.tasks.models import ColorRGBA as TaskColorRGBA

    expected = ExpectedMaterial(
        name="RedMaterial",
        base_color=TaskColorRGBA(r=1, g=0, b=0, a=1),
        tolerance=0.05,
    )
    imported = _snapshot_with_material("RedMaterial.001", ColorRGBA(r=1, g=0, b=0, a=1))
    matched = _match_material_by_parameters(expected, imported.materials)
    assert matched is not None
    assert matched.name == "RedMaterial.001"


def test_glb_transform_mismatch_has_diagnostics() -> None:
    import yaml

    task = BenchmarkTask.model_validate(
        yaml.safe_load(Path("tasks/export/export_002_glb_file.yaml").read_text(encoding="utf-8"))
    )
    imported = _snapshot_with_material("Mat", ColorRGBA(r=1, g=1, b=1, a=1))
    if imported.objects:
        imported.objects[0].location = Vector3(x=5, y=0, z=0)
    validator = GlbImportBackValidator(importer=lambda _path: imported)
    result = validator.validate(task, imported, artifacts_dir=Path("."))
    codes = {issue.code for issue in result.issues}
    assert "export_import_location_mismatch" in codes or "export_import_missing" in codes
    assert result.details.get("transform_mismatches")


def test_glb_material_loss_is_fatal() -> None:
    from benchmark.tasks.models import ColorRGBA as TaskColorRGBA, ExpectedExport

    task = BenchmarkTask(
        id="export_002",
        title="GLB export",
        category="export",
        difficulty="easy",
        prompt="export glb",
        tags=["export"],
        allowed_tools=["bma_export_scene"],
        expected_scene=ExpectedScene(
            objects=[ExpectedObject(name="Cube", type="MESH", primitive="cube")],
            materials=[
                ExpectedMaterial(
                    name="RedMaterial",
                    base_color=TaskColorRGBA(r=1, g=0, b=0, a=1),
                    tolerance=0.05,
                )
            ],
            exports=[ExpectedExport(format="glb")],
        ),
        success_criteria=[],
    )
    imported = SceneSnapshot(
        scene_name="Test",
        objects=[
            ObjectSnapshot(
                name="Cube",
                type="MESH",
                primitive_hint="cube",
                location=Vector3(x=0, y=0, z=0),
                rotation_euler=Vector3(x=0, y=0, z=0),
                scale=Vector3(x=1, y=1, z=1),
                dimensions=Vector3(x=1, y=1, z=1),
                material_slots=[],
                parent=None,
                collection_names=["Collection"],
                vertex_count=8,
                polygon_count=6,
            )
        ],
        materials=[],
        lights=[],
        cameras=[],
        collections=["Collection"],
        render_settings={
            "engine": "BLENDER_EEVEE",
            "resolution_x": 1920,
            "resolution_y": 1080,
            "frame_start": 1,
            "frame_end": 1,
            "frame_current": 1,
        },
        frame_current=1,
        blender_version="4.2.0",
        created_at="2026-01-01T00:00:00Z",
    )
    validator = GlbImportBackValidator(importer=lambda _path: imported)
    result = validator.validate(task, imported, artifacts_dir=Path("."))
    assert result.status.value == "failed"
    assert any(issue.code == "export_import_material_lost_after_export" for issue in result.issues)


def test_export_validator_uses_tool_output_path() -> None:
    assert Path("exports/result.glb") in STANDARD_EXPORT_CANDIDATES["glb"]
    assert Path("result.blend") in STANDARD_EXPORT_CANDIDATES["blend"]
