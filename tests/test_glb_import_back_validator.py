from pathlib import Path

from benchmark.blender.models import (
    ColorRGBA,
    MaterialSnapshot,
    ObjectSnapshot,
    RenderSettingsSnapshot,
    SceneSnapshot,
    Vector3 as SnapshotVector3,
)
from benchmark.tasks.models import (
    BenchmarkTask,
    ColorRGBA as ExpectedColor,
    DifficultyLevel,
    ExpectedExport,
    ExpectedMaterial,
    ExpectedObject,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
    Vector3,
)
from benchmark.validation.models import ValidationStatus
from benchmark.validation.validators.glb_import_back_validator import GlbImportBackValidator


def vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector3:
    return Vector3(x=x, y=y, z=z)


def snapshot_vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> SnapshotVector3:
    return SnapshotVector3(x=x, y=y, z=z)


def object_snapshot(name: str = "Cube", material: str = "Red") -> ObjectSnapshot:
    return ObjectSnapshot(
        name=name,
        type="MESH",
        primitive_hint="cube",
        location=snapshot_vector(),
        rotation_euler=snapshot_vector(),
        scale=snapshot_vector(1.0, 1.0, 1.0),
        dimensions=snapshot_vector(2.0, 2.0, 2.0),
        material_slots=[material],
        parent=None,
        collection_names=["Collection"],
        vertex_count=8,
        polygon_count=6,
    )


def scene_snapshot(objects: list[ObjectSnapshot] | None = None) -> SceneSnapshot:
    objects = objects or []
    return SceneSnapshot(
        scene_name="Scene",
        objects=objects,
        materials=[
            MaterialSnapshot(
                name="Red",
                base_color=ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
                roughness=0.5,
                metallic=0.0,
                use_nodes=True,
            )
        ],
        lights=[],
        cameras=[],
        mesh_object_count=len(objects),
        light_count=0,
        camera_count=0,
        all_object_count=len(objects),
        collections=["Collection"],
        render_settings=RenderSettingsSnapshot(
            engine="CYCLES",
            resolution_x=1920,
            resolution_y=1080,
            frame_start=1,
            frame_end=1,
            frame_current=1,
        ),
        frame_current=1,
        blender_version="4.0.0",
        created_at="2026-05-15T12:00:00Z",
    )


def task_with_glb() -> BenchmarkTask:
    return BenchmarkTask(
        id="export_002_glb_file",
        title="Export GLB",
        category=TaskCategory.EXPORT,
        difficulty=DifficultyLevel.MEDIUM,
        prompt="Export GLB.",
        tags=["export"],
        allowed_tools=[],
        expected_scene=ExpectedScene(
            objects=[
                ExpectedObject(
                    name="Cube",
                    type="MESH",
                    primitive="cube",
                    location=vector(),
                    dimensions=vector(2.0, 2.0, 2.0),
                    material="Red",
                    tolerance=0.1,
                )
            ],
            materials=[
                ExpectedMaterial(
                    name="Red",
                    base_color=ExpectedColor(r=1.0, g=0.0, b=0.0, a=1.0),
                    tolerance=0.05,
                )
            ],
            exports=[ExpectedExport(format="glb", filename="exports/result.glb")],
        ),
        success_criteria=[SuccessCriterion(metric="export_validity", weight=1.0)],
    )


def write_glb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"glTF" + b"0" * 32)


def test_glb_import_back_validator_passes_imported_snapshot(tmp_path: Path) -> None:
    write_glb(tmp_path / "exports" / "result.glb")
    validator = GlbImportBackValidator(importer=lambda path: scene_snapshot([object_snapshot("Cube")]))

    result = validator.validate(task_with_glb(), scene_snapshot(), tmp_path)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
    assert result.metrics[0].name == "export_import_score"


def test_glb_import_back_validator_fails_missing_file(tmp_path: Path) -> None:
    result = GlbImportBackValidator(importer=lambda path: scene_snapshot()).validate(
        task_with_glb(),
        scene_snapshot(),
        tmp_path,
    )

    assert result.status is ValidationStatus.FAILED
    assert result.issues[0].code == "export_import_missing"


def test_glb_import_back_validator_fails_corrupt_or_unimportable_file(tmp_path: Path) -> None:
    write_glb(tmp_path / "exports" / "result.glb")

    def fail_import(path: Path) -> SceneSnapshot:
        raise RuntimeError("not a valid glb")

    result = GlbImportBackValidator(importer=fail_import).validate(task_with_glb(), scene_snapshot(), tmp_path)

    assert result.status is ValidationStatus.FAILED
    assert result.issues[0].code == "export_import_failed"


def test_glb_import_back_validator_detects_extra_imported_objects(tmp_path: Path) -> None:
    write_glb(tmp_path / "exports" / "result.glb")
    imported = scene_snapshot([object_snapshot("Cube"), object_snapshot("Extra")])

    result = GlbImportBackValidator(importer=lambda path: imported).validate(
        task_with_glb(),
        scene_snapshot(),
        tmp_path,
    )

    assert result.status is ValidationStatus.FAILED
    assert any(issue.code == "export_import_mesh_count_mismatch" for issue in result.issues)
