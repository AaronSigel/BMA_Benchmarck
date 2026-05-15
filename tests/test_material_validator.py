import pytest

from benchmark.blender.models import (
    ColorRGBA as SnapshotColorRGBA,
    MaterialSnapshot,
    ObjectSnapshot,
    RenderSettingsSnapshot,
    SceneSnapshot,
    Vector3,
)
from benchmark.tasks.models import (
    BenchmarkTask,
    ColorRGBA,
    DifficultyLevel,
    ExpectedMaterial,
    ExpectedObject,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
)
from benchmark.validation.models import ValidationStatus
from benchmark.validation.validators.material_validator import MaterialValidator


def vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector3:
    return Vector3(x=x, y=y, z=z)


def material_snapshot(
    name: str,
    base_color: SnapshotColorRGBA | None = None,
    roughness: float | None = 0.5,
    metallic: float | None = 0.0,
) -> MaterialSnapshot:
    return MaterialSnapshot(
        name=name,
        base_color=base_color or SnapshotColorRGBA(r=1.0, g=0.0, b=0.0),
        roughness=roughness,
        metallic=metallic,
        use_nodes=True,
    )


def object_snapshot(name: str, material_slots: list[str] | None = None) -> ObjectSnapshot:
    return ObjectSnapshot(
        name=name,
        type="MESH",
        primitive_hint="cube",
        location=vector(),
        rotation_euler=vector(),
        scale=vector(1.0, 1.0, 1.0),
        dimensions=vector(2.0, 2.0, 2.0),
        material_slots=material_slots or [],
        parent=None,
        collection_names=["Collection"],
        vertex_count=None,
        polygon_count=None,
    )


def scene_snapshot(
    materials: list[MaterialSnapshot] | None = None,
    objects: list[ObjectSnapshot] | None = None,
) -> SceneSnapshot:
    return SceneSnapshot(
        scene_name="Scene",
        objects=objects or [],
        materials=materials or [],
        lights=[],
        cameras=[],
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


def task_with_scene(expected_scene: ExpectedScene) -> BenchmarkTask:
    return BenchmarkTask(
        id="materials_001_basic_colors",
        title="Create materials",
        category=TaskCategory.MATERIALS,
        difficulty=DifficultyLevel.EASY,
        prompt="Create the expected materials.",
        tags=["materials"],
        allowed_tools=[],
        expected_scene=expected_scene,
        success_criteria=[SuccessCriterion(metric="materials", weight=1.0)],
    )


def metric_score(result, name: str) -> float:
    return next(metric.score for metric in result.metrics if metric.name == name)


def test_material_validator_finds_material_by_name_with_matcher() -> None:
    task = task_with_scene(
        ExpectedScene(
            materials=[
                ExpectedMaterial(
                    name="red_material",
                    base_color=ColorRGBA(r=1.0, g=0.0, b=0.0),
                    roughness=0.5,
                    metallic=0.0,
                )
            ]
        )
    )
    snapshot = scene_snapshot(materials=[material_snapshot("Red Material.001")])

    result = MaterialValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
    assert result.issues == []
    assert metric_score(result, "material_existence_score") == 1.0
    assert metric_score(result, "material_parameter_score") == 1.0


def test_material_validator_compares_color_with_tolerance() -> None:
    task = task_with_scene(
        ExpectedScene(
            materials=[
                ExpectedMaterial(
                    name="Red",
                    base_color=ColorRGBA(r=1.0, g=0.0, b=0.0),
                    tolerance=0.1,
                )
            ]
        )
    )
    snapshot = scene_snapshot(
        materials=[material_snapshot("Red", base_color=SnapshotColorRGBA(r=0.85, g=0.0, b=0.0))]
    )

    result = MaterialValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert 0.0 < metric_score(result, "material_parameter_score") < 1.0
    assert result.issues[0].code == "material_color_mismatch"
    assert result.issues[0].expected_path == "expected_scene.materials[0].base_color"


def test_material_validator_compares_roughness_and_metallic_with_tolerance() -> None:
    task = task_with_scene(
        ExpectedScene(
            materials=[
                ExpectedMaterial(
                    name="Metal",
                    roughness=0.2,
                    metallic=1.0,
                    tolerance=0.1,
                )
            ]
        )
    )
    snapshot = scene_snapshot(materials=[material_snapshot("Metal", roughness=0.35, metallic=0.7)])

    result = MaterialValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert result.score < 1.0
    assert {issue.code for issue in result.issues} == {
        "material_roughness_mismatch",
        "material_metallic_mismatch",
    }


def test_material_validator_reports_missing_material() -> None:
    task = task_with_scene(ExpectedScene(materials=[ExpectedMaterial(name="Missing")]))

    result = MaterialValidator().validate(task, scene_snapshot(materials=[]))

    assert result.status is ValidationStatus.FAILED
    assert metric_score(result, "material_existence_score") == 0.0
    assert result.issues[0].code == "material_missing"
    assert result.issues[0].expected_path == "expected_scene.materials[0]"
    assert result.issues[0].message


def test_material_validator_checks_object_material_assignment() -> None:
    task = task_with_scene(
        ExpectedScene(
            objects=[ExpectedObject(name="Cube", type="MESH", material="Red Material")]
        )
    )
    snapshot = scene_snapshot(objects=[object_snapshot("Cube.001", material_slots=["Red_Material.001"])])

    result = MaterialValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
    assert metric_score(result, "object_material_assignment_score") == 1.0


def test_material_validator_reports_empty_material_slots() -> None:
    task = task_with_scene(
        ExpectedScene(objects=[ExpectedObject(name="Cube", type="MESH", material="Red")])
    )
    snapshot = scene_snapshot(objects=[object_snapshot("Cube", material_slots=[])])

    result = MaterialValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.FAILED
    assert metric_score(result, "object_material_assignment_score") == 0.0
    assert result.issues[0].code == "object_material_missing"
    assert result.issues[0].expected_path == "expected_scene.objects[0].material"
    assert result.issues[0].actual_path == "snapshot.objects[0].material_slots"
    assert result.issues[0].message


def test_material_validator_reports_missing_object_for_assignment() -> None:
    task = task_with_scene(
        ExpectedScene(objects=[ExpectedObject(name="Missing", type="MESH", material="Red")])
    )

    result = MaterialValidator().validate(task, scene_snapshot(objects=[]))

    assert result.status is ValidationStatus.FAILED
    assert result.issues[0].code == "object_missing_for_material"
    assert result.issues[0].expected_path == "expected_scene.objects[0]"


def test_material_validator_skips_when_no_material_expectations_exist() -> None:
    task = task_with_scene(ExpectedScene(objects=[ExpectedObject(name="Cube", type="MESH")]))
    snapshot = scene_snapshot(objects=[object_snapshot("Cube", material_slots=[])])

    result = MaterialValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.SKIPPED
    assert result.score == 0.0
    assert result.issues == []
    assert result.metrics == []


def test_unchecked_material_parameters_do_not_affect_score() -> None:
    task = task_with_scene(ExpectedScene(materials=[ExpectedMaterial(name="Red")]))
    snapshot = scene_snapshot(
        materials=[material_snapshot("Red", base_color=SnapshotColorRGBA(r=0.0, g=1.0, b=0.0))]
    )

    result = MaterialValidator().validate(task, snapshot)

    assert result.status is ValidationStatus.PASSED
    assert result.score == 1.0
