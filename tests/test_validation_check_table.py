from benchmark.blender.models import ObjectSnapshot, RenderSettingsSnapshot, SceneSnapshot, Vector3 as SnapshotVector3
from benchmark.tasks.models import (
    BenchmarkTask,
    DifficultyLevel,
    ExpectedObject,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
    Vector3,
)
from benchmark.validation.models import SceneValidationResult, ValidationStatus
from benchmark.validation.scene_validator import SceneValidator


def _snapshot(objects: list[ObjectSnapshot]) -> SceneSnapshot:
    return SceneSnapshot(
        scene_name="Scene",
        objects=objects,
        materials=[],
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


def _object(name: str, x: float = 0.0) -> ObjectSnapshot:
    return ObjectSnapshot(
        name=name,
        type="MESH",
        primitive_hint="cube",
        location=SnapshotVector3(x=x, y=0, z=0),
        rotation_euler=SnapshotVector3(x=0, y=0, z=0),
        scale=SnapshotVector3(x=1, y=1, z=1),
        dimensions=SnapshotVector3(x=2, y=2, z=2),
        material_slots=[],
        parent=None,
        collection_names=["Collection"],
        vertex_count=None,
        polygon_count=None,
    )


def _task() -> BenchmarkTask:
    return BenchmarkTask(
        id="geometry_001_basic_primitives",
        title="Create cube",
        category=TaskCategory.GEOMETRY,
        difficulty=DifficultyLevel.EASY,
        prompt="Create cube.",
        tags=["geometry"],
        allowed_tools=[],
        expected_scene=ExpectedScene(objects=[
            ExpectedObject(name="Cube", type="MESH", primitive="cube", location=Vector3(x=0, y=0, z=0), tolerance=0.05)
        ]),
        success_criteria=[SuccessCriterion(metric="geometry_accuracy", weight=1.0)],
    )


def test_validation_result_contains_check_table() -> None:
    result = SceneValidator().validate(_task(), _snapshot([_object("Cube")]))

    assert result.overall_status is ValidationStatus.PASSED
    assert result.check_table
    assert any(row.validator_name == "object_validator" for row in result.check_table)


def test_check_table_supports_status_field() -> None:
    from benchmark.validation.checks import check_row
    from benchmark.validation.models import CheckStatus

    row = check_row(
        validator_name="transform_validator",
        check_name="dimensions",
        entity_ref="Cube",
        field="dimensions",
        expected="x=1,y=1,z=1",
        actual="n/a",
        passed=False,
        status=CheckStatus.SKIP,
    )
    assert row.status is CheckStatus.SKIP
    assert row.actual == "n/a"


def test_check_table_does_not_break_old_validation_result_json() -> None:
    old_json = """{"task_id":"t","overall_status":"passed","total_score":1.0,"validators":[],"issues":[],"summary":{}}"""

    result = SceneValidationResult.model_validate_json(old_json)

    assert result.check_table == []


def test_failed_check_contains_expected_actual_issue_code() -> None:
    result = SceneValidator().validate(_task(), _snapshot([_object("Cube", x=10.0)]))

    failed = [row for row in result.check_table if row.passed is False]
    assert failed
    assert failed[0].expected is not None
    assert failed[0].actual is not None
    assert failed[0].issue_code
