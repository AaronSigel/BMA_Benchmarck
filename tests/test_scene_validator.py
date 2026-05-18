from pathlib import Path

from benchmark.blender.models import (
    ObjectSnapshot,
    RenderSettingsSnapshot,
    SceneSnapshot,
    Vector3 as SnapshotVector3,
)
from benchmark.tasks.models import (
    BenchmarkTask,
    DifficultyLevel,
    ExpectedExport,
    ExpectedObject,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
    Vector3,
)
from benchmark.validation.models import (
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    ValidatorResult,
)
from benchmark.validation.scene_validator import SceneValidator


def vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector3:
    return Vector3(x=x, y=y, z=z)


def snapshot_vector(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> SnapshotVector3:
    return SnapshotVector3(x=x, y=y, z=z)


def object_snapshot(name: str = "Cube") -> ObjectSnapshot:
    return ObjectSnapshot(
        name=name,
        type="MESH",
        primitive_hint="cube",
        location=snapshot_vector(),
        rotation_euler=snapshot_vector(),
        scale=snapshot_vector(1.0, 1.0, 1.0),
        dimensions=snapshot_vector(2.0, 2.0, 2.0),
        material_slots=[],
        parent=None,
        collection_names=["Collection"],
        vertex_count=None,
        polygon_count=None,
    )


def scene_snapshot(objects: list[ObjectSnapshot] | None = None) -> SceneSnapshot:
    return SceneSnapshot(
        scene_name="Scene",
        objects=objects or [],
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


def task_with_scene(
    expected_scene: ExpectedScene,
    success_criteria: list[SuccessCriterion] | None = None,
) -> BenchmarkTask:
    return BenchmarkTask(
        id="geometry_001_basic_primitives",
        title="Create cube",
        category=TaskCategory.GEOMETRY,
        difficulty=DifficultyLevel.EASY,
        prompt="Create the expected scene.",
        tags=["geometry"],
        allowed_tools=[],
        expected_scene=expected_scene,
        success_criteria=success_criteria
        or [
            SuccessCriterion(metric="object_existence", weight=0.4),
            SuccessCriterion(metric="geometry_accuracy", weight=0.3),
            SuccessCriterion(metric="object_placement", weight=0.3),
        ],
    )


class StubValidator:
    def __init__(
        self,
        name: str,
        status: ValidationStatus,
        score: float,
        issues: list[ValidationIssue] | None = None,
    ) -> None:
        self.name = name
        self.result = ValidatorResult(
            name=name,
            status=status,
            score=score,
            issues=issues or [],
        )

    def validate(self, task: BenchmarkTask, snapshot: SceneSnapshot) -> ValidatorResult:
        return self.result


class ArtifactAwareStubValidator(StubValidator):
    def validate(
        self,
        task: BenchmarkTask,
        snapshot: SceneSnapshot,
        artifacts_dir: Path | None = None,
    ) -> ValidatorResult:
        assert artifacts_dir is not None
        return self.result


def error_issue() -> ValidationIssue:
    return ValidationIssue(
        code="stub_error",
        message="Stub error.",
        severity=ValidationSeverity.ERROR,
    )


def test_scene_validator_returns_scene_validation_result_for_passed_scene() -> None:
    task = task_with_scene(
        ExpectedScene(
            objects=[
                ExpectedObject(
                    name="Cube",
                    type="MESH",
                    primitive="cube",
                    location=vector(0.0, 0.0, 0.0),
                )
            ]
        )
    )
    snapshot = scene_snapshot([object_snapshot("Cube.001")])

    result = SceneValidator().validate(task, snapshot)

    assert result.task_id == task.id
    assert result.overall_status is ValidationStatus.PASSED
    assert result.total_score == 1.0
    assert result.summary["validators_total"] == 6
    assert result.summary["validators_skipped"] == 4
    assert result.summary["actual_object_count"] == 1
    assert result.summary["expected_object_count"] == 1
    assert result.summary["extra_object_count"] == 0


def test_scene_validator_failed_or_warning_when_object_is_missing() -> None:
    task = task_with_scene(
        ExpectedScene(
            objects=[
                ExpectedObject(name="Cube", type="MESH", primitive="cube"),
                ExpectedObject(name="Sphere", type="MESH", primitive="sphere"),
            ]
        )
    )

    result = SceneValidator().validate(task, scene_snapshot([object_snapshot("Cube")]))

    assert result.overall_status in {ValidationStatus.FAILED, ValidationStatus.WARNING}
    assert result.total_score < 1.0
    assert any(issue.code == "object_missing" for issue in result.issues)


def test_scene_validator_warns_when_scene_contains_unexpected_objects() -> None:
    task = task_with_scene(
        ExpectedScene(objects=[ExpectedObject(name="Cube", type="MESH", primitive="cube")])
    )

    result = SceneValidator().validate(
        task,
        scene_snapshot([object_snapshot("Cube"), object_snapshot("OldCube")]),
    )

    assert result.summary["actual_object_count"] == 2
    assert result.summary["expected_object_count"] == 1
    assert result.summary["extra_object_count"] == 1
    assert any(issue.code == "scene_contains_unexpected_objects" for issue in result.issues)


def test_scene_validator_counts_duplicate_blender_base_names() -> None:
    task = task_with_scene(
        ExpectedScene(
            objects=[
                ExpectedObject(name="Cube", type="MESH", primitive="cube"),
                ExpectedObject(name="Sphere", type="MESH", primitive="sphere"),
            ]
        )
    )

    result = SceneValidator().validate(
        task,
        scene_snapshot(
            [
                object_snapshot("Cube"),
                object_snapshot("Cube.001"),
                object_snapshot("Sphere"),
            ]
        ),
    )

    assert result.summary["duplicate_name_count"] == 1


def test_scene_validator_ignores_skipped_validators_in_total_score() -> None:
    validators = [
        StubValidator("object_validator", ValidationStatus.PASSED, 1.0),
        StubValidator("material_validator", ValidationStatus.SKIPPED, 0.0),
    ]
    task = task_with_scene(ExpectedScene(), [SuccessCriterion(metric="object_existence", weight=1.0)])

    result = SceneValidator(validators=validators).validate(task, scene_snapshot())

    assert result.overall_status is ValidationStatus.PASSED
    assert result.total_score == 1.0
    assert result.summary["validators_run"] == 1
    assert result.summary["validators_skipped"] == 1


def test_scene_validator_uses_success_criteria_weights() -> None:
    validators = [
        StubValidator("object_validator", ValidationStatus.PASSED, 1.0),
        StubValidator("transform_validator", ValidationStatus.FAILED, 0.0),
    ]
    task = task_with_scene(
        ExpectedScene(),
        [
            SuccessCriterion(metric="object_existence", weight=0.25, required=False),
            SuccessCriterion(metric="object_placement", weight=0.75, required=False),
        ],
    )

    result = SceneValidator(validators=validators).validate(task, scene_snapshot())

    assert result.total_score == 0.25
    assert result.overall_status is ValidationStatus.FAILED
    assert result.summary["weights"] == {"object_validator": 0.25, "transform_validator": 0.75}


def test_scene_validator_required_error_forces_failed_status() -> None:
    validators = [
        StubValidator("object_validator", ValidationStatus.FAILED, 0.9, issues=[error_issue()]),
        StubValidator("transform_validator", ValidationStatus.PASSED, 1.0),
    ]
    task = task_with_scene(
        ExpectedScene(),
        [
            SuccessCriterion(metric="object_existence", weight=0.5, required=True),
            SuccessCriterion(metric="object_placement", weight=0.5, required=False),
        ],
    )

    result = SceneValidator(validators=validators).validate(task, scene_snapshot())

    assert result.total_score == 0.95
    assert result.overall_status is ValidationStatus.FAILED


def test_scene_validator_non_required_error_can_warn() -> None:
    validators = [
        StubValidator("object_validator", ValidationStatus.FAILED, 0.7, issues=[error_issue()]),
        StubValidator("transform_validator", ValidationStatus.PASSED, 1.0),
    ]
    task = task_with_scene(
        ExpectedScene(),
        [
            SuccessCriterion(metric="object_existence", weight=0.8, required=False),
            SuccessCriterion(metric="object_placement", weight=0.2, required=False),
        ],
    )

    result = SceneValidator(validators=validators).validate(task, scene_snapshot())

    assert result.total_score == 0.76
    assert result.overall_status is ValidationStatus.WARNING


def test_scene_validator_passes_artifacts_dir_to_export_validator(tmp_path: Path) -> None:
    validators = [ArtifactAwareStubValidator("export_validator", ValidationStatus.PASSED, 1.0)]
    task = task_with_scene(
        ExpectedScene(exports=[ExpectedExport(format="blend")]),
        [SuccessCriterion(metric="export_validity", weight=1.0)],
    )

    result = SceneValidator(validators=validators).validate(
        task,
        scene_snapshot(),
        artifacts_dir=tmp_path,
    )

    assert result.overall_status is ValidationStatus.PASSED
