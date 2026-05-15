import pytest
from pydantic import ValidationError

from benchmark.validation.models import (
    MetricScore,
    SceneValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    ValidatorResult,
)


def make_validation_result() -> SceneValidationResult:
    issue = ValidationIssue(
        code="missing_object",
        message="Expected object was not found.",
        severity=ValidationSeverity.ERROR,
        expected_path="expected_scene.objects[0]",
        actual_path=None,
        expected_value={"name": "Cube"},
        actual_value=None,
    )
    metric = MetricScore(
        name="object_existence",
        score=0.0,
        weight=0.5,
        passed=False,
        issues=[issue],
    )
    validator = ValidatorResult(
        name="object_validator",
        status=ValidationStatus.FAILED,
        score=0.0,
        issues=[issue],
        metrics=[metric],
    )
    return SceneValidationResult(
        task_id="geometry_001_basic_primitives",
        overall_status=ValidationStatus.FAILED,
        total_score=0.0,
        validators=[validator],
        issues=[issue],
        summary={"validators_run": 1},
    )


def test_scene_validation_result_json_round_trip() -> None:
    result = make_validation_result()

    raw_json = result.model_dump_json()
    restored = SceneValidationResult.model_validate_json(raw_json)

    assert restored == result
    assert restored.overall_status is ValidationStatus.FAILED
    assert restored.issues[0].severity is ValidationSeverity.ERROR


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: MetricScore(name="invalid", score=1.1, passed=False),
        lambda: MetricScore(name="invalid", score=-0.1, passed=False),
        lambda: MetricScore(name="invalid", score=0.5, weight=1.1, passed=True),
        lambda: ValidatorResult(name="invalid", status=ValidationStatus.FAILED, score=1.1),
        lambda: SceneValidationResult(
            task_id="invalid",
            overall_status=ValidationStatus.FAILED,
            total_score=1.1,
            summary={},
        ),
    ],
)
def test_score_and_weight_ranges_are_validated(model_factory) -> None:
    with pytest.raises(ValidationError):
        model_factory()


def test_issue_lists_are_not_shared_between_instances() -> None:
    first = MetricScore(name="first", score=1.0, passed=True)
    second = MetricScore(name="second", score=1.0, passed=True)

    first.issues.append(
        ValidationIssue(
            code="note",
            message="A note.",
            severity=ValidationSeverity.INFO,
        )
    )

    assert second.issues == []
