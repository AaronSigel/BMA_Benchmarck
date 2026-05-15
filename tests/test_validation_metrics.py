from benchmark.metrics.validation_metrics import metrics_from_validation_result
from benchmark.validation.models import (
    MetricScore,
    SceneValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    ValidatorResult,
)


def test_metrics_from_validation_result_extracts_summary_and_scores() -> None:
    result = SceneValidationResult(
        task_id="geometry_001_basic_primitives",
        overall_status=ValidationStatus.PASSED,
        total_score=0.9,
        validators=[
            ValidatorResult(
                name="object_validator",
                status=ValidationStatus.PASSED,
                score=1.0,
                metrics=[
                    MetricScore(
                        name="object_existence",
                        score=1.0,
                        weight=0.6,
                        passed=True,
                    )
                ],
            ),
            ValidatorResult(
                name="material_validator",
                status=ValidationStatus.FAILED,
                score=0.5,
                metrics=[
                    MetricScore(
                        name="material_accuracy",
                        score=0.5,
                        weight=0.4,
                        passed=False,
                    )
                ],
            ),
            ValidatorResult(
                name="camera_validator",
                status=ValidationStatus.SKIPPED,
                score=0.0,
            ),
        ],
        issues=[
            ValidationIssue(
                code="material_mismatch",
                message="Material mismatch.",
                severity=ValidationSeverity.ERROR,
            )
        ],
        summary={
            "issues_total": 1,
            "error_count": 1,
            "validators_passed": 1,
            "validators_failed": 1,
            "validators_skipped": 1,
        },
    )

    metrics = metrics_from_validation_result(
        run_id="run_001",
        task_id="geometry_001_basic_primitives",
        result=result,
    )
    by_name = {metric.name: metric for metric in metrics}

    assert by_name["total_score"].value == 0.9
    assert by_name["overall_status"].value == "passed"
    assert by_name["issues_total"].value == 1
    assert by_name["error_count"].value == 1
    assert by_name["validators_passed"].value == 1
    assert by_name["validators_failed"].value == 1
    assert by_name["validators_skipped"].value == 1
    assert by_name["validator.object_validator.score"].value == 1.0
    assert by_name["validator.material_validator.score"].value == 0.5
    assert by_name["metric.object_validator.object_existence.score"].value == 1.0
    assert by_name["metric.material_validator.material_accuracy.score"].value == 0.5

    assert by_name["overall_status"].run_id == "run_001"
    assert by_name["overall_status"].task_id == "geometry_001_basic_primitives"
    assert by_name["overall_status"].group == "validation"
    assert by_name["overall_status"].source == "validation_result"


def test_metrics_from_validation_result_counts_when_summary_is_missing() -> None:
    result = SceneValidationResult(
        task_id="geometry_001_basic_primitives",
        overall_status=ValidationStatus.FAILED,
        total_score=0.4,
        validators=[
            ValidatorResult(name="object_validator", status=ValidationStatus.FAILED, score=0.0),
            ValidatorResult(name="camera_validator", status=ValidationStatus.SKIPPED, score=0.0),
        ],
        issues=[
            ValidationIssue(
                code="object_missing",
                message="Object missing.",
                severity=ValidationSeverity.ERROR,
            )
        ],
        summary={},
    )

    by_name = {
        metric.name: metric
        for metric in metrics_from_validation_result("run_001", result.task_id, result)
    }

    assert by_name["issues_total"].value == 1
    assert by_name["error_count"].value == 1
    assert by_name["validators_passed"].value == 0
    assert by_name["validators_failed"].value == 1
    assert by_name["validators_skipped"].value == 1
