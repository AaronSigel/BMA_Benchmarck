"""Aggregate scene validator entry point."""

from inspect import signature
from pathlib import Path

from benchmark.blender.models import SceneSnapshot
from benchmark.tasks.models import BenchmarkTask, SuccessCriterion
from benchmark.validation.models import (
    SceneValidationResult,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    ValidatorResult,
)
from benchmark.validation.scoring import weighted_average
from benchmark.validation.validators import (
    CameraValidator,
    ExportValidator,
    LightValidator,
    MaterialValidator,
    ObjectValidator,
    TransformValidator,
)

DEFAULT_VALIDATOR_WEIGHT = 1.0
VALIDATOR_METRIC_ALIASES: dict[str, set[str]] = {
    "object_validator": {"object_existence", "geometry_accuracy"},
    "transform_validator": {"object_placement", "geometry_accuracy"},
    "material_validator": {"material_accuracy", "parameter_correctness"},
    "light_validator": {"light_existence", "lighting_correctness", "parameter_correctness"},
    "camera_validator": {"camera_existence", "camera_correctness", "target_visibility"},
    "export_validator": {"export_validity"},
}


class SceneValidator:
    def __init__(self, validators: list | None = None) -> None:
        self.validators = validators or [
            ObjectValidator(),
            TransformValidator(),
            MaterialValidator(),
            LightValidator(),
            CameraValidator(),
            ExportValidator(),
        ]

    def validate(
        self,
        task: BenchmarkTask,
        snapshot: SceneSnapshot,
        artifacts_dir: Path | None = None,
    ) -> SceneValidationResult:
        validator_results = [
            self._run_validator(validator, task, snapshot, artifacts_dir)
            for validator in self.validators
        ]
        active_results = [
            result for result in validator_results if result.status is not ValidationStatus.SKIPPED
        ]
        total_score = weighted_average(
            [
                (result.score, self._validator_weight(result.name, task.success_criteria))
                for result in active_results
            ]
        )
        issues = [
            issue
            for result in validator_results
            for issue in result.issues
        ]
        has_required_error = any(
            self._has_required_error(result, task.success_criteria)
            for result in active_results
        )
        overall_status = self._overall_status(total_score, has_required_error)

        return SceneValidationResult(
            task_id=task.id,
            overall_status=overall_status,
            total_score=total_score,
            validators=validator_results,
            issues=issues,
            summary=self._summary(validator_results, active_results, task.success_criteria),
        )

    def _run_validator(
        self,
        validator,
        task: BenchmarkTask,
        snapshot: SceneSnapshot,
        artifacts_dir: Path | None,
    ) -> ValidatorResult:
        validate = validator.validate
        parameters = signature(validate).parameters
        if "artifacts_dir" in parameters:
            return validate(task, snapshot, artifacts_dir=artifacts_dir)
        return validate(task, snapshot)

    def _validator_weight(
        self,
        validator_name: str,
        success_criteria: list[SuccessCriterion],
    ) -> float:
        aliases = VALIDATOR_METRIC_ALIASES.get(validator_name, {validator_name})
        matched_weights = [
            criterion.weight
            for criterion in success_criteria
            if criterion.metric in aliases or criterion.metric == validator_name
        ]
        return sum(matched_weights) if matched_weights else DEFAULT_VALIDATOR_WEIGHT

    def _has_required_error(
        self,
        result: ValidatorResult,
        success_criteria: list[SuccessCriterion],
    ) -> bool:
        if not any(issue.severity is ValidationSeverity.ERROR for issue in result.issues):
            return False

        aliases = VALIDATOR_METRIC_ALIASES.get(result.name, {result.name})
        matched_criteria = [
            criterion
            for criterion in success_criteria
            if criterion.metric in aliases or criterion.metric == result.name
        ]
        if not matched_criteria:
            return True
        return any(criterion.required for criterion in matched_criteria)

    def _overall_status(
        self,
        total_score: float,
        has_required_error: bool,
    ) -> ValidationStatus:
        if total_score < 0.6 or has_required_error:
            return ValidationStatus.FAILED
        if total_score >= 0.85:
            return ValidationStatus.PASSED
        return ValidationStatus.WARNING

    def _summary(
        self,
        validator_results: list[ValidatorResult],
        active_results: list[ValidatorResult],
        success_criteria: list[SuccessCriterion],
    ) -> dict[str, object]:
        return {
            "validators_total": len(validator_results),
            "validators_run": len(active_results),
            "validators_skipped": sum(
                1 for result in validator_results if result.status is ValidationStatus.SKIPPED
            ),
            "validators_passed": sum(
                1 for result in validator_results if result.status is ValidationStatus.PASSED
            ),
            "validators_failed": sum(
                1 for result in validator_results if result.status is ValidationStatus.FAILED
            ),
            "issues_total": sum(len(result.issues) for result in validator_results),
            "error_count": sum(
                1
                for result in validator_results
                for issue in result.issues
                if issue.severity is ValidationSeverity.ERROR
            ),
            "weights": {
                result.name: self._validator_weight(result.name, success_criteria)
                for result in active_results
            },
        }
