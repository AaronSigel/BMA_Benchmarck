"""Aggregate scene validator entry point."""

import re
from collections import Counter
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
    GlbImportBackValidator,
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
    "glb_import_back_validator": {"export_import_validity", "export_validity"},
}

_SCENE_VALIDATOR_NAMES = frozenset({
    "object_validator",
    "transform_validator",
    "material_validator",
    "light_validator",
    "camera_validator",
})
_EXPORT_VALIDATOR_NAMES = frozenset({"export_validator", "glb_import_back_validator"})


class SceneValidator:
    def __init__(self, validators: list | None = None) -> None:
        self.validators = validators or [
            ObjectValidator(),
            TransformValidator(),
            MaterialValidator(),
            LightValidator(),
            CameraValidator(),
            ExportValidator(),
            GlbImportBackValidator(),
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
        check_table = [
            row
            for result in validator_results
            for row in result.check_table
        ]
        contamination_summary, contamination_issues = self._scene_contamination_summary(task, snapshot)
        issues.extend(contamination_issues)
        issue_counts = Counter(issue.code for issue in issues)
        has_required_error = any(
            self._has_required_error(result, task.success_criteria)
            for result in active_results
        )
        overall_status = self._overall_status(total_score, has_required_error)

        summary = self._summary(validator_results, active_results, task.success_criteria)
        summary["issues_total"] = int(summary["issues_total"]) + len(contamination_issues)
        summary["issue_counts"] = dict(sorted(issue_counts.items()))
        summary["render_source"] = "final_scene"
        summary.update(contamination_summary)

        return SceneValidationResult(
            task_id=task.id,
            overall_status=overall_status,
            total_score=total_score,
            validators=validator_results,
            issues=issues,
            summary=summary,
            check_table=check_table,
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
        if not matched_weights:
            return DEFAULT_VALIDATOR_WEIGHT
        total = sum(matched_weights)
        if (
            validator_name in _EXPORT_VALIDATOR_NAMES
            and any(criterion.metric == "export_validity" for criterion in success_criteria)
        ):
            export_validity_weight = sum(
                criterion.weight
                for criterion in success_criteria
                if criterion.metric == "export_validity"
            )
            if export_validity_weight > 0:
                return export_validity_weight / len(_EXPORT_VALIDATOR_NAMES)
        return total

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

    def _component_score(
        self,
        validator_results: list[ValidatorResult],
        names: frozenset[str],
        success_criteria: list[SuccessCriterion],
    ) -> float | None:
        subset = [result for result in validator_results if result.name in names and result.status is not ValidationStatus.SKIPPED]
        if not subset:
            return None
        return weighted_average(
            [
                (result.score, self._validator_weight(result.name, success_criteria))
                for result in subset
            ]
        )

    def _summary(
        self,
        validator_results: list[ValidatorResult],
        active_results: list[ValidatorResult],
        success_criteria: list[SuccessCriterion],
    ) -> dict[str, object]:
        scene_score = self._component_score(validator_results, _SCENE_VALIDATOR_NAMES, success_criteria)
        export_score = self._component_score(
            validator_results,
            frozenset({"export_validator"}),
            success_criteria,
        )
        import_back_score = self._component_score(
            validator_results,
            frozenset({"glb_import_back_validator"}),
            success_criteria,
        )
        final_score = weighted_average(
            [
                (result.score, self._validator_weight(result.name, success_criteria))
                for result in active_results
            ]
        )
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
            "validation_coverage": (
                len(active_results) / len(validator_results) if validator_results else 0.0
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
            "scores": {
                "scene_score": scene_score,
                "export_score": export_score,
                "import_back_score": import_back_score,
                "final_score": final_score,
            },
        }

    def _scene_contamination_summary(
        self,
        task: BenchmarkTask,
        snapshot: SceneSnapshot,
    ) -> tuple[dict[str, object], list[ValidationIssue]]:
        mesh_objects = [obj for obj in snapshot.objects if obj.type.upper() == "MESH"]
        actual_object_count = len(mesh_objects)
        expected_object_count = len(task.expected_scene.objects)
        extra_object_count = max(0, actual_object_count - expected_object_count)
        actual_light_count = len(snapshot.lights)
        expected_light_count = len(task.expected_scene.lights)
        extra_light_count = max(0, actual_light_count - expected_light_count)
        actual_camera_count = len(snapshot.cameras)
        expected_camera_count = len(task.expected_scene.cameras)
        extra_camera_count = max(0, actual_camera_count - expected_camera_count)
        duplicate_name_count = _duplicate_base_name_count([obj.name for obj in mesh_objects])

        issues: list[ValidationIssue] = []
        if extra_object_count > 0:
            issues.append(
                ValidationIssue(
                    code="scene_contains_unexpected_objects",
                    message=(
                        "Scene contains more objects than expected; this may indicate "
                        "cross-run contamination."
                    ),
                    severity=ValidationSeverity.WARNING,
                    expected_path="expected_scene.objects",
                    actual_path="snapshot.objects",
                    expected_value=expected_object_count,
                    actual_value=actual_object_count,
                )
            )
        if extra_light_count > 0:
            issues.append(
                ValidationIssue(
                    code="scene_contains_unexpected_lights",
                    message=(
                        "Scene contains more lights than expected; this may indicate "
                        "cross-run contamination."
                    ),
                    severity=ValidationSeverity.WARNING,
                    expected_path="expected_scene.lights",
                    actual_path="snapshot.lights",
                    expected_value=expected_light_count,
                    actual_value=actual_light_count,
                )
            )
        if extra_camera_count > 0:
            issues.append(
                ValidationIssue(
                    code="scene_contains_unexpected_cameras",
                    message=(
                        "Scene contains more cameras than expected; this may indicate "
                        "cross-run contamination."
                    ),
                    severity=ValidationSeverity.WARNING,
                    expected_path="expected_scene.cameras",
                    actual_path="snapshot.cameras",
                    expected_value=expected_camera_count,
                    actual_value=actual_camera_count,
                )
            )
        if duplicate_name_count > 0:
            issues.append(
                ValidationIssue(
                    code="duplicate_object_detected",
                    message="Scene contains duplicate Blender base object names.",
                    severity=ValidationSeverity.ERROR,
                    expected_path="expected_scene.objects",
                    actual_path="snapshot.objects",
                    expected_value=0,
                    actual_value=duplicate_name_count,
                )
            )

        return (
            {
                "actual_object_count": actual_object_count,
                "expected_object_count": expected_object_count,
                "extra_object_count": extra_object_count,
                "mesh_object_count": snapshot.mesh_object_count if snapshot.mesh_object_count is not None else actual_object_count,
                "light_count": snapshot.light_count if snapshot.light_count is not None else actual_light_count,
                "camera_count": snapshot.camera_count if snapshot.camera_count is not None else actual_camera_count,
                "all_object_count": snapshot.all_object_count
                if snapshot.all_object_count is not None
                else actual_object_count + actual_light_count + actual_camera_count,
                "expected_light_count": expected_light_count,
                "extra_light_count": extra_light_count,
                "expected_camera_count": expected_camera_count,
                "extra_camera_count": extra_camera_count,
                "duplicate_name_count": duplicate_name_count,
            },
            issues,
        )


_BLENDER_NUMERIC_SUFFIX_RE = re.compile(r"\.\d{3}$")


def _duplicate_base_name_count(names: list[str]) -> int:
    bases = [_BLENDER_NUMERIC_SUFFIX_RE.sub("", name) for name in names]
    return sum(count - 1 for count in Counter(bases).values() if count > 1)
