"""Validation of expected object transforms."""

import math
from typing import Literal

from benchmark.blender.models import ObjectSnapshot, SceneSnapshot, Vector3 as BlenderVector3
from benchmark.tasks.models import BenchmarkTask, ExpectedObject, Vector3


def _deg_to_rad_v3(v: Vector3) -> BlenderVector3:
    """Convert a Vector3 in degrees to a BlenderVector3 in radians."""
    return BlenderVector3(x=math.radians(v.x), y=math.radians(v.y), z=math.radians(v.z))
from benchmark.validation.matcher import SceneMatcher
from benchmark.validation.models import (
    MetricScore,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    ValidatorResult,
)
from benchmark.validation.scoring import vector_tolerance_score, weighted_average

TransformField = Literal["location", "rotation", "scale", "dimensions"]


class TransformValidator:
    name = "transform_validator"

    def __init__(self, matcher: SceneMatcher | None = None) -> None:
        self.matcher = matcher or SceneMatcher()

    def validate(self, task: BenchmarkTask, snapshot: SceneSnapshot) -> ValidatorResult:
        expected_objects = task.expected_scene.objects
        available_objects = [obj for obj in snapshot.objects if obj.type.upper() == "MESH"]
        issues: list[ValidationIssue] = []
        metric_scores: list[MetricScore] = []

        for expected_index, expected in enumerate(expected_objects):
            fields = self._expected_transform_fields(expected)
            if not fields:
                continue

            expected_path = f"expected_scene.objects[{expected_index}]"
            actual = self.matcher.match_expected_object(expected, available_objects)
            if actual is None:
                issues.append(self._missing_issue(expected, expected_path))
                for field in fields:
                    metric_scores.append(
                        MetricScore(
                            name=f"{field}_score",
                            score=0.0,
                            passed=False,
                            issues=[issues[-1]],
                        )
                    )
                continue

            available_objects.remove(actual)
            actual_index = snapshot.objects.index(actual)
            actual_path = f"snapshot.objects[{actual_index}]"

            for field in fields:
                field_score = self._field_score(expected, actual, field)
                field_issues: list[ValidationIssue] = []
                if field_score < 1.0:
                    issue = self._mismatch_issue(expected, actual, field, expected_path, actual_path)
                    issues.append(issue)
                    field_issues.append(issue)

                metric_scores.append(
                    MetricScore(
                        name=f"{field}_score",
                        score=field_score,
                        passed=field_score == 1.0,
                        issues=field_issues,
                    )
                )

        if not metric_scores:
            return ValidatorResult(
                name=self.name,
                status=ValidationStatus.SKIPPED,
                score=0.0,
                metrics=[],
            )

        score = weighted_average([(metric.score, metric.weight) for metric in metric_scores])
        status = ValidationStatus.PASSED if not issues and score == 1.0 else ValidationStatus.FAILED
        return ValidatorResult(
            name=self.name,
            status=status,
            score=score,
            issues=issues,
            metrics=metric_scores,
        )

    def _expected_transform_fields(self, expected: ExpectedObject) -> list[TransformField]:
        fields: list[TransformField] = []
        if expected.location is not None:
            fields.append("location")
        if expected.rotation is not None:
            fields.append("rotation")
        if expected.scale is not None:
            fields.append("scale")
        if expected.dimensions is not None:
            fields.append("dimensions")
        return fields

    def _field_score(
        self,
        expected: ExpectedObject,
        actual: ObjectSnapshot,
        field: TransformField,
    ) -> float:
        expected_value = self._expected_value(expected, field)
        actual_value = self._actual_value(actual, field)
        if field == "rotation":
            expected_value = _deg_to_rad_v3(expected_value)
        return vector_tolerance_score(expected_value, actual_value, expected.tolerance)

    def _expected_value(self, expected: ExpectedObject, field: TransformField) -> Vector3:
        value = getattr(expected, field)
        if value is None:
            raise ValueError(f"Expected transform field is not set: {field}")
        return value

    def _actual_value(self, actual: ObjectSnapshot, field: TransformField):
        if field == "rotation":
            return actual.rotation_euler
        return getattr(actual, field)

    def _missing_issue(self, expected: ExpectedObject, expected_path: str) -> ValidationIssue:
        return ValidationIssue(
            code="object_missing_for_transform",
            message=f"Expected object was not found for transform validation: {expected.name or expected.type}.",
            severity=ValidationSeverity.ERROR,
            expected_path=expected_path,
            actual_path=None,
            expected_value=expected.model_dump(mode="json", exclude_none=True),
            actual_value=None,
        )

    def _mismatch_issue(
        self,
        expected: ExpectedObject,
        actual: ObjectSnapshot,
        field: TransformField,
        expected_path: str,
        actual_path: str,
    ) -> ValidationIssue:
        expected_value = self._expected_value(expected, field)
        actual_value = self._actual_value(actual, field)
        actual_field = "rotation_euler" if field == "rotation" else field
        expected_display = (
            _deg_to_rad_v3(expected_value).model_dump(mode="json")
            if field == "rotation"
            else expected_value.model_dump(mode="json")
        )
        return ValidationIssue(
            code=f"{field}_mismatch",
            message=f"Expected {field} within tolerance {expected.tolerance}, got a different value.",
            severity=ValidationSeverity.ERROR,
            expected_path=f"{expected_path}.{field}",
            actual_path=f"{actual_path}.{actual_field}",
            expected_value=expected_display,
            actual_value=actual_value.model_dump(mode="json"),
        )
