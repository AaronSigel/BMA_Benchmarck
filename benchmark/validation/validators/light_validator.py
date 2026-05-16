"""Validation of expected lights."""

import math

from benchmark.blender.models import LightSnapshot, SceneSnapshot, Vector3 as BlenderVector3
from benchmark.tasks.models import BenchmarkTask, ExpectedLight, Vector3


def _deg_to_rad_v3(v: Vector3) -> BlenderVector3:
    return BlenderVector3(x=math.radians(v.x), y=math.radians(v.y), z=math.radians(v.z))
from benchmark.validation.matcher import SceneMatcher
from benchmark.validation.models import (
    MetricScore,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    ValidatorResult,
)
from benchmark.validation.scoring import tolerance_score, vector_tolerance_score, weighted_average


class LightValidator:
    name = "light_validator"

    def __init__(self, matcher: SceneMatcher | None = None) -> None:
        self.matcher = matcher or SceneMatcher()

    def validate(self, task: BenchmarkTask, snapshot: SceneSnapshot) -> ValidatorResult:
        expected_lights = task.expected_scene.lights
        if not expected_lights:
            return ValidatorResult(
                name=self.name,
                status=ValidationStatus.SKIPPED,
                score=0.0,
                metrics=[],
            )

        issues: list[ValidationIssue] = []
        existence_scores: list[float] = []
        type_scores: list[float] = []
        transform_scores: list[float] = []
        energy_scores: list[float] = []
        available_lights = list(snapshot.lights)

        for expected_index, expected in enumerate(expected_lights):
            expected_path = f"expected_scene.lights[{expected_index}]"
            actual = self.matcher.match_expected_light(expected, available_lights)
            if actual is None:
                issue = self._missing_issue(expected, expected_path)
                issues.append(issue)
                existence_scores.append(0.0)
                type_scores.append(0.0)
                transform_scores.append(0.0 if self._has_transform_expectations(expected) else 1.0)
                energy_scores.append(0.0 if expected.energy is not None else 1.0)
                continue

            available_lights.remove(actual)
            actual_index = snapshot.lights.index(actual)
            actual_path = f"snapshot.lights[{actual_index}]"
            existence_scores.append(1.0)

            type_score = 1.0 if expected.type.upper() == actual.type.upper() else 0.0
            type_scores.append(type_score)
            if type_score < 1.0:
                issues.append(self._type_mismatch_issue(expected, actual, expected_path, actual_path))

            transform_score = self._transform_score(expected, actual)
            transform_scores.append(transform_score)
            self._append_transform_issues(expected, actual, expected_path, actual_path, issues)

            energy_score = self._energy_score(expected, actual)
            energy_scores.append(energy_score)
            if energy_score < 1.0:
                issues.append(self._energy_mismatch_issue(expected, actual, expected_path, actual_path))

        metrics = [
            self._metric("light_existence_score", existence_scores, 0.4),
            self._metric("light_type_score", type_scores, 0.2),
            self._metric("light_transform_score", transform_scores, 0.2),
            self._metric("light_energy_score", energy_scores, 0.2),
        ]
        score = weighted_average([(metric.score, metric.weight) for metric in metrics])
        status = ValidationStatus.PASSED if not issues and score == 1.0 else ValidationStatus.FAILED
        return ValidatorResult(
            name=self.name,
            status=status,
            score=score,
            issues=issues,
            metrics=metrics,
        )

    def _metric(self, name: str, scores: list[float], weight: float) -> MetricScore:
        score = sum(scores) / len(scores) if scores else 1.0
        return MetricScore(name=name, score=score, weight=weight, passed=score == 1.0)

    def _has_transform_expectations(self, expected: ExpectedLight) -> bool:
        return expected.location is not None or expected.rotation is not None

    def _transform_score(self, expected: ExpectedLight, actual: LightSnapshot) -> float:
        scores: list[float] = []
        if expected.location is not None:
            scores.append(vector_tolerance_score(expected.location, actual.location, expected.tolerance))
        if expected.rotation is not None:
            scores.append(
                vector_tolerance_score(
                    _deg_to_rad_v3(expected.rotation), actual.rotation_euler, expected.tolerance
                )
            )
        return sum(scores) / len(scores) if scores else 1.0

    def _append_transform_issues(
        self,
        expected: ExpectedLight,
        actual: LightSnapshot,
        expected_path: str,
        actual_path: str,
        issues: list[ValidationIssue],
    ) -> None:
        if expected.location is not None:
            score = vector_tolerance_score(expected.location, actual.location, expected.tolerance)
            if score < 1.0:
                issues.append(
                    self._vector_mismatch_issue(
                        code="light_location_mismatch",
                        field="location",
                        expected_value=expected.location,
                        actual_value=actual.location,
                        tolerance=expected.tolerance,
                        expected_path=expected_path,
                        actual_path=actual_path,
                    )
                )

        if expected.rotation is not None:
            score = vector_tolerance_score(
                _deg_to_rad_v3(expected.rotation), actual.rotation_euler, expected.tolerance
            )
            if score < 1.0:
                issues.append(
                    self._vector_mismatch_issue(
                        code="light_rotation_mismatch",
                        field="rotation",
                        expected_value=_deg_to_rad_v3(expected.rotation),
                        actual_value=actual.rotation_euler,
                        tolerance=expected.tolerance,
                        expected_path=expected_path,
                        actual_path=actual_path,
                    )
                )

    def _energy_score(self, expected: ExpectedLight, actual: LightSnapshot) -> float:
        if expected.energy is None:
            return 1.0
        if actual.energy is None:
            return 0.0
        return tolerance_score(expected.energy, actual.energy, expected.tolerance)

    def _missing_issue(self, expected: ExpectedLight, expected_path: str) -> ValidationIssue:
        return ValidationIssue(
            code="light_missing",
            message=f"Expected light was not found: {expected.name or expected.type}.",
            severity=ValidationSeverity.ERROR,
            expected_path=expected_path,
            actual_path=None,
            expected_value=expected.model_dump(mode="json", exclude_none=True),
            actual_value=None,
        )

    def _type_mismatch_issue(
        self,
        expected: ExpectedLight,
        actual: LightSnapshot,
        expected_path: str,
        actual_path: str,
    ) -> ValidationIssue:
        return ValidationIssue(
            code="light_type_mismatch",
            message=f"Expected light type {expected.type!r}, got {actual.type!r}.",
            severity=ValidationSeverity.ERROR,
            expected_path=f"{expected_path}.type",
            actual_path=f"{actual_path}.type",
            expected_value=expected.type,
            actual_value=actual.type,
        )

    def _vector_mismatch_issue(
        self,
        code: str,
        field: str,
        expected_value,
        actual_value,
        tolerance: float,
        expected_path: str,
        actual_path: str,
    ) -> ValidationIssue:
        actual_field = "rotation_euler" if field == "rotation" else field
        return ValidationIssue(
            code=code,
            message=f"Expected light {field} within tolerance {tolerance}, got a different value.",
            severity=ValidationSeverity.ERROR,
            expected_path=f"{expected_path}.{field}",
            actual_path=f"{actual_path}.{actual_field}",
            expected_value=expected_value.model_dump(mode="json"),
            actual_value=actual_value.model_dump(mode="json"),
        )

    def _energy_mismatch_issue(
        self,
        expected: ExpectedLight,
        actual: LightSnapshot,
        expected_path: str,
        actual_path: str,
    ) -> ValidationIssue:
        return ValidationIssue(
            code="light_energy_mismatch",
            message=f"Expected light energy within tolerance {expected.tolerance}, got a different value.",
            severity=ValidationSeverity.ERROR,
            expected_path=f"{expected_path}.energy",
            actual_path=f"{actual_path}.energy",
            expected_value=expected.energy,
            actual_value=actual.energy,
        )
