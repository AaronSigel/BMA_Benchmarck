"""Validation of expected object existence and basic object identity."""

from benchmark.blender.models import ObjectSnapshot, SceneSnapshot
from benchmark.tasks.models import BenchmarkTask, ExpectedObject
from benchmark.validation.matcher import SceneMatcher, normalize_name
from benchmark.validation.models import (
    MetricScore,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    ValidatorResult,
)
from benchmark.validation.scoring import weighted_average


class ObjectValidator:
    name = "object_validator"

    def __init__(self, matcher: SceneMatcher | None = None) -> None:
        self.matcher = matcher or SceneMatcher()

    def validate(self, task: BenchmarkTask, snapshot: SceneSnapshot) -> ValidatorResult:
        expected_objects = task.expected_scene.objects
        if not expected_objects:
            return ValidatorResult(
                name=self.name,
                status=ValidationStatus.SKIPPED,
                score=0.0,
                metrics=[],
            )

        issues: list[ValidationIssue] = []
        found_count = 0
        type_match_count = 0
        primitive_expected_count = 0
        primitive_match_count = 0
        available_objects = list(snapshot.objects)

        for expected_index, expected in enumerate(expected_objects):
            actual = self.matcher.match_expected_object(expected, available_objects)
            expected_path = f"expected_scene.objects[{expected_index}]"

            if actual is None:
                issues.append(self._missing_issue(expected, expected_path))
                continue

            found_count += 1
            available_objects.remove(actual)
            actual_index = snapshot.objects.index(actual)
            actual_path = f"snapshot.objects[{actual_index}]"

            if self._type_matches(expected, actual):
                type_match_count += 1
            else:
                issues.append(self._type_mismatch_issue(expected, actual, expected_path, actual_path))

            if expected.primitive is not None:
                primitive_expected_count += 1
                if self._primitive_matches(expected, actual):
                    primitive_match_count += 1
                else:
                    issues.append(
                        self._primitive_mismatch_issue(expected, actual, expected_path, actual_path)
                    )

        object_existence_score = found_count / len(expected_objects)
        type_score = type_match_count / found_count if found_count else 0.0
        primitive_score = (
            primitive_match_count / primitive_expected_count if primitive_expected_count else 1.0
        )
        score = weighted_average(
            [
                (object_existence_score, 0.6),
                (type_score, 0.2),
                (primitive_score, 0.2),
            ]
        )
        status = ValidationStatus.PASSED if not issues and score == 1.0 else ValidationStatus.FAILED

        return ValidatorResult(
            name=self.name,
            status=status,
            score=score,
            issues=issues,
            metrics=[
                MetricScore(
                    name="object_existence_score",
                    score=object_existence_score,
                    weight=0.6,
                    passed=object_existence_score == 1.0,
                ),
                MetricScore(
                    name="type_score",
                    score=type_score,
                    weight=0.2,
                    passed=type_score == 1.0,
                ),
                MetricScore(
                    name="primitive_score",
                    score=primitive_score,
                    weight=0.2,
                    passed=primitive_score == 1.0,
                ),
            ],
        )

    def _missing_issue(self, expected: ExpectedObject, expected_path: str) -> ValidationIssue:
        return ValidationIssue(
            code="object_missing",
            message=f"Expected object was not found: {expected.name or expected.type}.",
            severity=ValidationSeverity.ERROR,
            expected_path=expected_path,
            actual_path=None,
            expected_value=expected.model_dump(mode="json", exclude_none=True),
            actual_value=None,
        )

    def _type_mismatch_issue(
        self,
        expected: ExpectedObject,
        actual: ObjectSnapshot,
        expected_path: str,
        actual_path: str,
    ) -> ValidationIssue:
        return ValidationIssue(
            code="object_type_mismatch",
            message=f"Expected object type {expected.type!r}, got {actual.type!r}.",
            severity=ValidationSeverity.ERROR,
            expected_path=f"{expected_path}.type",
            actual_path=f"{actual_path}.type",
            expected_value=expected.type,
            actual_value=actual.type,
        )

    def _primitive_mismatch_issue(
        self,
        expected: ExpectedObject,
        actual: ObjectSnapshot,
        expected_path: str,
        actual_path: str,
    ) -> ValidationIssue:
        return ValidationIssue(
            code="primitive_mismatch",
            message=f"Expected primitive {expected.primitive!r}, got {actual.primitive_hint!r}.",
            severity=ValidationSeverity.ERROR,
            expected_path=f"{expected_path}.primitive",
            actual_path=f"{actual_path}.primitive_hint",
            expected_value=expected.primitive,
            actual_value=actual.primitive_hint,
        )

    def _type_matches(self, expected: ExpectedObject, actual: ObjectSnapshot) -> bool:
        return expected.type.lower() == actual.type.lower()

    def _primitive_matches(self, expected: ExpectedObject, actual: ObjectSnapshot) -> bool:
        if expected.primitive is None or actual.primitive_hint is None:
            return False
        return normalize_name(expected.primitive) == normalize_name(actual.primitive_hint)
