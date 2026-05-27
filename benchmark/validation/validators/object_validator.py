"""Validation of expected object existence and basic object identity."""

from benchmark.blender.models import ObjectSnapshot, SceneSnapshot
from benchmark.tasks.models import BenchmarkTask, ExpectedObject
from benchmark.validation.matcher import SceneMatcher, normalize_name
from benchmark.validation.checks import check_row
from benchmark.validation.models import (
    CheckStatus,
    MetricScore,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    ValidatorResult,
)
from benchmark.validation.skip import is_exact_name_match
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
        check_table = []
        found_count = 0
        type_match_count = 0
        primitive_expected_count = 0
        primitive_match_count = 0
        available_objects = [obj for obj in snapshot.objects if obj.type.upper() == "MESH"]

        for expected_index, expected in enumerate(expected_objects):
            actual = self.matcher.match_expected_object(expected, available_objects)
            expected_path = f"expected_scene.objects[{expected_index}]"

            if actual is None:
                issue = self._missing_issue(expected, expected_path)
                issues.append(issue)
                object_label = expected.name or expected.type
                check_table.append(check_row(
                    validator_name=self.name,
                    check_name="object exists",
                    entity_ref=object_label,
                    field="object",
                    expected="exists",
                    actual="not_found",
                    passed=False,
                    score=0.0,
                    status=CheckStatus.FAIL,
                    weight=2.0,
                    issue=issue,
                    message=f"Required object {object_label} was not found",
                ))
                continue

            found_count += 1
            available_objects.remove(actual)
            actual_index = snapshot.objects.index(actual)
            actual_path = f"snapshot.objects[{actual_index}]"
            object_label = expected.name or actual.name
            exact_name = is_exact_name_match(expected, actual.name)
            existence_actual = actual.name if exact_name else f"found: {actual.name}"
            check_table.append(check_row(
                validator_name=self.name,
                check_name="object exists",
                entity_ref=object_label,
                field="object",
                expected="exists",
                actual=existence_actual,
                passed=True,
                score=1.0,
                status=CheckStatus.PASS,
                matched_object=None if exact_name else actual.name,
                match_reason=None if exact_name else "name_similarity",
            ))

            if self._type_matches(expected, actual):
                type_match_count += 1
                check_table.append(check_row(
                    validator_name=self.name,
                    check_name="object type",
                    entity_ref=expected.name or actual.name,
                    field="type",
                    expected=expected.type,
                    actual=actual.type,
                    passed=True,
                    score=1.0,
                ))
            else:
                issue = self._type_mismatch_issue(expected, actual, expected_path, actual_path)
                issues.append(issue)
                check_table.append(check_row(
                    validator_name=self.name,
                    check_name="object type",
                    entity_ref=expected.name or actual.name,
                    field="type",
                    expected=expected.type,
                    actual=actual.type,
                    passed=False,
                    score=0.0,
                    issue=issue,
                ))

            if expected.primitive is not None:
                primitive_expected_count += 1
                if self._primitive_matches(expected, actual):
                    primitive_match_count += 1
                    check_table.append(check_row(
                        validator_name=self.name,
                        check_name="primitive hint",
                        entity_ref=expected.name or actual.name,
                        field="primitive",
                        expected=expected.primitive,
                        actual=actual.primitive_hint,
                        passed=True,
                        score=1.0,
                    ))
                else:
                    issue = self._primitive_mismatch_issue(expected, actual, expected_path, actual_path)
                    issues.append(issue)
                    check_table.append(check_row(
                        validator_name=self.name,
                        check_name="primitive hint",
                        entity_ref=expected.name or actual.name,
                        field="primitive",
                        expected=expected.primitive,
                        actual=actual.primitive_hint,
                        passed=False,
                        score=0.0,
                        issue=issue,
                    ))

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
            check_table=check_table,
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
