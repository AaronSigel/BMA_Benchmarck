"""Validation of materials and object material assignments."""

from typing import Literal

from benchmark.blender.models import MaterialSnapshot, ObjectSnapshot, SceneSnapshot
from benchmark.tasks.models import BenchmarkTask, ColorRGBA, ExpectedMaterial, ExpectedObject
from benchmark.validation.matcher import SceneMatcher, name_similarity
from benchmark.validation.checks import check_row
from benchmark.validation.models import (
    MetricScore,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    ValidatorResult,
)
from benchmark.validation.skip import skip_row
from benchmark.validation.scoring import clamp_score, tolerance_score, weighted_average

MaterialParameter = Literal["base_color", "roughness", "metallic"]


class MaterialValidator:
    name = "material_validator"

    def __init__(self, matcher: SceneMatcher | None = None) -> None:
        self.matcher = matcher or SceneMatcher()

    def validate(self, task: BenchmarkTask, snapshot: SceneSnapshot) -> ValidatorResult:
        issues: list[ValidationIssue] = []
        metric_scores: list[MetricScore] = []
        check_table = []

        material_metrics = self._validate_expected_materials(task, snapshot, issues, check_table)
        assignment_metric = self._validate_object_material_assignments(task, snapshot, issues, check_table)

        if material_metrics[0] is not None:
            metric_scores.append(material_metrics[0])
        if material_metrics[1] is not None:
            metric_scores.append(material_metrics[1])
        if assignment_metric is not None:
            metric_scores.append(assignment_metric)

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
            check_table=check_table,
        )

    def _validate_expected_materials(
        self,
        task: BenchmarkTask,
        snapshot: SceneSnapshot,
        issues: list[ValidationIssue],
        check_table: list,
    ) -> tuple[MetricScore | None, MetricScore | None]:
        expected_materials = task.expected_scene.materials
        if not expected_materials:
            return None, None

        found_count = 0
        parameter_scores: list[float] = []
        existence_issues: list[ValidationIssue] = []
        parameter_issues: list[ValidationIssue] = []

        for expected_index, expected in enumerate(expected_materials):
            expected_path = f"expected_scene.materials[{expected_index}]"
            actual = self.matcher.match_expected_material(expected, snapshot.materials)
            if actual is None:
                issue = self._material_missing_issue(expected, expected_path)
                issues.append(issue)
                existence_issues.append(issue)
                check_table.append(check_row(
                    validator_name=self.name,
                    check_name="material exists",
                    entity_ref=expected.name,
                    field="material",
                    expected=expected.name,
                    actual=None,
                    passed=False,
                    score=0.0,
                    issue=issue,
                ))
                continue

            found_count += 1
            actual_index = snapshot.materials.index(actual)
            actual_path = f"snapshot.materials[{actual_index}]"
            check_table.append(check_row(
                validator_name=self.name,
                check_name="material exists",
                entity_ref=expected.name,
                field="material",
                expected=expected.name,
                actual=actual.name,
                passed=True,
                score=1.0,
            ))
            for parameter in self._expected_parameters(expected):
                parameter_score = self._parameter_score(expected, actual, parameter)
                parameter_scores.append(parameter_score)
                if parameter_score < 1.0:
                    issue = self._parameter_mismatch_issue(
                        expected,
                        actual,
                        parameter,
                        expected_path,
                        actual_path,
                    )
                    issues.append(issue)
                    parameter_issues.append(issue)
                else:
                    issue = None
                check_table.append(check_row(
                    validator_name=self.name,
                    check_name=parameter,
                    entity_ref=expected.name,
                    field=parameter,
                    expected=self._json_value(getattr(expected, parameter)),
                    actual=self._json_value(getattr(actual, parameter)),
                    tolerance=expected.tolerance,
                    passed=parameter_score == 1.0,
                    score=parameter_score,
                    issue=issue,
                ))

        existence_score = found_count / len(expected_materials)
        parameter_score = (
            sum(parameter_scores) / len(parameter_scores) if parameter_scores else 1.0
        )
        return (
            MetricScore(
                name="material_existence_score",
                score=existence_score,
                passed=existence_score == 1.0,
                issues=existence_issues,
            ),
            MetricScore(
                name="material_parameter_score",
                score=clamp_score(parameter_score),
                passed=parameter_score == 1.0,
                issues=parameter_issues,
            ),
        )

    def _validate_object_material_assignments(
        self,
        task: BenchmarkTask,
        snapshot: SceneSnapshot,
        issues: list[ValidationIssue],
        check_table: list,
    ) -> MetricScore | None:
        expected_objects = [
            (index, expected)
            for index, expected in enumerate(task.expected_scene.objects)
            if expected.material is not None
        ]
        if not expected_objects:
            return None

        assignment_scores: list[float] = []
        assignment_issues: list[ValidationIssue] = []
        available_objects = list(snapshot.objects)

        for expected_index, expected in expected_objects:
            expected_path = f"expected_scene.objects[{expected_index}]"
            actual = self.matcher.match_expected_object(expected, available_objects)
            if actual is None:
                issue = self._object_missing_issue(expected, expected_path)
                issues.append(issue)
                assignment_issues.append(issue)
                object_label = expected.name or expected.type
                check_table.append(skip_row(
                    validator_name=self.name,
                    check_name="assignment",
                    entity_ref=object_label,
                    field="material",
                    expected=expected.material,
                    issue=issue,
                    message=(
                        f"Skipped because required object {object_label} was not found"
                    ),
                ))
                continue

            available_objects.remove(actual)
            actual_index = snapshot.objects.index(actual)
            actual_path = f"snapshot.objects[{actual_index}]"

            if self._material_slot_matches(expected.material or "", actual):
                assignment_scores.append(1.0)
                check_table.append(check_row(
                    validator_name=self.name,
                    check_name="assignment",
                    entity_ref=expected.name or actual.name,
                    field="material",
                    expected=expected.material,
                    actual=list(actual.material_slots),
                    passed=True,
                    score=1.0,
                ))
            else:
                issue = self._object_material_missing_issue(expected, actual, expected_path, actual_path)
                issues.append(issue)
                assignment_issues.append(issue)
                assignment_scores.append(0.0)
                check_table.append(check_row(
                    validator_name=self.name,
                    check_name="assignment",
                    entity_ref=expected.name or actual.name,
                    field="material",
                    expected=expected.material,
                    actual=list(actual.material_slots),
                    passed=False,
                    score=0.0,
                    issue=issue,
                ))

        if not assignment_scores:
            return MetricScore(
                name="object_material_assignment_score",
                score=1.0,
                passed=True,
                issues=assignment_issues,
            )
        score = sum(assignment_scores) / len(assignment_scores)
        return MetricScore(
            name="object_material_assignment_score",
            score=score,
            passed=score == 1.0,
            issues=assignment_issues,
        )

    def _expected_parameters(self, expected: ExpectedMaterial) -> list[MaterialParameter]:
        parameters: list[MaterialParameter] = []
        if expected.base_color is not None:
            parameters.append("base_color")
        if expected.roughness is not None:
            parameters.append("roughness")
        if expected.metallic is not None:
            parameters.append("metallic")
        return parameters

    def _parameter_score(
        self,
        expected: ExpectedMaterial,
        actual: MaterialSnapshot,
        parameter: MaterialParameter,
    ) -> float:
        if parameter == "base_color":
            if expected.base_color is None or actual.base_color is None:
                return 0.0
            return self._color_score(expected.base_color, actual.base_color, expected.tolerance)

        expected_value = getattr(expected, parameter)
        actual_value = getattr(actual, parameter)
        if expected_value is None or actual_value is None:
            return 0.0
        return tolerance_score(expected_value, actual_value, expected.tolerance)

    def _color_score(self, expected: ColorRGBA, actual, tolerance: float) -> float:
        scores = [
            tolerance_score(expected.r, actual.r, tolerance),
            tolerance_score(expected.g, actual.g, tolerance),
            tolerance_score(expected.b, actual.b, tolerance),
            tolerance_score(expected.a, actual.a, tolerance),
        ]
        return sum(scores) / len(scores)

    def _material_slot_matches(self, expected_material_name: str, actual: ObjectSnapshot) -> bool:
        return any(
            name_similarity(expected_material_name, actual_material_name) > 0.0
            for actual_material_name in actual.material_slots
        )

    def _material_missing_issue(
        self,
        expected: ExpectedMaterial,
        expected_path: str,
    ) -> ValidationIssue:
        return ValidationIssue(
            code="material_missing",
            message=f"Expected material was not found: {expected.name}.",
            severity=ValidationSeverity.ERROR,
            expected_path=expected_path,
            actual_path=None,
            expected_value=expected.model_dump(mode="json", exclude_none=True),
            actual_value=None,
        )

    def _parameter_mismatch_issue(
        self,
        expected: ExpectedMaterial,
        actual: MaterialSnapshot,
        parameter: MaterialParameter,
        expected_path: str,
        actual_path: str,
    ) -> ValidationIssue:
        return ValidationIssue(
            code=f"material_{parameter.replace('base_', '')}_mismatch",
            message=f"Expected material {parameter} within tolerance {expected.tolerance}, got a different value.",
            severity=ValidationSeverity.ERROR,
            expected_path=f"{expected_path}.{parameter}",
            actual_path=f"{actual_path}.{parameter}",
            expected_value=self._json_value(getattr(expected, parameter)),
            actual_value=self._json_value(getattr(actual, parameter)),
        )

    def _object_missing_issue(
        self,
        expected: ExpectedObject,
        expected_path: str,
    ) -> ValidationIssue:
        return ValidationIssue(
            code="object_missing_for_material",
            message=f"Expected object was not found for material validation: {expected.name or expected.type}.",
            severity=ValidationSeverity.ERROR,
            expected_path=expected_path,
            actual_path=None,
            expected_value=expected.model_dump(mode="json", exclude_none=True),
            actual_value=None,
        )

    def _object_material_missing_issue(
        self,
        expected: ExpectedObject,
        actual: ObjectSnapshot,
        expected_path: str,
        actual_path: str,
    ) -> ValidationIssue:
        return ValidationIssue(
            code="object_material_missing",
            message=f"Expected material {expected.material!r} was not assigned to object {actual.name!r}.",
            severity=ValidationSeverity.ERROR,
            expected_path=f"{expected_path}.material",
            actual_path=f"{actual_path}.material_slots",
            expected_value=expected.material,
            actual_value=list(actual.material_slots),
        )

    def _json_value(self, value):
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value
