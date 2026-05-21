"""Validation of expected cameras."""

import math

from benchmark.blender.models import CameraSnapshot, SceneSnapshot, Vector3 as BlenderVector3
from benchmark.tasks.models import BenchmarkTask, ExpectedCamera, Vector3


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


class CameraValidator:
    name = "camera_validator"

    def __init__(self, matcher: SceneMatcher | None = None) -> None:
        self.matcher = matcher or SceneMatcher()

    def validate(self, task: BenchmarkTask, snapshot: SceneSnapshot) -> ValidatorResult:
        expected_cameras = task.expected_scene.cameras
        if not expected_cameras:
            return ValidatorResult(
                name=self.name,
                status=ValidationStatus.SKIPPED,
                score=0.0,
                metrics=[],
            )

        issues: list[ValidationIssue] = []
        existence_scores: list[float] = []
        transform_scores: list[float] = []
        direction_scores: list[float] = []
        focal_length_scores: list[float] = []
        active_scores: list[float] = []
        available_cameras = list(snapshot.cameras)

        for expected_index, expected in enumerate(expected_cameras):
            require_active = expected.require_active if expected.require_active is not None else len(expected_cameras) == 1
            expected_path = f"expected_scene.cameras[{expected_index}]"
            actual = self.matcher.match_expected_camera(expected, available_cameras)
            if actual is None:
                issue = self._missing_issue(expected, expected_path)
                issues.append(issue)
                existence_scores.append(0.0)
                transform_scores.append(0.0 if self._has_transform_expectations(expected) else 1.0)
                direction_scores.append(0.0 if expected.target is not None else 1.0)
                focal_length_scores.append(0.0 if expected.focal_length is not None else 1.0)
                active_scores.append(0.0 if require_active else 1.0)
                continue

            available_cameras.remove(actual)
            actual_index = snapshot.cameras.index(actual)
            actual_path = f"snapshot.cameras[{actual_index}]"
            existence_scores.append(1.0)

            transform_score = self._transform_score(expected, actual)
            transform_scores.append(transform_score)
            self._append_transform_issues(expected, actual, expected_path, actual_path, issues)

            direction_score = self._direction_score(expected, actual)
            direction_scores.append(direction_score)
            if direction_score < 1.0:
                issues.append(self._direction_mismatch_issue(expected, actual, expected_path, actual_path))

            focal_length_score = self._focal_length_score(expected, actual)
            focal_length_scores.append(focal_length_score)
            if focal_length_score < 1.0:
                issues.append(
                    self._focal_length_mismatch_issue(expected, actual, expected_path, actual_path)
                )

            active_score = 1.0
            if require_active and not actual.is_active:
                active_score = 0.0
                issues.append(self._active_camera_mismatch_issue(expected, actual, expected_path, actual_path))
            active_scores.append(active_score)

        metrics = [
            self._metric("camera_existence_score", existence_scores, 0.3),
            self._metric("camera_transform_score", transform_scores, 0.25),
            self._metric("camera_direction_score", direction_scores, 0.15),
            self._metric("camera_focal_length_score", focal_length_scores, 0.2),
            self._metric("active_camera_score", active_scores, 0.1),
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

    def _has_transform_expectations(self, expected: ExpectedCamera) -> bool:
        return expected.location is not None or expected.rotation is not None or expected.target is not None

    def _transform_score(self, expected: ExpectedCamera, actual: CameraSnapshot) -> float:
        scores: list[float] = []
        if expected.location is not None:
            scores.append(vector_tolerance_score(expected.location, actual.location, expected.tolerance))
        if expected.rotation is not None and expected.target is None:
            scores.append(
                vector_tolerance_score(
                    _deg_to_rad_v3(expected.rotation), actual.rotation_euler, expected.tolerance
                )
            )
        return sum(scores) / len(scores) if scores else 1.0

    def _append_transform_issues(
        self,
        expected: ExpectedCamera,
        actual: CameraSnapshot,
        expected_path: str,
        actual_path: str,
        issues: list[ValidationIssue],
    ) -> None:
        if expected.location is not None:
            score = vector_tolerance_score(expected.location, actual.location, expected.tolerance)
            if score < 1.0:
                issues.append(
                    self._vector_mismatch_issue(
                        code="camera_location_mismatch",
                        field="location",
                        expected_value=expected.location,
                        actual_value=actual.location,
                        tolerance=expected.tolerance,
                        expected_path=expected_path,
                        actual_path=actual_path,
                    )
                )

        if expected.rotation is not None and expected.target is None:
            score = vector_tolerance_score(
                _deg_to_rad_v3(expected.rotation), actual.rotation_euler, expected.tolerance
            )
            if score < 1.0:
                issues.append(
                    self._vector_mismatch_issue(
                        code="camera_rotation_mismatch",
                        field="rotation",
                        expected_value=_deg_to_rad_v3(expected.rotation),
                        actual_value=actual.rotation_euler,
                        tolerance=expected.tolerance,
                        expected_path=expected_path,
                        actual_path=actual_path,
                    )
                )

    def _direction_score(self, expected: ExpectedCamera, actual: CameraSnapshot) -> float:
        if expected.target is None:
            return 1.0
        deviation = _angular_deviation_deg(_camera_forward(actual.rotation_euler), _target_direction(actual, expected.target))
        if deviation <= expected.direction_tolerance_deg:
            return 1.0
        if expected.direction_tolerance_deg <= 0.0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - ((deviation - expected.direction_tolerance_deg) / expected.direction_tolerance_deg)))

    def _focal_length_score(self, expected: ExpectedCamera, actual: CameraSnapshot) -> float:
        if expected.focal_length is None:
            return 1.0
        if actual.lens is None:
            return 0.0
        return tolerance_score(expected.focal_length, actual.lens, expected.tolerance)

    def _missing_issue(self, expected: ExpectedCamera, expected_path: str) -> ValidationIssue:
        return ValidationIssue(
            code="camera_missing",
            message=f"Expected camera was not found: {expected.name or 'camera'}.",
            severity=ValidationSeverity.ERROR,
            expected_path=expected_path,
            actual_path=None,
            expected_value=expected.model_dump(mode="json", exclude_none=True),
            actual_value=None,
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
            message=f"Expected camera {field} within tolerance {tolerance}, got a different value.",
            severity=ValidationSeverity.ERROR,
            expected_path=f"{expected_path}.{field}",
            actual_path=f"{actual_path}.{actual_field}",
            expected_value=expected_value.model_dump(mode="json"),
            actual_value=actual_value.model_dump(mode="json"),
        )

    def _focal_length_mismatch_issue(
        self,
        expected: ExpectedCamera,
        actual: CameraSnapshot,
        expected_path: str,
        actual_path: str,
    ) -> ValidationIssue:
        return ValidationIssue(
            code="camera_focal_length_mismatch",
            message=f"Expected camera focal_length within tolerance {expected.tolerance}, got a different lens.",
            severity=ValidationSeverity.ERROR,
            expected_path=f"{expected_path}.focal_length",
            actual_path=f"{actual_path}.lens",
            expected_value=expected.focal_length,
            actual_value=actual.lens,
        )

    def _direction_mismatch_issue(
        self,
        expected: ExpectedCamera,
        actual: CameraSnapshot,
        expected_path: str,
        actual_path: str,
    ) -> ValidationIssue:
        deviation = _angular_deviation_deg(
            _camera_forward(actual.rotation_euler),
            _target_direction(actual, expected.target),
        )
        return ValidationIssue(
            code="camera_direction_mismatch",
            message=(
                "Expected camera to look at target within "
                f"{expected.direction_tolerance_deg} degrees, got {deviation:.3f} degrees."
            ),
            severity=ValidationSeverity.ERROR,
            expected_path=f"{expected_path}.target",
            actual_path=f"{actual_path}.rotation_euler",
            expected_value=expected.target.model_dump(mode="json") if expected.target is not None else None,
            actual_value={
                "rotation_euler": actual.rotation_euler.model_dump(mode="json"),
                "angular_deviation_deg": deviation,
            },
        )

    def _active_camera_mismatch_issue(
        self,
        expected: ExpectedCamera,
        actual: CameraSnapshot,
        expected_path: str,
        actual_path: str,
    ) -> ValidationIssue:
        return ValidationIssue(
            code="active_camera_mismatch",
            message=f"Expected the only camera candidate to be active, but {actual.name!r} is not active.",
            severity=ValidationSeverity.ERROR,
            expected_path=expected_path,
            actual_path=f"{actual_path}.is_active",
            expected_value=True,
            actual_value=actual.is_active,
        )


def _target_direction(actual: CameraSnapshot, target: Vector3 | None) -> tuple[float, float, float]:
    if target is None:
        return (0.0, 0.0, -1.0)
    return _normalize((
        target.x - actual.location.x,
        target.y - actual.location.y,
        target.z - actual.location.z,
    ))


def _camera_forward(rotation: BlenderVector3) -> tuple[float, float, float]:
    # Blender cameras look along local -Z. Default Euler order is XYZ.
    cx, sx = math.cos(rotation.x), math.sin(rotation.x)
    cy, sy = math.cos(rotation.y), math.sin(rotation.y)
    cz, sz = math.cos(rotation.z), math.sin(rotation.z)
    x, y, z = 0.0, 0.0, -1.0
    y, z = y * cx - z * sx, y * sx + z * cx
    x, z = x * cy + z * sy, -x * sy + z * cy
    x, y = x * cz - y * sz, x * sz + y * cz
    return _normalize((x, y, z))


def _angular_deviation_deg(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    a = _normalize(a)
    b = _normalize(b)
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    return math.degrees(math.acos(dot))


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(part * part for part in v))
    if length <= 0.0:
        return (0.0, 0.0, 0.0)
    return tuple(part / length for part in v)  # type: ignore[return-value]
