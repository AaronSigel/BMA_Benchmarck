"""Validation of expected lights."""

import math
from typing import Any

from benchmark.blender.models import LightSnapshot, SceneSnapshot, Vector3 as BlenderVector3
from benchmark.tasks.models import BenchmarkTask, ExpectedLight, Vector3
from benchmark.validation.matcher import SceneMatcher
from benchmark.validation.models import (
    MetricScore,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatus,
    ValidatorResult,
)
from benchmark.validation.scoring import tolerance_score, vector_tolerance_score, weighted_average


def _deg_to_rad_v3(v: Vector3) -> BlenderVector3:
    return BlenderVector3(x=math.radians(v.x), y=math.radians(v.y), z=math.radians(v.z))


def _euler_to_direction(rotation_euler: BlenderVector3) -> tuple[float, float, float]:
    """Convert Blender XYZ Euler angles (radians) to a normalized direction vector.

    Blender area/spot lights point in the -Z direction in their local frame.
    We rotate that vector by the Euler angles to get world-space direction.
    """
    rx, ry, rz = rotation_euler.x, rotation_euler.y, rotation_euler.z
    # Rotation matrices applied in XYZ order
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # Local -Z axis transformed by R_z * R_y * R_x
    dx = sy * cx * cz + sx * sz
    dy = sy * cx * sz - sx * cz
    dz = -cy * cx
    # -Z axis of the light
    dx = -sx * sy * cz - cx * sz
    dy = -sx * sy * sz + cx * cz
    dz = sx * cy
    # Actually use local +Z rotated: light default direction is (0,0,-1) in local
    lx = -(cy * sz)
    ly = cy * cz
    lz = -sy
    # Standard Blender: rotate (0,0,-1) by XYZ Euler
    local_z = (0.0, 0.0, -1.0)
    # Apply rx
    x1, y1, z1 = local_z
    y2 = y1 * cx - z1 * sx
    z2 = y1 * sx + z1 * cx
    x2 = x1
    # Apply ry
    x3 = x2 * cy + z2 * sy
    y3 = y2
    z3 = -x2 * sy + z2 * cy
    # Apply rz
    x4 = x3 * cz - y3 * sz
    y4 = x3 * sz + y3 * cz
    z4 = z3
    length = math.sqrt(x4 * x4 + y4 * y4 + z4 * z4)
    if length < 1e-9:
        return (0.0, 0.0, -1.0)
    return (x4 / length, y4 / length, z4 / length)


def _direction_to_target(
    location: BlenderVector3, target: Vector3
) -> tuple[float, float, float]:
    """Compute normalized direction from location toward target."""
    dx = target.x - location.x
    dy = target.y - location.y
    dz = target.z - location.z
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-9:
        return (0.0, 0.0, -1.0)
    return (dx / length, dy / length, dz / length)


def _angle_between_deg(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


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
        return (
            expected.location is not None
            or expected.rotation is not None
            or expected.target is not None
        )

    def _transform_score(self, expected: ExpectedLight, actual: LightSnapshot) -> float:
        scores: list[float] = []
        if expected.location is not None:
            scores.append(vector_tolerance_score(expected.location, actual.location, expected.tolerance))

        if expected.target is not None:
            # Direction-based check: compare actual light direction to expected target direction
            actual_dir = _euler_to_direction(actual.rotation_euler)
            if expected.location is not None:
                expected_dir = _direction_to_target(actual.location, expected.target)
            else:
                expected_dir = _direction_to_target(actual.location, expected.target)
            angle_deg = _angle_between_deg(actual_dir, expected_dir)
            tol = expected.direction_tolerance_deg
            dir_score = max(0.0, 1.0 - angle_deg / max(tol, 1.0))
            scores.append(dir_score)
        elif expected.rotation is not None:
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

        if expected.target is not None:
            actual_dir = _euler_to_direction(actual.rotation_euler)
            expected_dir = _direction_to_target(actual.location, expected.target)
            angle_deg = _angle_between_deg(actual_dir, expected_dir)
            tol = expected.direction_tolerance_deg
            if angle_deg > tol:
                issues.append(ValidationIssue(
                    code="light_direction_mismatch",
                    message=(
                        f"Light direction deviates {angle_deg:.1f}° from target "
                        f"(tolerance {tol}°)."
                    ),
                    severity=ValidationSeverity.ERROR,
                    expected_path=f"{expected_path}.target",
                    actual_path=f"{actual_path}.rotation_euler",
                    expected_value={"target": expected.target.model_dump(mode="json"), "tolerance_deg": tol},
                    actual_value={"direction": list(actual_dir), "angle_deg": round(angle_deg, 2)},
                ))
        elif expected.rotation is not None:
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
        # Use relative tolerance: tolerance represents the allowed fraction of expected energy.
        # e.g. tolerance=0.10 → ±10% of expected_energy.
        relative_band = abs(expected.energy) * expected.tolerance
        return tolerance_score(expected.energy, actual.energy, max(relative_band, 1.0))

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
        pct = round(expected.tolerance * 100)
        return ValidationIssue(
            code="light_energy_mismatch",
            message=(
                f"Expected light energy {expected.energy}W within {pct}% tolerance "
                f"(±{abs(expected.energy or 0) * expected.tolerance:.1f}W), "
                f"got {actual.energy}W."
            ),
            severity=ValidationSeverity.ERROR,
            expected_path=f"{expected_path}.energy",
            actual_path=f"{actual_path}.energy",
            expected_value=expected.energy,
            actual_value=actual.energy,
        )
