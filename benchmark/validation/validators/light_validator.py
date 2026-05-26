"""Validation of expected lights."""

import math
from typing import Any

from benchmark.blender.models import LightSnapshot, SceneSnapshot, Vector3 as BlenderVector3
from benchmark.tasks.models import BenchmarkTask, ExpectedLight, ExpectedScene, Vector3
from benchmark.validation.matcher import SceneMatcher
from benchmark.validation.checks import check_row, json_value
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
    """Convert Blender XYZ Euler angles (radians) to a normalized direction vector."""
    rx, ry, rz = rotation_euler.x, rotation_euler.y, rotation_euler.z
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    x1, y1, z1 = 0.0, 0.0, -1.0
    y2 = y1 * cx - z1 * sx
    z2 = y1 * sx + z1 * cx
    x2 = x1
    x3 = x2 * cy + z2 * sy
    y3 = y2
    z3 = -x2 * sy + z2 * cy
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


def _prefer_direction_for_rotation(expected: ExpectedLight) -> bool:
    return expected.type.upper() in {"SUN", "AREA", "SPOT"}


_DIRECTION_GRACE_DEG = 5.0


def _resolve_target(target: Vector3 | str | None, snapshot: SceneSnapshot) -> Vector3 | None:
    if target is None:
        return None
    if isinstance(target, str):
        from benchmark.tasks.models import ExpectedObject

        obj = SceneMatcher().match_expected_object(
            ExpectedObject(name=target, type="MESH"),
            snapshot.objects,
        )
        if obj is None:
            return None
        return Vector3(x=obj.location.x, y=obj.location.y, z=obj.location.z)
    return target


def _scene_center_target(expected_scene: ExpectedScene) -> Vector3 | None:
    locations = [obj.location for obj in expected_scene.objects if obj.location is not None]
    if not locations:
        return None
    count = len(locations)
    return Vector3(
        x=sum(loc.x for loc in locations) / count,
        y=sum(loc.y for loc in locations) / count,
        z=sum(loc.z for loc in locations) / count,
    )


def _resolve_light_target(
    expected: ExpectedLight,
    expected_scene: ExpectedScene,
    snapshot: SceneSnapshot,
) -> Vector3 | None:
    if expected.target is not None:
        return _resolve_target(expected.target, snapshot)
    return _scene_center_target(expected_scene)


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
        check_table = []
        available_lights = list(snapshot.lights)

        for expected_index, expected in enumerate(expected_lights):
            expected_path = f"expected_scene.lights[{expected_index}]"
            actual = self.matcher.match_expected_light(expected, available_lights)
            if actual is None:
                issue = self._missing_issue(expected, expected_path)
                issues.append(issue)
                check_table.append(check_row(
                    validator_name=self.name,
                    check_name="light exists",
                    entity_ref=expected.name or expected.type,
                    field="light",
                    expected=expected.model_dump(mode="json", exclude_none=True),
                    actual=None,
                    passed=False,
                    score=0.0,
                    issue=issue,
                ))
                existence_scores.append(0.0)
                type_scores.append(0.0)
                transform_scores.append(0.0 if self._has_transform_expectations(expected) else 1.0)
                energy_scores.append(0.0 if expected.energy is not None else 1.0)
                continue

            available_lights.remove(actual)
            actual_index = snapshot.lights.index(actual)
            actual_path = f"snapshot.lights[{actual_index}]"
            existence_scores.append(1.0)
            check_table.append(check_row(
                validator_name=self.name,
                check_name="light exists",
                entity_ref=expected.name or actual.name,
                field="light",
                expected=expected.name or expected.type,
                actual=actual.name,
                passed=True,
                score=1.0,
            ))

            type_score = 1.0 if expected.type.upper() == actual.type.upper() else 0.0
            type_scores.append(type_score)
            if type_score < 1.0:
                issue = self._type_mismatch_issue(expected, actual, expected_path, actual_path)
                issues.append(issue)
            else:
                issue = None
            check_table.append(check_row(
                validator_name=self.name,
                check_name="type",
                entity_ref=expected.name or actual.name,
                field="type",
                expected=expected.type,
                actual=actual.type,
                passed=type_score == 1.0,
                score=type_score,
                issue=issue,
            ))

            transform_score = self._transform_score(expected, actual, task.expected_scene, snapshot)
            transform_scores.append(transform_score)
            before = len(issues)
            self._append_transform_issues(
                expected,
                actual,
                expected_path,
                actual_path,
                issues,
                task.expected_scene,
                snapshot,
            )
            transform_issue_by_field = {
                str(issue.expected_path).rsplit(".", 1)[-1]: issue
                for issue in issues[before:]
            }
            for field in ("location", "rotation"):
                if getattr(expected, field) is None:
                    continue
                field_score = (
                    vector_tolerance_score(expected.location, actual.location, expected.tolerance)
                    if field == "location"
                    else vector_tolerance_score(_deg_to_rad_v3(expected.rotation), actual.rotation_euler, expected.tolerance)
                )
                check_table.append(check_row(
                    validator_name=self.name,
                    check_name=field,
                    entity_ref=expected.name or actual.name,
                    field=field,
                    expected=json_value(_deg_to_rad_v3(expected.rotation) if field == "rotation" else expected.location),
                    actual=json_value(actual.rotation_euler if field == "rotation" else actual.location),
                    tolerance=expected.tolerance,
                    passed=field_score == 1.0,
                    score=field_score,
                    issue=transform_issue_by_field.get(field),
                ))
            if expected.target is not None:
                dir_score, actual_dir, expected_dir, angle_deg = self._direction_score(
                    expected, actual, task.expected_scene, snapshot
                )
                check_table.append(check_row(
                    validator_name=self.name,
                    check_name="target direction",
                    entity_ref=expected.name or actual.name,
                    field="target",
                    expected={"direction": list(expected_dir), "target": json_value(expected.target)},
                    actual={"direction": list(actual_dir), "angle_deg": round(angle_deg, 3)},
                    tolerance=expected.direction_tolerance_deg,
                    passed=dir_score == 1.0,
                    score=dir_score,
                    issue=transform_issue_by_field.get("target") or transform_issue_by_field.get("rotation"),
                ))

            energy_score = self._energy_score(expected, actual)
            energy_scores.append(energy_score)
            if energy_score < 1.0:
                issue = self._energy_mismatch_issue(expected, actual, expected_path, actual_path)
                issues.append(issue)
            else:
                issue = None
            if expected.energy is not None:
                check_table.append(check_row(
                    validator_name=self.name,
                    check_name="energy",
                    entity_ref=expected.name or actual.name,
                    field="energy",
                    expected=expected.energy,
                    actual=actual.energy,
                    tolerance=max(abs(expected.energy) * expected.tolerance, 1.0),
                    passed=energy_score == 1.0,
                    score=energy_score,
                    issue=issue,
                ))

        metrics = [
            self._metric("light_existence_score", existence_scores, 0.4),
            self._metric("light_type_score", type_scores, 0.2),
            self._metric("light_transform_score", transform_scores, 0.2),
            self._metric("light_energy_score", energy_scores, 0.2),
        ]
        score = weighted_average([(metric.score, metric.weight) for metric in metrics])
        blocking_issues = [issue for issue in issues if issue.severity == ValidationSeverity.ERROR]
        status = ValidationStatus.PASSED if not blocking_issues else ValidationStatus.FAILED
        return ValidatorResult(
            name=self.name,
            status=status,
            score=score,
            issues=issues,
            metrics=metrics,
            check_table=check_table,
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

    def _direction_score(
        self,
        expected: ExpectedLight,
        actual: LightSnapshot,
        expected_scene: ExpectedScene,
        snapshot: SceneSnapshot,
    ) -> tuple[float, tuple[float, float, float], tuple[float, float, float], float]:
        actual_dir = _euler_to_direction(actual.rotation_euler)
        tol = expected.direction_tolerance_deg

        if expected.target is not None:
            target_point = _resolve_light_target(expected, expected_scene, snapshot)
            if target_point is not None:
                expected_dir = _direction_to_target(actual.location, target_point)
                angle_deg = _angle_between_deg(actual_dir, expected_dir)
                dir_score = 1.0 if angle_deg <= tol else max(0.0, 1.0 - (angle_deg - tol) / max(tol, 1.0))
                return dir_score, actual_dir, expected_dir, angle_deg

        if expected.rotation is not None and _prefer_direction_for_rotation(expected):
            expected_dir = _euler_to_direction(_deg_to_rad_v3(expected.rotation))
            angle_deg = _angle_between_deg(actual_dir, expected_dir)
            dir_score = 1.0 if angle_deg <= tol else max(0.0, 1.0 - (angle_deg - tol) / max(tol, 1.0))
            return dir_score, actual_dir, expected_dir, angle_deg

        return 1.0, actual_dir, (0.0, 0.0, -1.0), 0.0

    def _transform_score(
        self,
        expected: ExpectedLight,
        actual: LightSnapshot,
        expected_scene: ExpectedScene,
        snapshot: SceneSnapshot,
    ) -> float:
        scores: list[float] = []
        if expected.location is not None:
            scores.append(vector_tolerance_score(expected.location, actual.location, expected.tolerance))

        if expected.target is not None or (
            expected.rotation is not None and _prefer_direction_for_rotation(expected)
        ):
            dir_score, _, _, _ = self._direction_score(expected, actual, expected_scene, snapshot)
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
        expected_scene: ExpectedScene,
        snapshot: SceneSnapshot,
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

        uses_direction = expected.target is not None or (
            expected.rotation is not None and _prefer_direction_for_rotation(expected)
        )
        direction_ok = True
        within_grace = True
        if uses_direction:
            _, actual_dir, expected_dir, angle_deg = self._direction_score(
                expected, actual, expected_scene, snapshot
            )
            tol = expected.direction_tolerance_deg
            grace_tol = tol + _DIRECTION_GRACE_DEG
            direction_ok = angle_deg <= tol
            within_grace = angle_deg <= grace_tol
            if not direction_ok:
                target_point = _resolve_light_target(expected, expected_scene, snapshot)
                severity = (
                    ValidationSeverity.WARNING
                    if within_grace
                    else ValidationSeverity.ERROR
                )
                issues.append(
                    ValidationIssue(
                        code="light_direction_mismatch",
                        message=(
                            f"Light direction deviates {angle_deg:.1f}° from expected "
                            f"(tolerance {tol}°)."
                        ),
                        severity=severity,
                        expected_path=(
                            f"{expected_path}.target"
                            if expected.target is not None
                            else f"{expected_path}.rotation"
                        ),
                        actual_path=f"{actual_path}.rotation_euler",
                        expected_value={
                            "direction": list(expected_dir),
                            "target": target_point.model_dump(mode="json") if target_point else None,
                            "tolerance_deg": tol,
                        },
                        actual_value={"direction": list(actual_dir), "angle_deg": round(angle_deg, 2)},
                    )
                )

        if expected.rotation is not None and _prefer_direction_for_rotation(expected) and (
            direction_ok or within_grace
        ):
            euler_score = vector_tolerance_score(
                _deg_to_rad_v3(expected.rotation), actual.rotation_euler, expected.tolerance
            )
            if euler_score < 1.0:
                issues.append(
                    self._vector_mismatch_issue(
                        code="light_rotation_mismatch",
                        field="rotation",
                        expected_value=_deg_to_rad_v3(expected.rotation),
                        actual_value=actual.rotation_euler,
                        tolerance=expected.tolerance,
                        expected_path=expected_path,
                        actual_path=actual_path,
                        severity=ValidationSeverity.WARNING,
                    )
                )
        elif expected.rotation is not None and not _prefer_direction_for_rotation(expected):
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
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> ValidationIssue:
        actual_field = "rotation_euler" if field == "rotation" else field
        return ValidationIssue(
            code=code,
            message=f"Expected light {field} within tolerance {tolerance}, got a different value.",
            severity=severity,
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
