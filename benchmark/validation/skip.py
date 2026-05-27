"""Хелперы для skip-каскада зависимых проверок."""

from __future__ import annotations

from typing import Any

from benchmark.blender.models import SceneSnapshot
from benchmark.tasks.models import BenchmarkTask, ExpectedObject
from benchmark.validation.checks import check_row
from benchmark.validation.matcher import SceneMatcher, normalize_name
from benchmark.validation.models import CheckStatus, ValidationCheckRow, ValidationIssue


def skip_row(
    *,
    validator_name: str,
    check_name: str,
    entity_ref: str | None,
    field: str | None,
    expected: Any,
    issue: ValidationIssue,
    message: str | None = None,
) -> ValidationCheckRow:
    """Строка check_table со status=skip для зависимой проверки."""
    return check_row(
        validator_name=validator_name,
        check_name=check_name,
        entity_ref=entity_ref,
        field=field,
        expected=expected,
        actual="n/a",
        passed=False,
        score=None,
        status=CheckStatus.SKIP,
        issue=issue,
        message=message or issue.message,
    )


def object_missing_names(
    task: BenchmarkTask,
    snapshot: SceneSnapshot,
    matcher: SceneMatcher | None = None,
) -> set[str]:
    """Имена expected-объектов, не найденных в снимке сцены."""
    scene_matcher = matcher or SceneMatcher()
    available = [obj for obj in snapshot.objects if obj.type.upper() == "MESH"]
    missing: set[str] = set()
    for expected in task.expected_scene.objects:
        actual = scene_matcher.match_expected_object(expected, available)
        if actual is None:
            label = expected.name or expected.type
            if label:
                missing.add(label)
        else:
            available.remove(actual)
    return missing


def is_exact_name_match(expected: ExpectedObject, actual_name: str) -> bool:
    if not expected.name:
        return True
    return normalize_name(expected.name) == normalize_name(actual_name)
