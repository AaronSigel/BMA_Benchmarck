from __future__ import annotations

from typing import Any

from benchmark.validation.models import ValidationCheckRow, ValidationIssue


def json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return list(value)
    return value


def check_row(
    *,
    validator_name: str,
    check_name: str,
    entity_ref: str | None = None,
    field: str | None = None,
    expected: Any | None = None,
    actual: Any | None = None,
    tolerance: float | None = None,
    passed: bool,
    score: float | None = None,
    issue: ValidationIssue | None = None,
) -> ValidationCheckRow:
    return ValidationCheckRow(
        validator_name=validator_name,
        check_name=check_name,
        entity_ref=entity_ref,
        field=field,
        expected=json_value(expected),
        actual=json_value(actual),
        tolerance=tolerance,
        passed=passed,
        score=score,
        issue_code=issue.code if issue is not None else None,
        severity=issue.severity if issue is not None else None,
    )
