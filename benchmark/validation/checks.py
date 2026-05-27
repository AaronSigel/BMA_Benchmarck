from __future__ import annotations

from typing import Any

from benchmark.validation.models import CheckStatus, ValidationCheckRow, ValidationIssue


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
    status: CheckStatus | None = None,
    weight: float | None = None,
    message: str | None = None,
    matched_object: str | None = None,
    match_reason: str | None = None,
    issue: ValidationIssue | None = None,
) -> ValidationCheckRow:
    resolved_status = status
    if resolved_status is None:
        resolved_status = CheckStatus.PASS if passed else CheckStatus.FAIL
    resolved_score = score
    if resolved_status is CheckStatus.SKIP:
        resolved_score = None
    return ValidationCheckRow(
        validator_name=validator_name,
        check_name=check_name,
        entity_ref=entity_ref,
        field=field,
        expected=json_value(expected),
        actual=json_value(actual),
        tolerance=tolerance,
        passed=passed,
        score=resolved_score,
        status=resolved_status,
        weight=weight,
        message=message or (issue.message if issue is not None else None),
        matched_object=matched_object,
        match_reason=match_reason,
        issue_code=issue.code if issue is not None else None,
        severity=issue.severity if issue is not None else None,
    )
