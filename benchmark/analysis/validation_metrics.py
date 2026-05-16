from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from benchmark.analysis.models import ValidationMetric
from benchmark.analysis.trace_reader import read_validation_result
from benchmark.validation.models import SceneValidationResult, ValidationStatus

# Map from validator name → metric field name suffix
_VALIDATOR_SCORE_MAP: dict[str, str] = {
    "object_validator": "object_score",
    "transform_validator": "transform_score",
    "material_validator": "material_score",
    "light_validator": "light_score",
    "camera_validator": "camera_score",
    "export_validator": "export_score",
}

_UNKNOWN_STATUS = "unknown"


# ---------------------------------------------------------------------------
# Summary model
# ---------------------------------------------------------------------------


class ValidationMetricsSummary(BaseModel):
    """All validation metrics derived from a SceneValidationResult."""

    scene_total_score: float | None = Field(default=None, ge=0.0, le=1.0)
    scene_overall_status: str = _UNKNOWN_STATUS

    # Per-validator scores (None when validator absent or skipped)
    object_score: float | None = Field(default=None, ge=0.0, le=1.0)
    transform_score: float | None = Field(default=None, ge=0.0, le=1.0)
    material_score: float | None = Field(default=None, ge=0.0, le=1.0)
    light_score: float | None = Field(default=None, ge=0.0, le=1.0)
    camera_score: float | None = Field(default=None, ge=0.0, le=1.0)
    export_score: float | None = Field(default=None, ge=0.0, le=1.0)

    # Issue counts (across top-level + all validators)
    validation_error_count: int = Field(default=0, ge=0)
    validation_warning_count: int = Field(default=0, ge=0)

    # Validator status counts
    passed_validator_count: int = Field(default=0, ge=0)
    failed_validator_count: int = Field(default=0, ge=0)
    skipped_validator_count: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_validation_summary(val: SceneValidationResult | None) -> ValidationMetricsSummary:
    """Compute all validation metrics from a SceneValidationResult.

    When *val* is None, returns a summary with status='unknown' and zero counts.
    Skipped validators are excluded from per-validator scores (they don't affect
    the average) but are counted in *skipped_validator_count*.
    """
    if val is None:
        return ValidationMetricsSummary()

    # Overall
    scene_total_score = val.total_score
    scene_overall_status = val.overall_status.value

    # Per-validator scores — skipped validators get None
    scores: dict[str, float | None] = {k: None for k in _VALIDATOR_SCORE_MAP.values()}
    passed = failed = skipped = 0

    for v in val.validators:
        field_name = _VALIDATOR_SCORE_MAP.get(v.name)
        if v.status == ValidationStatus.SKIPPED:
            skipped += 1
            # Leave score as None — does not affect average
        else:
            if v.status == ValidationStatus.PASSED:
                passed += 1
            else:
                failed += 1
            if field_name is not None:
                scores[field_name] = v.score

    # Issue counts across top-level issues + all validator issues
    error_count = 0
    warning_count = 0
    for issue in val.issues:
        if issue.severity.value == "error":
            error_count += 1
        elif issue.severity.value == "warning":
            warning_count += 1
    for v in val.validators:
        for issue in v.issues:
            if issue.severity.value == "error":
                error_count += 1
            elif issue.severity.value == "warning":
                warning_count += 1

    return ValidationMetricsSummary(
        scene_total_score=scene_total_score,
        scene_overall_status=scene_overall_status,
        object_score=scores["object_score"],
        transform_score=scores["transform_score"],
        material_score=scores["material_score"],
        light_score=scores["light_score"],
        camera_score=scores["camera_score"],
        export_score=scores["export_score"],
        validation_error_count=error_count,
        validation_warning_count=warning_count,
        passed_validator_count=passed,
        failed_validator_count=failed,
        skipped_validator_count=skipped,
    )


# ---------------------------------------------------------------------------
# Lower-level helpers (used by report_builder)
# ---------------------------------------------------------------------------


def extract_validation_metrics(val: SceneValidationResult) -> list[ValidationMetric]:
    """Return per-validator ValidationMetric list from a SceneValidationResult."""
    return [
        ValidationMetric(
            validator_name=v.name,
            score=v.score,
            status=v.status.value,
            issue_count=len(v.issues),
        )
        for v in val.validators
    ]


def extract_score_and_status(val: SceneValidationResult) -> tuple[float | None, str | None]:
    """Return (total_score, overall_status) from a SceneValidationResult."""
    return val.total_score, val.overall_status.value


def extract_issues(val: SceneValidationResult) -> list[dict[str, Any]]:
    """Collect all issues (top-level + per-validator) from a SceneValidationResult."""
    issues: list[dict[str, Any]] = [i.model_dump() for i in val.issues]
    for v in val.validators:
        for issue in v.issues:
            issues.append({**issue.model_dump(), "validator": v.name})
    return issues


def load_validation_result(path: Path | str) -> SceneValidationResult:
    return read_validation_result(path)
