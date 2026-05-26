from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ValidationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationIssue(BaseModel):
    code: str
    message: str
    severity: ValidationSeverity
    expected_path: str | None = None
    actual_path: str | None = None
    expected_value: Any | None = None
    actual_value: Any | None = None


class MetricScore(BaseModel):
    name: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    criterion_name: str | None = None
    checked_entity: str | None = None
    checked_field: str | None = None
    expected_path: str | None = None
    actual_path: str | None = None
    expected_value: Any | None = None
    actual_value: Any | None = None
    tolerance: float | None = None
    required: bool | None = None
    scoring_rule: str | None = None


class ValidationCheckRow(BaseModel):
    validator_name: str
    check_name: str
    entity_ref: str | None = None
    field: str | None = None
    expected: Any | None = None
    actual: Any | None = None
    tolerance: float | None = None
    passed: bool
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    issue_code: str | None = None
    severity: ValidationSeverity | None = None


class ValidatorResult(BaseModel):
    name: str
    status: ValidationStatus
    score: float = Field(ge=0.0, le=1.0)
    max_score: float = Field(default=1.0, ge=0.0, le=1.0)
    issues: list[ValidationIssue] = Field(default_factory=list)
    metrics: list[MetricScore] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    check_table: list[ValidationCheckRow] = Field(default_factory=list)


class SceneValidationResult(BaseModel):
    task_id: str
    overall_status: ValidationStatus
    total_score: float = Field(ge=0.0, le=1.0)
    validators: list[ValidatorResult] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    summary: dict[str, Any]
    check_table: list[ValidationCheckRow] = Field(default_factory=list)
