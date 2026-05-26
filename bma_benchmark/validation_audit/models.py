from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ValidatorCheckedField(BaseModel):
    validator_name: str
    checked_entity: str
    checked_field: str
    criterion_name: str
    scoring_rule: str = "exact_or_tolerance"
    default_tolerance: float | None = None


class ValidatorLimitation(BaseModel):
    validator_name: str
    limitation: str


class ValidatorAuditRow(BaseModel):
    task_id: str
    category: str
    validator_name: str
    criterion_name: str
    checked_entity: str
    checked_field: str
    expected_path: str
    actual_path: str
    tolerance: float | None = None
    weight: float | None = None
    required: bool | None = None
    scoring_rule: str
    pass_threshold: float | None = None
    warning_threshold: float | None = None
    issue_codes: list[str] = Field(default_factory=list)
    limitation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidatorAuditReport(BaseModel):
    tasks_dir: str
    task_count: int
    rows: list[ValidatorAuditRow] = Field(default_factory=list)
    limitations: list[ValidatorLimitation] = Field(default_factory=list)
