from typing import Any

from pydantic import BaseModel, Field

MetricValue = float | str | int | bool


class RunMetric(BaseModel):
    run_id: str
    task_id: str
    name: str
    value: MetricValue
    group: str
    source: str


class RunMetricRow(BaseModel):
    run_id: str
    task_id: str
    total_score: float | None = Field(default=None, ge=0.0, le=1.0)
    overall_status: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class MetricSummary(BaseModel):
    total_runs: int = Field(ge=0)
    average_score: float = Field(ge=0.0, le=1.0)
    passed_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    error_runs: int = Field(ge=0)


class MetricsSummary(BaseModel):
    total_runs: int = Field(ge=0)
    attempted_runs: int = Field(default=0, ge=0)
    completed_runs: int = Field(default=0, ge=0)
    validated_runs: int = Field(default=0, ge=0)
    passed_runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    error_runs: int = Field(ge=0)
    average_score: float | None = Field(default=None, ge=0.0, le=1.0)
    average_score_on_validated_runs: float | None = Field(default=None, ge=0.0, le=1.0)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    max_score: float | None = Field(default=None, ge=0.0, le=1.0)
    success_rate_on_all_attempted_runs: float | None = Field(default=None, ge=0.0, le=1.0)
    metrics: list[RunMetric] = Field(default_factory=list)
