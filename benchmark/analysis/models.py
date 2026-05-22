from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

MetricValue = float | str | int | bool


class ErrorCategory(str, Enum):
    TOOL_DISABLED = "tool_disabled"
    TOOL_UNKNOWN = "tool_unknown"
    TOOL_INVALID_ARGUMENTS = "tool_invalid_arguments"
    TOOL_RUNTIME_ERROR = "tool_runtime_error"
    LLM_PARSE_ERROR = "llm_parse_error"
    LLM_TIMEOUT = "llm_timeout"
    AGENT_STEP_LIMIT = "agent_step_limit"
    SCENE_OBJECT_MISSING = "scene_object_missing"
    SCENE_TRANSFORM_MISMATCH = "scene_transform_mismatch"
    SCENE_MATERIAL_MISMATCH = "scene_material_mismatch"
    SCENE_LIGHT_MISMATCH = "scene_light_mismatch"
    SCENE_CAMERA_MISMATCH = "scene_camera_mismatch"
    SCENE_EXPORT_MISSING = "scene_export_missing"
    MCP_CONNECTION_ERROR = "mcp_connection_error"
    REMOTE_AGENT_ERROR = "remote_agent_error"
    UNKNOWN_ERROR = "unknown_error"


class ComparisonDimension(str, Enum):
    STRATEGY = "strategy"
    MODEL = "model"
    MCP_PROFILE = "mcp_profile"
    RUN = "run"
    AGENT_ID = "agent_id"
    TASK_CATEGORY = "task_category"
    DIFFICULTY = "difficulty"
    REMOTE_PROVIDER = "remote_provider"


class ToolCallMetric(BaseModel):
    """Aggregated per-tool statistics extracted from a single AgentTrace."""

    tool_name: str
    total_calls: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    success_rate: float = Field(ge=0.0, le=1.0)
    avg_duration_sec: float | None = Field(default=None, ge=0.0)
    total_duration_sec: float = Field(default=0.0, ge=0.0)


class AgentMetric(BaseModel):
    """A single named measurement associated with an agent run."""

    name: str
    value: MetricValue
    group: str = "agent"
    source: str = "trace"


class ValidationMetric(BaseModel):
    """Per-validator score extracted from a validation_result.json."""

    validator_name: str
    score: float = Field(ge=0.0, le=1.0)
    status: str
    issue_count: int = Field(default=0, ge=0)


class ErrorRecord(BaseModel):
    """A single classified error from an agent trace step."""

    run_id: str
    task_id: str
    step_index: int = Field(ge=0)
    category: ErrorCategory
    message: str
    tool_name: str | None = None


class RunAnalysisResult(BaseModel):
    """Flat analysis result for a single benchmark run."""

    run_id: str
    task_id: str
    agent_id: str
    strategy: str
    model: str | None = None
    mcp_profile: str | None = None

    # Validation
    total_score: float | None = Field(default=None, ge=0.0, le=1.0)
    validation_status: str | None = None
    run_status: str | None = None
    agent_status: str | None = None
    scene_status: str | None = None

    # Pass classification: clean_pass | soft_pass | failed_validation | runtime_error
    pass_type: str | None = None

    # Agent / trajectory counters
    tool_call_count: int = Field(default=0, ge=0)
    invalid_tool_call_count: int = Field(default=0, ge=0)
    trajectory_length: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    duration_sec: float | None = Field(default=None, ge=0.0)
    llm_call_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    success: bool | None = None

    # Structured sub-results
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


class ExperimentSummary(BaseModel):
    """Aggregate statistics for an experiment (collection of runs)."""

    total_runs: int = Field(default=0, ge=0)
    successful_runs: int = Field(default=0, ge=0)
    failed_runs: int = Field(default=0, ge=0)
    error_runs: int = Field(default=0, ge=0)

    # Pass-type breakdown
    clean_pass_count: int = Field(default=0, ge=0)
    soft_pass_count: int = Field(default=0, ge=0)
    failed_validation_count: int = Field(default=0, ge=0)
    runtime_error_count: int = Field(default=0, ge=0)
    failure_rate: float | None = Field(default=None, ge=0.0, le=1.0)

    # Backward-compatible aliases for older tests and consumers.
    failed_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    clean_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    soft_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    strict_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    reported_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    reported_success_rate_all_runs: float | None = Field(default=None, ge=0.0, le=1.0)
    strict_success_rate_excluding_infra: float | None = Field(default=None, ge=0.0, le=1.0)

    # Agent completion breakdown
    agent_completed_count: int = Field(default=0, ge=0)
    agent_completed_after_scene_passed_count: int = Field(default=0, ge=0)
    agent_incomplete_count: int = Field(default=0, ge=0)
    agent_error_count: int = Field(default=0, ge=0)

    average_scene_score: float | None = Field(default=None, ge=0.0, le=1.0)
    average_score_completed: float | None = Field(default=None, ge=0.0, le=1.0)
    average_score_strict: float | None = Field(default=None, ge=0.0, le=1.0)
    average_score_passed_only: float | None = Field(default=None, ge=0.0, le=1.0)
    scene_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    run_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    agent_completion_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    average_tool_call_count: float | None = Field(default=None, ge=0.0)
    average_duration_sec: float | None = Field(default=None, ge=0.0)
    average_llm_calls: float | None = Field(default=None, ge=0.0)

    most_common_errors: list[tuple[str, int]] = Field(default_factory=list)

    # Infrastructure / model failure rates
    infra_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    model_failure_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    soft_success_diagnostic_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    validation_failure_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    tool_runtime_failure_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    reported_success_rate_excluding_infra: float | None = Field(default=None, ge=0.0, le=1.0)
    infra_socket_timeouts: int = Field(default=0, ge=0)
    infra_empty_socket_responses: int = Field(default=0, ge=0)
    infra_worker_restarts: int = Field(default=0, ge=0)
    no_progress_by_reason: dict[str, int] = Field(default_factory=dict)

    best_run: str | None = None
    worst_run: str | None = None


class ExperimentAnalysisResult(BaseModel):
    """Collection of RunAnalysisResult for an experiment (multiple runs/tasks)."""

    experiment_id: str
    runs: list[RunAnalysisResult] = Field(default_factory=list)
    summary: ExperimentSummary = Field(default_factory=ExperimentSummary)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_runs(self) -> int:
        return len(self.runs)

    @property
    def passed_runs(self) -> int:
        return sum(1 for r in self.runs if r.success is True)

    @property
    def avg_score(self) -> float | None:
        scores = [r.total_score for r in self.runs if r.total_score is not None]
        return sum(scores) / len(scores) if scores else None


class ComparisonGroup(BaseModel):
    """Aggregated statistics for one value of a comparison dimension."""

    dimension: ComparisonDimension
    value: str
    run_count: int = Field(ge=0)
    success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    avg_score: float | None = Field(default=None, ge=0.0, le=1.0)
    avg_tool_calls: float | None = Field(default=None, ge=0.0)
    avg_duration_sec: float | None = Field(default=None, ge=0.0)
    avg_cost: float | None = Field(default=None, ge=0.0)
    validation_failures: int = Field(default=0, ge=0)


class ComparisonReport(BaseModel):
    """Result of comparing runs along one dimension."""

    dimension: ComparisonDimension
    groups: list[ComparisonGroup] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReportConfig(BaseModel):
    """Configuration for report generation."""

    report_id: str = "default"
    title: str = "Benchmark Report"
    input_dir: Path = Field(default=Path("./results"))
    output_dir: Path = Field(default=Path("./reports"))
    formats: list[str] = Field(default_factory=lambda: ["json", "csv", "markdown", "html"])

    # Section toggles
    include_runs: bool = True
    include_group_comparison: bool = True
    include_error_taxonomy: bool = True
    include_trace_details: bool = True
    include_artifact_links: bool = True

    # Legacy toggles kept for backward compatibility
    include_per_tool: bool = True
    include_errors: bool = True
    include_validation_details: bool = True

    metadata: dict[str, Any] = Field(default_factory=dict)


class ReportArtifact(BaseModel):
    """Describes a single exported report file."""

    format: str
    path: Path
    size_bytes: int = Field(default=0, ge=0)


class RankedRun(BaseModel):
    """A RunAnalysisResult with an attached rank and score used for ranking."""

    rank: int = Field(ge=1)
    run: RunAnalysisResult
    score_used: float | None = None
    time_efficiency: float | None = Field(default=None, ge=0.0)
    tool_efficiency: float | None = Field(default=None, ge=0.0)


class RankedGroup(BaseModel):
    """A ComparisonGroup with an attached rank and the score used for ranking."""

    rank: int = Field(ge=1)
    group: ComparisonGroup
    score_used: float | None = None
    time_efficiency: float | None = Field(default=None, ge=0.0)
    tool_efficiency: float | None = Field(default=None, ge=0.0)


# ---------------------------------------------------------------------------
# Backward-compatible aliases (used by internal analysis sub-modules)
# ---------------------------------------------------------------------------
ToolCallMetrics = ToolCallMetric
AgentMetrics = RunAnalysisResult
ValidationMetrics = ValidationMetric
RunReport = RunAnalysisResult
DimensionStats = ComparisonGroup
