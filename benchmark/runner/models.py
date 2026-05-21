from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExecutionMode(str, Enum):
    REPLAY = "replay"
    BLENDER_SMOKE = "blender_smoke"
    EXTERNAL_SNAPSHOT = "external_snapshot"
    MCP_SMOKE = "mcp_smoke"
    MCP_EXTERNAL = "mcp_external"
    AGENT_MCP = "agent_mcp"
    REMOTE_AGENT = "remote_agent"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class AgentStatus(str, Enum):
    COMPLETED = "completed"
    COMPLETED_AFTER_SCENE_PASSED = "completed_after_scene_passed"
    MAX_STEPS_REACHED = "max_steps_reached"
    INVALID_RESPONSE = "invalid_response"
    TOOL_ERROR = "tool_error"
    RUNTIME_ERROR = "runtime_error"
    REPEATED_ACTION_DETECTED = "repeated_action_detected"
    DUPLICATE_OBJECT_DETECTED = "duplicate_object_detected"
    NO_PROGRESS_DETECTED = "no_progress_detected"


class SceneStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_AVAILABLE = "not_available"


class RunConfig(BaseModel):
    run_id: str
    task_id: str
    execution_mode: ExecutionMode
    task_path: Path | None = None
    snapshot_path: Path | None = None
    artifacts_dir: Path
    output_dir: Path
    # MCP-specific fields (used when execution_mode is mcp_smoke or mcp_external)
    mcp_config_path: Path | None = None
    mcp_profile: str | None = None
    # Agent-specific fields (used when execution_mode is agent_mcp or remote_agent)
    agent_config_path: Path | None = None
    agent_output_dir: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "task_id")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value


class ExperimentConfig(BaseModel):
    experiment_id: str
    runs: list[RunConfig]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_id")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value


class RunResult(BaseModel):
    run_id: str
    task_id: str
    status: RunStatus
    run_status: RunStatus | None = None
    agent_status: AgentStatus | None = None
    scene_status: SceneStatus | None = None
    execution_mode: ExecutionMode
    validation_result_path: Path | None
    scene_snapshot_path: Path | None
    artifacts_dir: Path
    total_score: float | None = Field(default=None, ge=0.0, le=1.0)
    overall_status: str | None
    started_at: str
    finished_at: str | None
    duration_sec: float | None = Field(default=None, ge=0.0)
    error: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "task_id", "started_at")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value


class ExperimentResult(BaseModel):
    experiment_id: str
    runs: list[RunResult]
    summary: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_id")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value
