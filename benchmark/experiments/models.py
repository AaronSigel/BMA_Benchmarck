from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from benchmark.runner.models import ExecutionMode
from benchmark.tasks.models import DifficultyLevel, TaskCategory


def _validate_non_empty_items(values: list[str]) -> list[str]:
    for value in values:
        if not value.strip():
            raise ValueError("list values must not be empty")
    return values


class MatrixTaskSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(default_factory=list)
    categories: list[TaskCategory] = Field(default_factory=list)
    difficulties: list[DifficultyLevel] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("ids", "tags")
    @classmethod
    def validate_string_lists(cls, values: list[str]) -> list[str]:
        return _validate_non_empty_items(values)


class MatrixAgentSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    include_remote_agents: bool = False
    config_root: Path = Path("configs/agents")

    @field_validator("ids", "strategies")
    @classmethod
    def validate_string_lists(cls, values: list[str]) -> list[str]:
        return _validate_non_empty_items(values)


class MatrixMcpSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profiles: list[str] = Field(default_factory=list)
    config_root: Path = Path("configs/mcp")

    @field_validator("profiles")
    @classmethod
    def validate_profiles(cls, values: list[str]) -> list[str]:
        return _validate_non_empty_items(values)


class MatrixModelSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(default_factory=list)
    providers: list[str] = Field(default_factory=list)

    @field_validator("ids", "providers")
    @classmethod
    def validate_string_lists(cls, values: list[str]) -> list[str]:
        return _validate_non_empty_items(values)


class MatrixRunVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    agent_id: str
    mcp_profile: str
    model_id: str | None = None
    execution_mode: ExecutionMode
    repetition: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id", "agent_id", "mcp_profile")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value


class EnvironmentRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    required: bool = True
    description: str | None = None
    env_var: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be empty")
        return value


class ReadinessCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    requirements: list[EnvironmentRequirement] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedExperimentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matrix_id: str
    generated_at: str
    git_commit: str | None = None
    python_version: str | None = None
    platform: str | None = None
    task_ids: list[str] = Field(default_factory=list)
    agent_ids: list[str] = Field(default_factory=list)
    mcp_profiles: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    execution_modes: list[ExecutionMode] = Field(default_factory=list)
    repetitions: int = Field(gt=0)
    env_requirements: list[EnvironmentRequirement] = Field(default_factory=list)
    config_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("matrix_id")
    @classmethod
    def validate_matrix_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("matrix_id must not be empty")
        return value


class ExperimentMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matrix_id: str
    title: str | None = None
    description: str | None = None
    tasks: MatrixTaskSelector = Field(default_factory=MatrixTaskSelector)
    agents: MatrixAgentSelector = Field(default_factory=MatrixAgentSelector)
    mcp_profiles: list[str] = Field(default_factory=list)
    models: MatrixModelSelector = Field(default_factory=MatrixModelSelector)
    execution_modes: list[ExecutionMode] = Field(default_factory=list)
    repetitions: int = Field(default=1, gt=0)
    output_root: Path = Path("artifacts/experiments")
    report_config_path: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("matrix_id")
    @classmethod
    def validate_matrix_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("matrix_id must not be empty")
        return value

    @field_validator("mcp_profiles")
    @classmethod
    def validate_mcp_profiles(cls, values: list[str]) -> list[str]:
        return _validate_non_empty_items(values)
