from __future__ import annotations

import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentStrategyName(str, Enum):
    DIRECT_TOOL_CALLING = "direct_tool_calling"
    REACT = "react"
    PLAN_AND_EXECUTE = "plan_and_execute"
    REMOTE_AGENT = "remote_agent"
    PLAN_EXECUTE_REACT_REPAIR = "plan_execute_react_repair"


class LlmProvider(str, Enum):
    OPENROUTER = "openrouter"
    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


class RemoteAgentProvider(str, Enum):
    CODEX = "codex"
    CLAUDE_CODE = "claude_code"
    GENERIC_HTTP = "generic_http"
    GENERIC_COMMAND = "generic_command"
    MOCK = "mock"


class AgentRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class AgentStepType(str, Enum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    PLAN = "plan"
    FINAL = "final"
    ERROR = "error"


class ToolCallStatus(str, Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class LlmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: LlmProvider = LlmProvider.MOCK
    model: str = "mock"
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, gt=0)
    timeout_sec: int = Field(default=120, gt=0)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be empty")
        return value


class RemoteAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: RemoteAgentProvider = RemoteAgentProvider.MOCK
    agent_id: str = "mock"
    endpoint_url: str | None = None
    api_key_env: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    workspace_dir: Path | None = None
    timeout_sec: int = Field(default=300, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("agent_id must not be empty")
        return value

    @model_validator(mode="after")
    def validate_provider_requirements(self) -> "RemoteAgentConfig":
        if self.provider == RemoteAgentProvider.GENERIC_HTTP and not self.endpoint_url:
            raise ValueError("endpoint_url is required for generic_http remote agents")
        if self.provider == RemoteAgentProvider.GENERIC_COMMAND and not self.command:
            raise ValueError("command is required for generic_command remote agents")
        return self


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    strategy: AgentStrategyName = AgentStrategyName.DIRECT_TOOL_CALLING
    mcp_profile: str = "minimal"
    llm: LlmConfig | None = None
    remote_agent: RemoteAgentConfig | None = None
    max_steps: int = Field(default=20, gt=0)
    max_retries: int = Field(default=1, ge=0)
    step_timeout_sec: int = Field(default=120, gt=0)
    stop_after_scene_passed: bool = False
    detect_repeated_actions: bool = True
    detect_duplicate_objects: bool = True
    detect_no_progress: bool = True
    no_progress_limit: int = Field(default=2, ge=1)
    repeated_action_mode: str = "debug"
    allow_python_tools: bool = False
    allow_inspection_tools: bool = True
    system_prompt_template: str | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    max_steps_by_category: dict[str, int] = Field(default_factory=dict)
    trace_enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_id", "mcp_profile")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value

    @model_validator(mode="after")
    def validate_backend_config(self) -> "AgentConfig":
        if self.strategy == AgentStrategyName.REMOTE_AGENT:
            if self.remote_agent is None:
                raise ValueError("remote_agent is required for remote_agent strategy")
        elif self.llm is None:
            raise ValueError(f"llm is required for {self.strategy.value} strategy")
        if not self.allow_python_tools:
            if "execute_blender_code" in self.allowed_tools:
                raise ValueError("execute_blender_code requires allow_python_tools=true")
            if (
                self.system_prompt_template is not None
                and "execute_blender_code" in self.system_prompt_template
            ):
                raise ValueError("system_prompt_template must not include execute_blender_code")
        return self


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tool name must not be empty")
        return value


class ToolCallResult(BaseModel):
    name: str
    status: ToolCallStatus
    result: dict[str, Any] | list[Any] | str | int | float | bool | None = None
    error: str | None = None
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None
    duration_sec: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tool name must not be empty")
        return value


class AgentStep(BaseModel):
    step_index: int = Field(ge=0)
    step_type: AgentStepType
    thought: str | None = None
    action: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    observation: str | dict[str, Any] | list[Any] | None = None
    error: str | None = None
    raw_llm_response: dict[str, Any] | list[Any] | str | None = None
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None
    duration_sec: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    run_id: str
    task_id: str
    agent_id: str
    strategy: AgentStrategyName
    model: str | None = None
    steps: list[AgentStep] = Field(default_factory=list)
    final_message: str | None = None
    success: bool | None = None
    error: str | dict[str, Any] | None = None
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None
    duration_sec: float | None = Field(default=None, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "task_id", "agent_id")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value

    @field_validator("steps")
    @classmethod
    def sort_steps(cls, value: list[AgentStep]) -> list[AgentStep]:
        return sorted(value, key=lambda step: step.step_index)

    def add_step(self, step_type: AgentStepType | str, **kwargs: Any) -> "AgentTrace":
        if "step_index" not in kwargs:
            kwargs["step_index"] = len(self.steps)
        step = AgentStep(step_type=step_type, **kwargs)
        return self.model_copy(update={"steps": sorted([*self.steps, step], key=lambda item: item.step_index)})


class AgentRunResult(BaseModel):
    ok: bool
    run_id: str
    task_id: str
    agent_id: str
    trace_path: Path | None = None
    scene_snapshot_path: Path | None = None
    artifacts_dir: Path | None = None
    status: AgentRunStatus
    error: str | None = None
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None
    duration_sec: float | None = Field(default=None, ge=0.0)
    summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "task_id", "agent_id")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value


class AgentBackendType(str, Enum):
    LLM = "llm"
    REMOTE_AGENT = "remote_agent"
    MOCK = "mock"
