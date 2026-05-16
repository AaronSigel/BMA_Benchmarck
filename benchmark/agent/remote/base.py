from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator

from benchmark.agent.models import AgentTrace


class RemoteAgentArtifact(BaseModel):
    name: str
    path: Path
    kind: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("artifact name must not be empty")
        return value


class RemoteAgentRequest(BaseModel):
    task: dict[str, Any]
    mcp_config_path: Path | None = None
    mcp_profile: str | None = None
    tool_contracts: list[dict[str, Any]] = Field(default_factory=list)
    output_dir: Path
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("mcp_profile")
    @classmethod
    def validate_optional_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("mcp_profile must not be empty")
        return value


class RemoteAgentResponse(BaseModel):
    ok: bool
    trace: AgentTrace | None = None
    scene_snapshot_path: Path | None = None
    artifacts: list[RemoteAgentArtifact] = Field(default_factory=list)
    error: str | None = None
    raw_response: Any = None


@runtime_checkable
class RemoteAgentClient(Protocol):
    def run_task(self, request: RemoteAgentRequest) -> RemoteAgentResponse:
        """Run a benchmark task through a provider-neutral remote agent."""
