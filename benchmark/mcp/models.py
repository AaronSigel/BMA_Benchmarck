from __future__ import annotations

import datetime
from typing import Any

from pydantic import BaseModel, Field


class McpSmokeResult(BaseModel):
    """Result of an MCP smoke-check run (no LLM involved)."""

    ok: bool
    profile: str
    server_distribution: str
    blender_socket_available: bool
    telemetry_disabled: bool
    available_tools: list[str] = Field(default_factory=list)
    disabled_tools: list[str] = Field(default_factory=list)
    scene_info: dict[str, Any] | None = None
    profile_info: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime.datetime | None = None
    finished_at: datetime.datetime | None = None
    duration_sec: float | None = None

    @classmethod
    def failure(
        cls,
        *,
        profile: str,
        server_distribution: str,
        blender_socket_available: bool,
        telemetry_disabled: bool,
        error: str,
        started_at: datetime.datetime | None = None,
        finished_at: datetime.datetime | None = None,
    ) -> "McpSmokeResult":
        duration = None
        if started_at and finished_at:
            duration = (finished_at - started_at).total_seconds()
        return cls(
            ok=False,
            profile=profile,
            server_distribution=server_distribution,
            blender_socket_available=blender_socket_available,
            telemetry_disabled=telemetry_disabled,
            error=error,
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=duration,
        )

    def finish(self, finished_at: datetime.datetime | None = None) -> "McpSmokeResult":
        """Return a copy with finished_at and duration_sec filled in."""
        ts = finished_at or datetime.datetime.now(datetime.timezone.utc)
        duration = None
        if self.started_at:
            duration = (ts - self.started_at).total_seconds()
        return self.model_copy(update={"finished_at": ts, "duration_sec": duration})
