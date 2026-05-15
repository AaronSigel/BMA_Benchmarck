from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from benchmark.mcp.errors import McpConfigError

_DEFAULT_HOST = "localhost"
_DEFAULT_PORT = 9876
_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs" / "mcp"

ServerMode = Literal["external", "fork", "adapter"]
ServerDistribution = Literal["upstream", "fork", "local"]


class McpServerConfig(BaseModel):
    # How the MCP server is accessed
    server_mode: ServerMode = "external"
    # Which distribution of blender-mcp to use
    server_distribution: ServerDistribution = "upstream"
    # Package path or name for fork/local distributions
    package_source: str | None = None
    # Launch command and arguments (used when server_mode != "external")
    command: str = "uvx"
    args: list[str] = Field(default_factory=lambda: ["blender-mcp"])
    # Blender add-on socket coordinates
    blender_host: str = _DEFAULT_HOST
    blender_port: int = Field(default=_DEFAULT_PORT, ge=1, le=65535)
    # Telemetry
    disable_telemetry: bool = True
    # Active BMA tool-gating profile
    profile: str = "full"
    # Timeouts
    startup_timeout_sec: int = Field(default=30, ge=1)
    request_timeout_sec: int = Field(default=60, ge=1)
    # Extra env vars injected into the server process
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("blender_host")
    @classmethod
    def _non_empty_host(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("blender_host must not be empty")
        return value

    @field_validator("profile")
    @classmethod
    def _non_empty_profile(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("profile must not be empty")
        return value.lower()


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def load_mcp_config(path: Path | str) -> McpServerConfig:
    """Load McpServerConfig from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise McpConfigError(f"MCP config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise McpConfigError(f"Failed to parse MCP config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise McpConfigError(f"MCP config must be a YAML mapping: {path}")
    try:
        return McpServerConfig.model_validate(raw)
    except Exception as exc:
        raise McpConfigError(f"Invalid MCP config {path}: {exc}") from exc


def dump_mcp_config(config: McpServerConfig, path: Path | str) -> None:
    """Serialise McpServerConfig to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = config.model_dump(mode="json")
    try:
        path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    except OSError as exc:
        raise McpConfigError(f"Failed to write MCP config to {path}: {exc}") from exc


def build_mcp_env(config: McpServerConfig) -> dict[str, str]:
    """Return the env-var dict that should be passed to the MCP server process."""
    result = dict(config.env)
    result["BMA_MCP_PROFILE"] = config.profile
    result["BMA_SOCKET_HOST"] = config.blender_host
    result["BMA_SOCKET_PORT"] = str(config.blender_port)
    result["BMA_SERVER_MODE"] = config.server_mode
    result["BMA_SERVER_DISTRIBUTION"] = config.server_distribution
    # Always propagate the telemetry flag so the server process inherits the right value.
    # Telemetry is only enabled when explicitly opted in via BMA_ENABLE_TELEMETRY=true.
    result["DISABLE_TELEMETRY"] = "false" if not config.disable_telemetry else "true"
    if config.package_source is not None:
        result["BMA_PACKAGE_SOURCE"] = config.package_source
    return result


# ---------------------------------------------------------------------------
# Convenience helpers (kept for internal use)
# ---------------------------------------------------------------------------

def load_named_config(name: str, configs_dir: Path = _CONFIGS_DIR) -> McpServerConfig:
    """Load a named preset config from configs/mcp/<name>.yaml."""
    return load_mcp_config(configs_dir / f"{name}.yaml")


def config_from_env() -> McpServerConfig:
    """Build McpServerConfig from environment variables."""
    return McpServerConfig(
        server_mode=os.environ.get("BMA_SERVER_MODE", "external"),  # type: ignore[arg-type]
        server_distribution=os.environ.get("BMA_SERVER_DISTRIBUTION", "upstream"),  # type: ignore[arg-type]
        package_source=os.environ.get("BMA_PACKAGE_SOURCE"),
        blender_host=os.environ.get("BMA_SOCKET_HOST", _DEFAULT_HOST),
        blender_port=int(os.environ.get("BMA_SOCKET_PORT", str(_DEFAULT_PORT))),
        # Telemetry is off by default; opt in explicitly with BMA_ENABLE_TELEMETRY=true.
        disable_telemetry=os.environ.get("BMA_ENABLE_TELEMETRY", "").lower() != "true",
        profile=os.environ.get("BMA_MCP_PROFILE", "full"),
    )
