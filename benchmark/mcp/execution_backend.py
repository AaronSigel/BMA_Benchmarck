"""MCP execution backends integrated with Experiment Runner.

McpExecutionBackend handles ExecutionMode.MCP_SMOKE and ExecutionMode.MCP_EXTERNAL.
It does NOT use LLM, agent runtime, or tool-call metrics.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from benchmark.mcp.config import McpServerConfig, config_from_env, load_mcp_config
from benchmark.mcp.connection_check import is_blender_socket_available
from benchmark.mcp.errors import (
    BlenderSocketUnavailableError,
    McpConfigError,
    McpExecutionError,
    McpLayerError,
    McpSmokeError,
)
from benchmark.mcp.models import McpSmokeResult
from benchmark.mcp.profiles import McpProfile, is_tool_allowed
from benchmark.mcp.server_adapter import ExternalBlenderMcpServerAdapter
from benchmark.mcp.tool_registry import McpToolRegistry
from benchmark.runner.execution import ExecutionBackend, ExecutionResult, _error_result
from benchmark.runner.models import ExecutionMode, RunConfig

# Tools that must never be called in safe profiles regardless of what was requested.
_PYTHON_RESTRICTED_PROFILES = frozenset({"minimal", "no_python", "inspection_enabled"})


class McpExecutionBackend(ExecutionBackend):
    """Execution backend for MCP smoke and external modes.

    Integrates with Experiment Runner (ExecutionBackend ABC).
    Produces ExecutionResult for the runner and saves a McpSmokeResult JSON artifact.

    Constraints:
    - No LLM / agent runtime / tool-call metrics.
    - Never calls execute_blender_code for no_python/minimal/inspection_enabled profiles.
    """

    mode = ExecutionMode.MCP_SMOKE  # primary mode; also handles MCP_EXTERNAL

    def execute(self, config: RunConfig) -> ExecutionResult:
        started_at = datetime.datetime.now(datetime.timezone.utc)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load MCP config
        try:
            mcp_cfg = _load_mcp_config(config)
        except McpConfigError as exc:
            return _error_result(output_dir, f"MCP config error: {exc}")

        # 2. Apply mcp_profile override from RunConfig
        if config.mcp_profile:
            mcp_cfg = mcp_cfg.model_copy(update={"profile": config.mcp_profile})

        profile_str = mcp_cfg.profile
        is_fork = mcp_cfg.server_distribution in ("fork", "local")

        # 3. Resolve tool list for this mode — no LLM, no agent metrics
        tools = _smoke_tools_for_mode(config.execution_mode, profile_str, is_fork)

        # 4. Probe Blender socket
        socket_ok = is_blender_socket_available(mcp_cfg)
        registry = McpToolRegistry()

        if not socket_ok:
            result = McpSmokeResult.failure(
                profile=profile_str,
                server_distribution=mcp_cfg.server_distribution,
                blender_socket_available=False,
                telemetry_disabled=mcp_cfg.disable_telemetry,
                error=(
                    f"Blender socket not reachable at "
                    f"{mcp_cfg.blender_host}:{mcp_cfg.blender_port}"
                ),
                started_at=started_at,
                finished_at=datetime.datetime.now(datetime.timezone.utc),
            )
            return _save_and_build(result, output_dir)

        # 5. Execute tools — no execute_blender_code in restricted profiles
        adapter = ExternalBlenderMcpServerAdapter(mcp_cfg, registry=registry)
        tool_results: dict[str, Any] = {}
        error_msg: str | None = None

        for tool in tools:
            # Hard guard: never call execute_blender_code in restricted profiles
            if tool == "execute_blender_code" and profile_str in _PYTHON_RESTRICTED_PROFILES:
                tool_results[tool] = {
                    "status": "skipped",
                    "reason": f"execute_blender_code blocked in profile '{profile_str}'",
                }
                continue

            try:
                raw = adapter.call_tool(tool)
                tool_results[tool] = {"status": "ok", "result": raw}
            except McpLayerError as exc:
                tool_results[tool] = {"status": "error", "error": str(exc)}
                error_msg = str(exc)
                break  # stop on first failure

        # 6. Build McpSmokeResult
        try:
            profile_enum = McpProfile(profile_str)
        except ValueError:
            profile_enum = McpProfile.FULL

        available = [c.name for c in registry.list_for_profile(profile_enum)]
        disabled = [c.name for c in registry.list_disabled_tools(profile_enum)]

        scene_info = _extract_result(tool_results, "get_scene_info")
        profile_info = _extract_result(tool_results, "get_bma_profile_info")

        smoke_result = McpSmokeResult(
            ok=(error_msg is None),
            profile=profile_str,
            server_distribution=mcp_cfg.server_distribution,
            blender_socket_available=socket_ok,
            telemetry_disabled=mcp_cfg.disable_telemetry,
            available_tools=available,
            disabled_tools=disabled,
            scene_info=scene_info,
            profile_info=profile_info,
            error=error_msg,
            started_at=started_at,
        ).finish()

        return _save_and_build(smoke_result, output_dir)


# ---------------------------------------------------------------------------
# Legacy backends (kept for backward compat; delegate to McpExecutionBackend)
# ---------------------------------------------------------------------------

class McpSmokeBackend(McpExecutionBackend):
    """Alias kept for backward compatibility."""
    mode = ExecutionMode.MCP_SMOKE


class McpExternalBackend(McpExecutionBackend):
    """Alias kept for backward compatibility."""
    mode = ExecutionMode.MCP_EXTERNAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mcp_config(run_config: RunConfig) -> McpServerConfig:
    if run_config.mcp_config_path:
        return load_mcp_config(run_config.mcp_config_path)
    return config_from_env()


def _smoke_tools_for_mode(
    mode: ExecutionMode,
    profile: str,
    is_fork: bool,
) -> list[str]:
    """Return the ordered list of tools to call for this smoke run.

    No LLM tools, no agent-runtime tools, no tool-call metrics.
    """
    tools: list[str] = []

    if is_fork:
        tools.append("get_bma_profile_info")

    tools.append("get_scene_info")

    if mode == ExecutionMode.MCP_EXTERNAL:
        # External mode adds object inspection
        tools.append("get_object_info")

    return tools


def _extract_result(tool_results: dict, tool_name: str) -> dict | None:
    entry = tool_results.get(tool_name)
    if entry and entry.get("status") == "ok":
        result = entry.get("result")
        if isinstance(result, dict):
            return result
        return {"raw": result}
    return None


def _save_and_build(result: McpSmokeResult, output_dir: Path) -> ExecutionResult:
    """Persist McpSmokeResult JSON and return the runner-compatible ExecutionResult."""
    result_path = output_dir / "mcp_smoke_result.json"
    result_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return ExecutionResult(
        ok=result.ok,
        scene_snapshot_path=None,  # MCP smoke doesn't produce a scene snapshot
        artifacts_dir=output_dir,
        output_files=[result_path],
        error=result.error,
        metadata={
            "mode": "mcp_smoke",
            "profile": result.profile,
            "server_distribution": result.server_distribution,
            "blender_socket_available": result.blender_socket_available,
            "telemetry_disabled": result.telemetry_disabled,
            "duration_sec": result.duration_sec,
        },
    )
