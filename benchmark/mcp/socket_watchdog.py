from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from benchmark.mcp.config import McpServerConfig
from benchmark.mcp.connection_check import check_blender_socket
from benchmark.mcp.errors import BlenderSocketUnavailableError

log = logging.getLogger(__name__)


class WatchdogAction(str, Enum):
    ALLOW_RETRY = "allow_retry"
    RESTARTED = "restarted"
    MARK_INFRA_ERROR = "mark_infra_error"
    HEALTHY = "healthy"


@dataclass
class SocketFailureEvent:
    error_type: str
    tool_name: str | None = None
    message: str = ""
    failure_stage: str | None = None
    attempt: int = 1


@dataclass
class WatchdogCounters:
    infra_socket_timeouts: int = 0
    infra_empty_socket_responses: int = 0
    infra_worker_restarts: int = 0
    infra_reset_failures: int = 0
    infra_snapshot_failures: int = 0
    restart_reasons: list[str] = field(default_factory=list)

    def record_failure(self, error_type: str, *, reset: bool = False, snapshot: bool = False) -> None:
        if error_type in {"ToolTimeout", "SocketTimeout"}:
            self.infra_socket_timeouts += 1
        if error_type == "EmptySocketResponse":
            self.infra_empty_socket_responses += 1
        if reset:
            self.infra_reset_failures += 1
        if snapshot:
            self.infra_snapshot_failures += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "infra_socket_timeouts": self.infra_socket_timeouts,
            "infra_empty_socket_responses": self.infra_empty_socket_responses,
            "infra_worker_restarts": self.infra_worker_restarts,
            "infra_reset_failures": self.infra_reset_failures,
            "infra_snapshot_failures": self.infra_snapshot_failures,
            "restart_reasons": list(self.restart_reasons),
        }


_GLOBAL_COUNTERS = WatchdogCounters()
_GLOBAL_RUNS_SINCE_RESTART = 0


def get_watchdog_counters() -> WatchdogCounters:
    return _GLOBAL_COUNTERS


def reset_watchdog_counters() -> None:
    global _GLOBAL_COUNTERS, _GLOBAL_RUNS_SINCE_RESTART
    _GLOBAL_COUNTERS = WatchdogCounters()
    _GLOBAL_RUNS_SINCE_RESTART = 0


def increment_runs_since_restart() -> int:
    global _GLOBAL_RUNS_SINCE_RESTART
    _GLOBAL_RUNS_SINCE_RESTART += 1
    return _GLOBAL_RUNS_SINCE_RESTART


def get_runs_since_last_restart() -> int:
    return _GLOBAL_RUNS_SINCE_RESTART


class BlenderSocketWatchdog:
    """Health-check и CLI restart Blender worker при socket failures."""

    def __init__(
        self,
        config: McpServerConfig,
        *,
        mcp_config_path: Path | str | None = None,
        counters: WatchdogCounters | None = None,
    ) -> None:
        self._config = config
        self._mcp_config_path = Path(mcp_config_path) if mcp_config_path else None
        self._counters = counters or _GLOBAL_COUNTERS

    def health_check(self, timeout_sec: float = 5.0) -> bool:
        try:
            check_blender_socket(
                self._config.blender_host,
                self._config.blender_port,
                timeout_sec=timeout_sec,
            )
            return True
        except BlenderSocketUnavailableError:
            return False

    def on_socket_failure(self, event: SocketFailureEvent) -> WatchdogAction:
        self._counters.record_failure(
            event.error_type,
            reset=event.failure_stage == "reset_scene",
            snapshot=event.failure_stage in {"pre_run_snapshot", "post_run_snapshot", "snapshot_collection"},
        )
        log.warning(
            "[watchdog] socket failure tool=%s type=%s stage=%s attempt=%d",
            event.tool_name,
            event.error_type,
            event.failure_stage,
            event.attempt,
        )
        if self.health_check():
            log.info("[watchdog] socket alive after failure — allow retry")
            return WatchdogAction.ALLOW_RETRY
        log.warning("[watchdog] socket dead — attempting worker restart")
        if self.restart_worker(reason=f"{event.error_type}:{event.tool_name or 'unknown'}"):
            if self.health_check(timeout_sec=15.0):
                return WatchdogAction.RESTARTED
        return WatchdogAction.MARK_INFRA_ERROR

    def restart_worker(self, *, reason: str) -> bool:
        if self._mcp_config_path is None:
            log.warning("[watchdog] no mcp_config_path — cannot restart worker")
            return False
        try:
            stop_cmd = [
                sys.executable,
                "-m",
                "benchmark.mcp.cli",
                "--config",
                str(self._mcp_config_path),
                "stop-headless-blender",
            ]
            subprocess.run(stop_cmd, check=False, capture_output=True, text=True, timeout=30)
            start_cmd = [
                sys.executable,
                "-m",
                "benchmark.mcp.cli",
                "--config",
                str(self._mcp_config_path),
                "start-headless-blender",
                "--wait",
            ]
            result = subprocess.run(
                start_cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(60.0, float(self._config.startup_timeout_sec) + 30.0),
            )
            if result.returncode != 0:
                log.error("[watchdog] worker restart failed: %s", result.stderr or result.stdout)
                return False
            self._counters.infra_worker_restarts += 1
            self._counters.restart_reasons.append(reason)
            global _GLOBAL_RUNS_SINCE_RESTART
            _GLOBAL_RUNS_SINCE_RESTART = 0
            log.info("[watchdog] worker restarted reason=%s", reason)
            return True
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.error("[watchdog] worker restart exception: %s", exc)
            return False

    def mini_check(self, adapter: Any) -> bool:
        """Preflight mini-check после restart: ping → reset → create cube → snapshot."""
        if not self.health_check():
            return False
        reset = adapter.reset_scene()
        if isinstance(reset, dict) and not reset.get("ok", True) and reset.get("warning"):
            return False
        create = adapter.call_tool(
            "bma_create_object",
            {"name": "WatchdogCube", "object_type": "CUBE", "if_exists": "update"},
        )
        if isinstance(create, dict) and not create.get("ok"):
            return False
        snap = adapter._call_get_scene_snapshot()
        return isinstance(snap, dict) and snap.get("ok")

    def proactive_restart_if_due(self, every_n_runs: int) -> bool:
        if every_n_runs <= 0:
            return False
        if get_runs_since_last_restart() < every_n_runs:
            return False
        restarted = self.restart_worker(reason=f"proactive_every_{every_n_runs}_runs")
        return restarted
