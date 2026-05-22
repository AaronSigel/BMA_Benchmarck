"""Blender worker lifecycle policy tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from benchmark.mcp.config import McpServerConfig
from benchmark.mcp.socket_watchdog import BlenderSocketWatchdog, SocketFailureEvent, WatchdogAction


def _watchdog(**lifecycle) -> BlenderSocketWatchdog:
    config = McpServerConfig(profile="minimal", blender_host="localhost", blender_port=9876)
    return BlenderSocketWatchdog(
        config,
        mcp_config_path="/tmp/mcp.yaml",
        worker_lifecycle={
            "restart_on_tool_timeout": True,
            "restart_on_empty_socket_response": True,
            "restart_on_reset_failure": True,
            "restart_on_consecutive_snapshot_failures": 2,
            **lifecycle,
        },
    )


def test_tool_timeout_triggers_health_check() -> None:
    watchdog = _watchdog()
    with patch.object(watchdog, "restart_worker", return_value=True) as restart:
        action = watchdog.on_socket_failure(
            SocketFailureEvent(error_type="ToolTimeout", tool_name="bma_set_transform")
        )
    assert restart.called
    assert action == WatchdogAction.RESTARTED


def test_empty_socket_response_triggers_restart() -> None:
    watchdog = _watchdog()
    with patch.object(watchdog, "health_check", return_value=False), patch.object(
        watchdog, "restart_worker", return_value=True
    ) as restart:
        action = watchdog.on_socket_failure(
            SocketFailureEvent(error_type="EmptySocketResponse", tool_name="bma_get_scene_snapshot")
        )
    assert restart.called
    assert action == WatchdogAction.RESTARTED


def test_mini_check_failure_forces_restart_when_tcp_alive() -> None:
    watchdog = _watchdog(restart_on_empty_socket_response=False)
    adapter = MagicMock()
    with patch.object(watchdog, "health_check", return_value=True), patch.object(
        watchdog, "mini_check", return_value=False
    ), patch.object(watchdog, "restart_worker", return_value=True) as restart:
        action = watchdog.on_socket_failure(
            SocketFailureEvent(error_type="EmptySocketResponse", tool_name="bma_assign_material"),
            adapter=adapter,
        )
    restart.assert_called_once()
    assert action == WatchdogAction.RESTARTED


def test_worker_restart_count_recorded() -> None:
    watchdog = _watchdog()
    with patch("benchmark.mcp.socket_watchdog.subprocess.run", return_value=MagicMock(returncode=0)):
        watchdog.restart_worker(reason="test_restart")
    counters = watchdog._counters.to_dict()
    assert counters["worker_restart_count"] == 1
    assert counters["restart_reasons"] == ["test_restart"]


def test_restart_worker_start_command_has_no_wait_flag() -> None:
    watchdog = _watchdog()
    captured: list[list[str]] = []

    def _capture_run(cmd, **kwargs):
        captured.append(list(cmd))
        return MagicMock(returncode=0)

    with patch("benchmark.mcp.socket_watchdog.subprocess.run", side_effect=_capture_run):
        watchdog.restart_worker(reason="test_restart")

    assert len(captured) == 2
    start_cmd = captured[1]
    assert "start-headless-blender" in start_cmd
    assert "--wait" not in start_cmd
