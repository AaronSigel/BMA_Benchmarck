from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from benchmark.mcp.config import McpServerConfig
from benchmark.mcp.errors import BlenderSocketUnavailableError
from benchmark.mcp.socket_watchdog import (
    BlenderSocketWatchdog,
    SocketFailureEvent,
    WatchdogAction,
    reset_watchdog_counters,
)


@pytest.fixture(autouse=True)
def _reset_counters() -> None:
    reset_watchdog_counters()


def test_socket_timeout_triggers_health_check() -> None:
    cfg = McpServerConfig(profile="minimal", blender_host="localhost", blender_port=9876)
    watchdog = BlenderSocketWatchdog(cfg)
    with patch.object(watchdog, "health_check", return_value=True) as health:
        action = watchdog.on_socket_failure(
            SocketFailureEvent(error_type="ToolTimeout", tool_name="bma_create_light")
        )
    health.assert_called_once()
    assert action == WatchdogAction.ALLOW_RETRY


def test_dead_socket_triggers_worker_restart() -> None:
    cfg = McpServerConfig(profile="minimal", blender_host="localhost", blender_port=9876)
    watchdog = BlenderSocketWatchdog(cfg, mcp_config_path="/tmp/mcp.yaml")
    with patch.object(watchdog, "health_check", side_effect=[False, True]):
        with patch.object(watchdog, "restart_worker", return_value=True) as restart:
            action = watchdog.on_socket_failure(
                SocketFailureEvent(error_type="EmptySocketResponse", tool_name="bma_assign_material")
            )
    restart.assert_called_once()
    assert action == WatchdogAction.RESTARTED


def test_empty_socket_response_classified_as_infra() -> None:
    from benchmark.runner.error_classification import classify_failure

    result = classify_failure(error_type="EmptySocketResponse", failure_stage="socket_response")
    assert result.is_infra_failure is True
    assert result.is_model_failure is False
    assert result.error_class.value == "INFRA_ERROR"
