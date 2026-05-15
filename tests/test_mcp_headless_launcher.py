"""Tests for benchmark.mcp.headless.launcher (no Blender required)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark.mcp.config import McpServerConfig
from benchmark.mcp.errors import McpServerStartError
from benchmark.mcp.headless.launcher import HeadlessBlenderMcpLauncher


def make_config(**overrides) -> McpServerConfig:
    defaults = dict(
        server_distribution="local",
        package_source="./vendor/blender-mcp-bma",
        blender_host="127.0.0.1",
        blender_port=9876,
        profile="minimal",
        disable_telemetry=True,
    )
    defaults.update(overrides)
    return McpServerConfig(**defaults)


# ---------------------------------------------------------------------------
# build_command
# ---------------------------------------------------------------------------

def test_build_command_includes_background_and_factory_startup():
    cfg = make_config()
    launcher = HeadlessBlenderMcpLauncher(cfg, addon_path=Path("/fake/addon.py"))
    with patch("benchmark.mcp.headless.launcher._find_blender", return_value="blender"):
        cmd = launcher.build_command()
    assert "blender" in cmd
    assert "--background" in cmd
    assert "--factory-startup" in cmd


def test_build_command_includes_python_script():
    cfg = make_config()
    launcher = HeadlessBlenderMcpLauncher(cfg, addon_path=Path("/fake/addon.py"))
    with patch("benchmark.mcp.headless.launcher._find_blender", return_value="blender"):
        cmd = launcher.build_command()
    assert "--python" in cmd


def test_build_command_includes_separator_and_addon():
    cfg = make_config()
    launcher = HeadlessBlenderMcpLauncher(cfg, addon_path=Path("/fake/addon.py"))
    with patch("benchmark.mcp.headless.launcher._find_blender", return_value="blender"):
        cmd = launcher.build_command()
    assert "--" in cmd
    assert "--addon" in cmd
    assert "/fake/addon.py" in cmd


def test_build_command_includes_host_and_port():
    cfg = make_config(blender_host="10.0.0.1", blender_port=1234)
    launcher = HeadlessBlenderMcpLauncher(cfg, addon_path=Path("/fake/addon.py"))
    with patch("benchmark.mcp.headless.launcher._find_blender", return_value="blender"):
        cmd = launcher.build_command()
    assert "--host" in cmd
    assert "10.0.0.1" in cmd
    assert "--port" in cmd
    assert "1234" in cmd


def test_build_command_includes_disable_external_assets_for_minimal():
    cfg = make_config(profile="minimal")
    launcher = HeadlessBlenderMcpLauncher(cfg, addon_path=Path("/fake/addon.py"))
    with patch("benchmark.mcp.headless.launcher._find_blender", return_value="blender"):
        cmd = launcher.build_command()
    assert "--disable-external-assets" in cmd


def test_build_command_raises_when_blender_not_found():
    cfg = make_config()
    launcher = HeadlessBlenderMcpLauncher(cfg, addon_path=Path("/fake/addon.py"))
    with patch("benchmark.mcp.headless.launcher._find_blender", return_value=None):
        with pytest.raises(McpServerStartError, match="not found"):
            launcher.build_command()


# ---------------------------------------------------------------------------
# build_env
# ---------------------------------------------------------------------------

def test_build_env_sets_bma_headless():
    cfg = make_config()
    launcher = HeadlessBlenderMcpLauncher(cfg, addon_path=Path("/fake/addon.py"))
    env = launcher.build_env()
    assert env.get("BMA_HEADLESS") == "1"


def test_build_env_sets_bma_addon_path():
    cfg = make_config()
    launcher = HeadlessBlenderMcpLauncher(cfg, addon_path=Path("/fake/addon.py"))
    env = launcher.build_env()
    assert env.get("BMA_ADDON_PATH") == "/fake/addon.py"


def test_build_env_sets_bma_mcp_profile():
    cfg = make_config(profile="no_python")
    launcher = HeadlessBlenderMcpLauncher(cfg, addon_path=Path("/fake/addon.py"))
    env = launcher.build_env()
    assert env.get("BMA_MCP_PROFILE") == "no_python"


# ---------------------------------------------------------------------------
# start / stop / is_running
# ---------------------------------------------------------------------------

def test_start_returns_popen():
    cfg = make_config()
    launcher = HeadlessBlenderMcpLauncher(cfg, addon_path=Path("/fake/addon.py"))
    mock_proc = MagicMock()
    mock_proc.pid = 999
    mock_proc.poll.return_value = None  # simulate running process

    with patch("benchmark.mcp.headless.launcher._find_blender", return_value="blender"), \
         patch("benchmark.mcp.headless.launcher.subprocess.Popen", return_value=mock_proc):
        launcher.start()

    assert launcher.is_running


def test_start_file_not_found_raises():
    cfg = make_config()
    launcher = HeadlessBlenderMcpLauncher(cfg, addon_path=Path("/fake/addon.py"))

    with patch("benchmark.mcp.headless.launcher._find_blender", return_value="blender"), \
         patch(
             "benchmark.mcp.headless.launcher.subprocess.Popen",
             side_effect=FileNotFoundError("blender not found"),
         ):
        with pytest.raises(McpServerStartError):
            launcher.start()


def test_stop_terminates_process():
    cfg = make_config()
    launcher = HeadlessBlenderMcpLauncher(cfg, addon_path=Path("/fake/addon.py"))
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.pid = 42
    launcher._process = mock_proc

    with patch("benchmark.mcp.headless.launcher.os.getpgid", return_value=42), \
         patch("benchmark.mcp.headless.launcher.os.killpg"):
        launcher.stop()

    assert launcher._process is None


def test_context_manager():
    cfg = make_config()
    launcher = HeadlessBlenderMcpLauncher(cfg, addon_path=Path("/fake/addon.py"))
    mock_proc = MagicMock()
    mock_proc.pid = 1
    mock_proc.poll.return_value = 0

    with patch("benchmark.mcp.headless.launcher._find_blender", return_value="blender"), \
         patch("benchmark.mcp.headless.launcher.subprocess.Popen", return_value=mock_proc):
        with launcher:
            pass  # start and stop via __enter__/__exit__
