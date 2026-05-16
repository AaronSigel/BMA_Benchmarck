"""Tests for benchmark.mcp.cli (no Blender, no MCP server required)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark.mcp.cli import main


def run_cli(*args: str, expect_exit: int | None = None) -> dict | str:
    """Run CLI and return parsed stdout JSON (or raw string)."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    exit_code = None
    try:
        with redirect_stdout(buf):
            main(list(args))
    except SystemExit as exc:
        exit_code = exc.code

    if expect_exit is not None:
        assert exit_code == expect_exit, f"Expected exit {expect_exit}, got {exit_code}"

    output = buf.getvalue().strip()
    # Return first JSON object from output (may have extra lines)
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return output


# ---------------------------------------------------------------------------
# list-tools
# ---------------------------------------------------------------------------

def test_list_tools_minimal_profile():
    result = run_cli("--profile", "minimal", "list-tools", expect_exit=0)
    assert isinstance(result, dict)
    assert result["profile"] == "minimal"
    assert "get_scene_info" in result["tools"]
    assert "execute_blender_code" not in result["tools"]


def test_list_tools_full_profile():
    result = run_cli("--profile", "full", "list-tools", expect_exit=0)
    assert "execute_blender_code" in result["tools"]
    assert "get_polyhaven_status" in result["tools"]


def test_list_tools_no_python_profile():
    result = run_cli("--profile", "no_python", "list-tools", expect_exit=0)
    assert "execute_blender_code" not in result["tools"]


def test_list_tools_inspection_enabled():
    result = run_cli("--profile", "inspection_enabled", "list-tools", expect_exit=0)
    assert "get_viewport_screenshot" in result["tools"]
    assert "execute_blender_code" not in result["tools"]


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def test_check_fails_when_no_blender():
    result = run_cli("--host", "localhost", "--port", "19999", "check", expect_exit=1)
    assert result["status"] == "FAIL"
    assert result["blender_socket_available"] is False


def test_check_returns_all_required_fields():
    result = run_cli("--host", "localhost", "--port", "19999", "check", expect_exit=1)
    for field in ["status", "blender_socket_available", "host", "port", "profile",
                  "telemetry_disabled", "issues"]:
        assert field in result, f"check missing field: {field}"


def test_check_succeeds_with_mock_socket():
    mock_sock = MagicMock()
    with patch("benchmark.mcp.connection_check.socket.create_connection", return_value=mock_sock):
        result = run_cli("check", expect_exit=0)
    assert result["status"] == "OK"
    assert result["blender_socket_available"] is True


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def test_smoke_fails_when_no_blender(tmp_path):
    out = tmp_path / "smoke.json"
    run_cli(
        "--host", "localhost", "--port", "19999",
        "smoke",
        "--output", str(out),
        expect_exit=1,
    )
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["status"] == "FAIL"
    assert data["checks"]["config_loaded"] is True
    assert data["checks"]["blender_socket_available"] is False


def test_smoke_checks_telemetry_disabled(tmp_path):
    out = tmp_path / "smoke.json"
    run_cli(
        "--host", "localhost", "--port", "19999",
        "smoke",
        "--output", str(out),
        expect_exit=1,
    )
    data = json.loads(out.read_text())
    # Default config has telemetry disabled
    assert data["checks"]["telemetry_disabled"] is True


def test_smoke_output_contains_timestamp(tmp_path):
    out = tmp_path / "smoke.json"
    run_cli("--port", "19999", "smoke", "--output", str(out), expect_exit=1)
    data = json.loads(out.read_text())
    assert "timestamp" in data


def test_smoke_with_config_file(tmp_path):
    out = tmp_path / "result.json"
    run_cli(
        "--config", "configs/mcp/minimal.yaml",
        "smoke",
        "--output", str(out),
        expect_exit=1,
    )
    data = json.loads(out.read_text())
    assert data["profile"] == "minimal"


# ---------------------------------------------------------------------------
# start-server (mocked)
# ---------------------------------------------------------------------------

def test_start_server_upstream_builds_command():
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    mock_proc = MagicMock()
    mock_proc.pid = 1234

    with patch("benchmark.mcp.server_adapter.subprocess.Popen", return_value=mock_proc), \
         redirect_stdout(buf):
        try:
            main(["start-server"])
        except SystemExit:
            pass

    output = buf.getvalue()
    assert "starting" in output or "uvx" in output


def test_start_server_fork_without_source_fails():
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            main(["start-server", "--distribution", "fork"])
        except SystemExit:
            pass

    output = buf.getvalue()
    assert "FAIL" in output or "package_source" in output


# ---------------------------------------------------------------------------
# start-headless-blender (mocked)
# ---------------------------------------------------------------------------

def test_start_headless_blender_shows_command():
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    mock_proc = MagicMock()
    mock_proc.pid = 42

    with patch("benchmark.mcp.headless.launcher._find_blender", return_value="blender"), \
         patch("benchmark.mcp.headless.launcher.subprocess.Popen", return_value=mock_proc), \
         redirect_stdout(buf):
        try:
            main(["start-headless-blender"])
        except SystemExit:
            pass

    output = buf.getvalue()
    assert "starting" in output or "blender" in output.lower()


# ---------------------------------------------------------------------------
# Real MCP tests (require running server)
# ---------------------------------------------------------------------------

@pytest.mark.mcp
@pytest.mark.mcp_e2e
def test_smoke_with_real_blender():
    """Requires Blender running with blender-mcp addon."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "smoke.json"
        run_cli(
            "--config", "configs/mcp/inspection_enabled.yaml",
            "smoke",
            "--output", str(out),
            expect_exit=0,
        )
        data = json.loads(out.read_text())
        assert data["status"] == "PASS"
        assert data["checks"]["blender_socket_available"] is True


@pytest.mark.mcp
@pytest.mark.mcp_e2e
def test_check_with_real_blender():
    """Requires Blender running with blender-mcp addon."""
    result = run_cli("check", expect_exit=0)
    assert result["status"] == "OK"
