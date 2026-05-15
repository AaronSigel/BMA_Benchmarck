"""Tests for benchmark.mcp.config (no Blender, no MCP server required)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from benchmark.mcp.config import (
    McpServerConfig,
    build_mcp_env,
    config_from_env,
    dump_mcp_config,
    load_mcp_config,
    load_named_config,
)
from benchmark.mcp.errors import McpConfigError


def make_config(**overrides) -> McpServerConfig:
    defaults = dict(
        server_distribution="upstream",
        blender_host="localhost",
        blender_port=9876,
        profile="full",
        disable_telemetry=True,
    )
    defaults.update(overrides)
    return McpServerConfig(**defaults)


def test_default_config_values():
    cfg = McpServerConfig(
        artifacts_dir=None,
        output_dir=None,
    ) if False else McpServerConfig()
    assert cfg.blender_host == "localhost"
    assert cfg.blender_port == 9876
    assert cfg.disable_telemetry is True
    assert cfg.profile == "full"
    assert cfg.server_distribution == "upstream"


def test_load_mcp_config_from_yaml(tmp_path):
    data = {
        "server_distribution": "upstream",
        "blender_host": "127.0.0.1",
        "blender_port": 9999,
        "profile": "minimal",
        "disable_telemetry": True,
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")

    cfg = load_mcp_config(p)
    assert cfg.blender_host == "127.0.0.1"
    assert cfg.blender_port == 9999
    assert cfg.profile == "minimal"


def test_load_mcp_config_missing_file(tmp_path):
    with pytest.raises(McpConfigError, match="not found"):
        load_mcp_config(tmp_path / "nonexistent.yaml")


def test_load_mcp_config_invalid_yaml(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("not: valid: yaml: [", encoding="utf-8")
    with pytest.raises(McpConfigError):
        load_mcp_config(p)


def test_dump_and_reload_roundtrip(tmp_path):
    cfg = make_config(blender_port=8888, profile="minimal")
    p = tmp_path / "out.yaml"
    dump_mcp_config(cfg, p)
    restored = load_mcp_config(p)
    assert restored.blender_port == 8888
    assert restored.profile == "minimal"


def test_build_mcp_env_includes_required_keys():
    cfg = make_config(profile="no_python", blender_host="host", blender_port=1234)
    env = build_mcp_env(cfg)
    assert env["BMA_MCP_PROFILE"] == "no_python"
    assert env["BMA_SOCKET_HOST"] == "host"
    assert env["BMA_SOCKET_PORT"] == "1234"
    assert env["DISABLE_TELEMETRY"] == "true"


def test_build_mcp_env_telemetry_off_by_default():
    cfg = make_config(disable_telemetry=True)
    env = build_mcp_env(cfg)
    assert env["DISABLE_TELEMETRY"] == "true"


def test_build_mcp_env_telemetry_opt_in():
    cfg = make_config(disable_telemetry=False)
    env = build_mcp_env(cfg)
    assert env["DISABLE_TELEMETRY"] == "false"


def test_build_mcp_env_package_source():
    cfg = make_config(server_distribution="local", package_source="./vendor/bma")
    env = build_mcp_env(cfg)
    assert env.get("BMA_PACKAGE_SOURCE") == "./vendor/bma"


def test_load_named_config_inspection_enabled():
    cfg = load_named_config("inspection_enabled")
    assert cfg.profile == "inspection_enabled"
    assert cfg.disable_telemetry is True


def test_load_named_config_minimal():
    cfg = load_named_config("minimal")
    assert cfg.profile == "minimal"


def test_config_from_env_uses_defaults(monkeypatch):
    monkeypatch.delenv("BMA_MCP_PROFILE", raising=False)
    monkeypatch.delenv("BMA_SOCKET_HOST", raising=False)
    monkeypatch.delenv("BMA_SOCKET_PORT", raising=False)
    monkeypatch.delenv("BMA_ENABLE_TELEMETRY", raising=False)
    cfg = config_from_env()
    assert cfg.profile == "full"
    assert cfg.blender_host == "localhost"
    assert cfg.blender_port == 9876
    assert cfg.disable_telemetry is True


def test_config_from_env_reads_env_vars(monkeypatch):
    monkeypatch.setenv("BMA_MCP_PROFILE", "minimal")
    monkeypatch.setenv("BMA_SOCKET_HOST", "10.0.0.1")
    monkeypatch.setenv("BMA_SOCKET_PORT", "4321")
    cfg = config_from_env()
    assert cfg.profile == "minimal"
    assert cfg.blender_host == "10.0.0.1"
    assert cfg.blender_port == 4321


def test_blender_host_must_not_be_empty():
    with pytest.raises(Exception):
        McpServerConfig(blender_host="  ")


def test_profile_must_not_be_empty():
    with pytest.raises(Exception):
        McpServerConfig(profile="")
