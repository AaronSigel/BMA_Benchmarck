"""Tests for benchmark.mcp.profiles (no Blender, no MCP server required)."""
from __future__ import annotations

import pytest

from benchmark.mcp.profiles import (
    McpProfile,
    _ALL_TOOLS,
    _BMA_SAFE_TOOLS,
    _EXTERNAL_ASSET_TOOLS,
    _PYTHON_TOOLS,
    get_allowed_tools,
    is_tool_allowed,
    profile_from_env,
)


def test_all_profiles_exist():
    values = {p.value for p in McpProfile}
    assert values == {"minimal", "no_python", "python_enabled", "inspection_enabled", "full"}


def test_minimal_allows_bma_safe_tools():
    for tool in _BMA_SAFE_TOOLS:
        assert is_tool_allowed(tool, McpProfile.MINIMAL), f"{tool} should be allowed in minimal"


def test_minimal_blocks_execute_blender_code():
    assert not is_tool_allowed("execute_blender_code", McpProfile.MINIMAL)


def test_minimal_blocks_external_asset_tools():
    for tool in _EXTERNAL_ASSET_TOOLS:
        assert not is_tool_allowed(tool, McpProfile.MINIMAL), f"{tool} must be blocked in minimal"


def test_no_python_blocks_execute_blender_code():
    assert not is_tool_allowed("execute_blender_code", McpProfile.NO_PYTHON)


def test_no_python_blocks_external_asset_tools():
    for tool in _EXTERNAL_ASSET_TOOLS:
        assert not is_tool_allowed(tool, McpProfile.NO_PYTHON)


def test_python_enabled_allows_execute_blender_code():
    assert is_tool_allowed("execute_blender_code", McpProfile.PYTHON_ENABLED)


def test_python_enabled_blocks_external_assets():
    for tool in _EXTERNAL_ASSET_TOOLS:
        assert not is_tool_allowed(tool, McpProfile.PYTHON_ENABLED)


def test_inspection_enabled_allows_viewport_screenshot():
    assert is_tool_allowed("get_viewport_screenshot", McpProfile.INSPECTION_ENABLED)


def test_inspection_enabled_blocks_python():
    assert not is_tool_allowed("execute_blender_code", McpProfile.INSPECTION_ENABLED)


def test_full_allows_all_registered_tools():
    assert get_allowed_tools(McpProfile.FULL) is None  # None = unrestricted


def test_full_allows_every_known_tool():
    for tool in _ALL_TOOLS:
        assert is_tool_allowed(tool, McpProfile.FULL), f"{tool} should be allowed in full"


def test_is_tool_allowed_unknown_tool_in_full():
    # Unknown tools are allowed in full (unrestricted)
    assert is_tool_allowed("totally_unknown_tool", McpProfile.FULL)


def test_is_tool_allowed_unknown_tool_in_minimal():
    assert not is_tool_allowed("totally_unknown_tool", McpProfile.MINIMAL)


def test_profile_from_env_valid():
    assert profile_from_env("minimal") == McpProfile.MINIMAL
    assert profile_from_env("full") == McpProfile.FULL
    assert profile_from_env("no_python") == McpProfile.NO_PYTHON


def test_profile_from_env_none_returns_full():
    assert profile_from_env(None) == McpProfile.FULL


def test_profile_from_env_unknown_falls_back_to_full():
    assert profile_from_env("nonexistent_profile") == McpProfile.FULL


def test_bma_safe_tools_not_in_external_assets():
    assert _BMA_SAFE_TOOLS.isdisjoint(_EXTERNAL_ASSET_TOOLS)


def test_python_tools_not_in_bma_safe():
    assert _PYTHON_TOOLS.isdisjoint(_BMA_SAFE_TOOLS)
