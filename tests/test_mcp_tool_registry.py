"""Tests for benchmark.mcp.tool_registry (no Blender, no MCP server required)."""
from __future__ import annotations

import pytest

from benchmark.mcp.errors import ToolDisabledError, UnknownToolError
from benchmark.mcp.profiles import McpProfile
from benchmark.mcp.tool_registry import McpToolRegistry


@pytest.fixture()
def registry() -> McpToolRegistry:
    return McpToolRegistry()


def test_registry_has_contracts(registry):
    assert len(registry) > 0


def test_get_contract_known_tool(registry):
    tc = registry.get_contract("get_scene_info")
    assert tc.name == "get_scene_info"


def test_get_contract_unknown_raises(registry):
    with pytest.raises(UnknownToolError):
        registry.get_contract("totally_nonexistent_tool")


def test_assert_tool_allowed_permitted(registry):
    tc = registry.assert_tool_allowed("get_scene_info", McpProfile.MINIMAL)
    assert tc.name == "get_scene_info"


def test_assert_tool_allowed_disabled_raises(registry):
    with pytest.raises(ToolDisabledError):
        registry.assert_tool_allowed("execute_blender_code", McpProfile.NO_PYTHON)


def test_assert_tool_allowed_unknown_raises(registry):
    with pytest.raises(UnknownToolError):
        registry.assert_tool_allowed("ghost_tool", McpProfile.FULL)


def test_is_allowed_returns_true_for_known_permitted(registry):
    assert registry.is_allowed("get_scene_info", McpProfile.MINIMAL) is True


def test_is_allowed_returns_false_for_disabled(registry):
    assert registry.is_allowed("execute_blender_code", McpProfile.MINIMAL) is False


def test_is_allowed_returns_false_for_unknown(registry):
    assert registry.is_allowed("ghost_tool", McpProfile.FULL) is False


def test_list_available_tools_minimal(registry):
    tools = registry.list_available_tools(McpProfile.MINIMAL)
    names = {tc.name for tc in tools}
    assert "get_scene_info" in names
    assert "execute_blender_code" not in names
    assert "get_polyhaven_status" not in names


def test_list_available_tools_no_python(registry):
    tools = registry.list_available_tools(McpProfile.NO_PYTHON)
    names = {tc.name for tc in tools}
    assert "execute_blender_code" not in names


def test_list_available_tools_full_includes_all(registry):
    full_tools = registry.list_available_tools(McpProfile.FULL)
    all_tools = registry.list_all()
    assert len(full_tools) == len(all_tools)


def test_list_disabled_tools_no_python_includes_execute(registry):
    disabled = registry.list_disabled_tools(McpProfile.NO_PYTHON)
    names = {tc.name for tc in disabled}
    assert "execute_blender_code" in names


def test_list_disabled_tools_full_is_empty(registry):
    disabled = registry.list_disabled_tools(McpProfile.FULL)
    assert disabled == []


def test_register_extra_contract(registry):
    from benchmark.mcp.tool_contract import ToolContract
    extra = ToolContract(name="custom_test_tool", description="test")
    registry.register(extra)
    assert registry.is_allowed("custom_test_tool", McpProfile.FULL) is True


def test_backward_compat_get_alias(registry):
    tc = registry.get("get_scene_info")
    assert tc.name == "get_scene_info"


def test_backward_compat_list_for_profile(registry):
    tools = registry.list_for_profile(McpProfile.MINIMAL)
    assert isinstance(tools, list)
    assert len(tools) > 0
