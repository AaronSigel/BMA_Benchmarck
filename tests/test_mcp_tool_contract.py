"""Tests for benchmark.mcp.tool_contract (no Blender, no MCP server required)."""
from __future__ import annotations

import pytest

from benchmark.mcp.tool_contract import (
    TOOL_CONTRACT_MAP,
    TOOL_CONTRACTS,
    ToolCategory,
    ToolContract,
    ToolParameter,
)


def test_tool_contracts_not_empty():
    assert len(TOOL_CONTRACTS) > 0


def test_tool_contract_map_keys_match_contracts():
    assert set(TOOL_CONTRACT_MAP.keys()) == {tc.name for tc in TOOL_CONTRACTS}


def test_get_scene_info_is_benchmark_safe():
    tc = TOOL_CONTRACT_MAP["get_scene_info"]
    assert tc.benchmark_safe is True
    assert tc.requires_python is False
    assert tc.requires_external_network is False


def test_get_bma_profile_info_is_benchmark_safe():
    tc = TOOL_CONTRACT_MAP["get_bma_profile_info"]
    assert tc.benchmark_safe is True


def test_execute_blender_code_requires_python():
    tc = TOOL_CONTRACT_MAP["execute_blender_code"]
    assert tc.requires_python is True
    assert tc.benchmark_safe is False


def test_asset_tools_require_external_network():
    asset_tools = [
        "get_polyhaven_status",
        "search_polyhaven_assets",
        "download_polyhaven_asset",
        "get_sketchfab_status",
        "search_sketchfab_models",
        "download_sketchfab_model",
        "get_hyper3d_status",
        "generate_hyper3d_model_via_text",
        "get_hunyuan3d_status",
        "generate_hunyuan3d_model",
    ]
    for name in asset_tools:
        tc = TOOL_CONTRACT_MAP[name]
        assert tc.requires_external_network is True, f"{name} should require external network"
        assert tc.benchmark_safe is False, f"{name} should not be benchmark_safe"


def test_bma_star_tools_are_benchmark_safe():
    bma_tools = [
        "bma_get_scene_info",
        "bma_create_object",
        "bma_set_transform",
        "bma_set_material",
        "bma_create_light",
        "bma_create_camera",
        "bma_export_scene",
    ]
    for name in bma_tools:
        tc = TOOL_CONTRACT_MAP[name]
        assert tc.benchmark_safe is True, f"{name} should be benchmark_safe"
        assert tc.requires_python is False, f"{name} must not require Python"
        assert tc.requires_external_network is False, f"{name} must not require external network"


def test_tool_categories_populated():
    categories = {tc.category for tc in TOOL_CONTRACTS}
    assert ToolCategory.INSPECTION in categories
    assert ToolCategory.PYTHON in categories
    assert ToolCategory.ASSET in categories


def test_tool_parameter_required_defaults():
    p = ToolParameter(name="foo", type="str")
    assert p.required is True
    assert p.default is None


def test_tool_parameter_optional():
    p = ToolParameter(name="size", type="int", required=False, default=800)
    assert p.required is False
    assert p.default == 800


def test_tool_contract_required_params():
    tc = TOOL_CONTRACT_MAP["get_object_info"]
    required = tc.required_params
    assert any(p.name == "object_name" for p in required)


def test_tool_contract_optional_params():
    tc = TOOL_CONTRACT_MAP["get_viewport_screenshot"]
    optional = tc.optional_params
    assert any(p.name == "max_size" for p in optional)


def test_all_tool_names_are_unique():
    names = [tc.name for tc in TOOL_CONTRACTS]
    assert len(names) == len(set(names)), "Duplicate tool names in TOOL_CONTRACTS"


def test_no_contract_has_both_python_and_safe():
    for tc in TOOL_CONTRACTS:
        if tc.requires_python:
            assert not tc.benchmark_safe, (
                f"{tc.name} cannot be both requires_python=True and benchmark_safe=True"
            )
