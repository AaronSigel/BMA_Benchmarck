import json

from benchmark.agent.tool_context import ToolSchemaProvider
from benchmark.mcp.profiles import McpProfile
from benchmark.mcp.tool_contract import TOOL_CONTRACT_MAP


def test_tool_schema_provider_filters_no_python_tools() -> None:
    provider = ToolSchemaProvider()

    tools = provider.get_tools_for_profile(McpProfile.NO_PYTHON)
    names = {tool.name for tool in tools}

    assert "execute_blender_code" not in names
    assert "get_scene_info" in names


def test_tool_schema_provider_filters_asset_tools_from_minimal() -> None:
    provider = ToolSchemaProvider()

    tools = provider.get_tools_for_profile("minimal")
    names = {tool.name for tool in tools}

    assert "download_polyhaven_asset" not in names
    assert "search_sketchfab_models" not in names


def test_tool_schema_provider_includes_python_tool_only_when_profile_allows_it() -> None:
    provider = ToolSchemaProvider()

    python_enabled = {tool.name for tool in provider.get_tools_for_profile("python_enabled")}
    full = {tool.name for tool in provider.get_tools_for_profile("full")}

    assert "execute_blender_code" in python_enabled
    assert "execute_blender_code" in full


def test_openai_tool_schema_is_json_compatible_for_mcp_tool_contract() -> None:
    provider = ToolSchemaProvider()
    contract = TOOL_CONTRACT_MAP["get_object_info"]

    schema = provider.to_openai_tool_schema(contract)

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "get_object_info"
    assert schema["function"]["parameters"]["properties"]["object_name"]["type"] == "string"
    assert schema["function"]["parameters"]["required"] == ["object_name"]
    json.dumps(schema)


def test_prompt_tool_description_uses_mcp_tool_contract() -> None:
    provider = ToolSchemaProvider()
    contract = TOOL_CONTRACT_MAP["bma_create_object"]

    description = provider.to_prompt_tool_description(contract)

    assert "bma_create_object" in description
    assert "type: str" in description
    assert "location: list[float]" in description


def test_json_action_schema_is_json_compatible() -> None:
    provider = ToolSchemaProvider()
    contracts = [
        TOOL_CONTRACT_MAP["get_scene_info"],
        TOOL_CONTRACT_MAP["get_object_info"],
    ]

    schema = provider.to_json_action_schema(contracts)

    assert schema["properties"]["tool_name"]["enum"] == ["get_scene_info", "get_object_info"]
    assert schema["required"] == ["tool_name", "arguments"]
    json.dumps(schema)


def test_list_tool_schemas_excludes_disabled_tools() -> None:
    provider = ToolSchemaProvider()

    schemas = provider.list_tool_schemas("no_python")
    names = {schema["function"]["name"] for schema in schemas}

    assert "execute_blender_code" not in names
    assert "get_scene_info" in names
    json.dumps(schemas)
