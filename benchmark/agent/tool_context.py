from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from benchmark.mcp.profiles import McpProfile
from benchmark.mcp.tool_contract import ToolContract, ToolParameter
from benchmark.mcp.tool_registry import McpToolRegistry


class AgentToolContext(BaseModel):
    run_id: str | None = None
    task_id: str | None = None
    artifacts_dir: Path | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "task_id")
    @classmethod
    def validate_optional_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("value must not be empty")
        return value

    def is_tool_allowed(self, tool_name: str) -> bool:
        return not self.allowed_tools or tool_name in self.allowed_tools


class ToolSchemaProvider:
    """Build provider-neutral tool schemas from benchmark MCP ToolContracts."""

    def __init__(self, registry: McpToolRegistry | None = None) -> None:
        self.registry = registry or McpToolRegistry()

    def get_tools_for_profile(self, profile: McpProfile | str) -> list[ToolContract]:
        mcp_profile = _coerce_profile(profile)
        return self.registry.list_for_profile(mcp_profile)

    def list_tool_schemas(self, profile: McpProfile | str = McpProfile.MINIMAL) -> list[dict[str, Any]]:
        return [
            self.to_openai_tool_schema(contract)
            for contract in self.get_tools_for_profile(profile)
        ]

    def to_openai_tool_schema(self, tool_contract: ToolContract) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for parameter in tool_contract.parameters:
            properties[parameter.name] = _parameter_to_json_schema(parameter)
            if parameter.required:
                required.append(parameter.name)

        return {
            "type": "function",
            "function": {
                "name": tool_contract.name,
                "description": tool_contract.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    def to_prompt_tool_description(self, tool_contract: ToolContract) -> str:
        parameters = ", ".join(
            (
                f"{parameter.name}: {parameter.type}"
                if parameter.required
                else f"{parameter.name}: {parameter.type} = {parameter.default!r}"
            )
            for parameter in tool_contract.parameters
        )
        suffix = f" Parameters: {parameters}." if parameters else ""
        return f"- {tool_contract.name}: {tool_contract.description}{suffix}"

    def to_json_action_schema(self, tool_contracts: list[ToolContract]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "enum": [contract.name for contract in tool_contracts],
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the selected tool.",
                },
            },
            "required": ["tool_name", "arguments"],
            "additionalProperties": False,
        }


def _coerce_profile(profile: McpProfile | str) -> McpProfile:
    if isinstance(profile, McpProfile):
        return profile
    return McpProfile(profile)


def _parameter_to_json_schema(parameter: ToolParameter) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": _json_schema_type(parameter.type),
        "description": parameter.description,
    }
    if not parameter.required and parameter.default is not None:
        schema["default"] = parameter.default
    return schema


def _json_schema_type(parameter_type: str) -> str:
    normalized = parameter_type.lower()
    if normalized in {"str", "string", "path"}:
        return "string"
    if normalized in {"int", "integer"}:
        return "integer"
    if normalized in {"float", "number"}:
        return "number"
    if normalized in {"bool", "boolean"}:
        return "boolean"
    if normalized.startswith("list") or normalized.startswith("tuple") or normalized == "array":
        return "array"
    if normalized.startswith("dict") or normalized == "object":
        return "object"
    return "string"
