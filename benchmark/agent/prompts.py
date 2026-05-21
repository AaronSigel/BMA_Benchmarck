from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from benchmark.agent.models import AgentConfig

_SECRET_KEYS = frozenset({"api_key", "api_key_env", "authorization", "token", "secret", "password"})
_PYTHON_RESTRICTED_PROFILES = frozenset({"minimal", "no_python", "inspection_enabled"})
_EXTERNAL_ASSET_TOOLS = frozenset(
    {"download_asset", "search_assets", "import_external_asset", "load_external_asset"}
)
_AGENT_HIDDEN_TOOLS = frozenset({"get_bma_profile_info", "get_scene_info", "get_object_info"})


class PromptContext(BaseModel):
    task_id: str | None = None
    task: dict[str, Any] = Field(default_factory=dict)
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptBuilder:
    """Builds provider-neutral prompts for agent strategies."""

    def build_system_prompt(
        self,
        agent_config: AgentConfig,
        tool_profile: str | None = None,
        tool_contracts: list[dict[str, Any]] | None = None,
    ) -> str:
        profile = tool_profile or agent_config.mcp_profile
        contracts = _filter_tool_contracts(agent_config, profile, tool_contracts or [])
        tool_names = [str(contract.get("name")) for contract in contracts if contract.get("name")]
        tools_json = json.dumps(_strip_secrets(contracts), indent=2, sort_keys=True)

        lines = [
            "You are an agent solving Blender benchmark tasks through MCP tools.",
            "Goal: modify the Blender scene to satisfy BenchmarkTask.prompt.",
            f"Strategy: {agent_config.strategy.value}.",
            f"MCP tool profile: {profile}.",
            "Use only the MCP tools listed below; do not invent tools.",
            "Do not use generic tool names such as create_object, assign_material, get_scene_info, get_object_info, get_bma_profile_info, or export_scene.",
            f"Allowed tools: {', '.join(tool_names) if tool_names else 'none'}.",
            f"Tool contracts:\n{tools_json}",
            "When a task specifies object dimensions, use the dimensions parameter; do not approximate dimensions with scale.",
            "When assigning materials, prefer bma_assign_material; include material_name, base_color, roughness, and metallic when specified.",
            "When a camera must look at a target point, use target with bma_create_camera or bma_create_camera_look_at instead of manual Euler rotation.",
            "Return tool_calls when the API supports them, otherwise return a JSON action in content.",
            (
                "Fallback JSON action format: "
                '{"tool_name": "<tool>", "arguments": {"key": "value"}}.'
            ),
            "After modifying the scene, use inspection tools when they are available.",
        ]
        if profile in _PYTHON_RESTRICTED_PROFILES or not agent_config.allow_python_tools:
            lines.append("Do not use execute_blender_code or any Python execution tool.")
        if not _allows_external_assets(profile):
            lines.append("Do not use external asset tools or fetch assets from outside the MCP profile.")
        if agent_config.system_prompt_template:
            lines.append(_strip_secret_text(agent_config.system_prompt_template))
        return "\n".join(lines)

    def build_task_prompt(self, task: dict[str, Any] | PromptContext) -> str:
        task_data = task.task if isinstance(task, PromptContext) else task
        task_id = task.task_id if isinstance(task, PromptContext) else task_data.get("id")
        prompt = task_data.get("prompt") or task_data.get("description") or ""
        lines = []
        if task_id:
            lines.append(f"Task ID: {task_id}")
        lines.append(f"BenchmarkTask.prompt: {prompt}")
        stripped_task = _strip_secrets(task_data)
        lines.append(f"Task data:\n{json.dumps(stripped_task, indent=2, sort_keys=True)}")
        return "\n".join(lines)

    def build_react_prompt_context(
        self,
        task: dict[str, Any],
        observations: list[str | dict[str, Any]],
    ) -> str:
        lines = [
            self.build_task_prompt(task),
            "Use a ReAct loop: reason briefly, call one allowed tool, then use the observation.",
            "Observations:",
        ]
        for index, observation in enumerate(observations):
            lines.append(f"{index}: {_format_value(_strip_secrets(observation))}")
        return "\n".join(lines)

    def build_plan_prompt(self, task: dict[str, Any]) -> str:
        return "\n".join(
            [
                self.build_task_prompt(task),
                "Create a concise execution plan using only allowed MCP tools.",
                "Plan steps must be actionable and aimed at modifying the Blender scene.",
                "Return only JSON. Do not wrap it in Markdown and do not include explanatory text.",
                (
                    'Required schema: {"plan":[{"step":1,"description":"...",'
                    '"tool":"allowed_tool_name","arguments":{}}]}.'
                ),
                "Use integer step numbers starting at 1. Every step must include description, tool, and arguments.",
            ]
        )

    def build_tool_result_message(self, tool_name: str, result: Any) -> str:
        return "\n".join(
            [
                f"Tool result for {tool_name}:",
                _format_value(_strip_secrets(result)),
            ]
        )


def build_system_prompt(
    agent_config: AgentConfig,
    tool_profile: str | None = None,
    tool_contracts: list[dict[str, Any]] | None = None,
) -> str:
    return PromptBuilder().build_system_prompt(agent_config, tool_profile, tool_contracts)


def build_task_prompt(task: dict[str, Any] | PromptContext) -> str:
    return PromptBuilder().build_task_prompt(task)


def build_react_prompt_context(
    task: dict[str, Any],
    observations: list[str | dict[str, Any]],
) -> str:
    return PromptBuilder().build_react_prompt_context(task, observations)


def build_plan_prompt(task: dict[str, Any]) -> str:
    return PromptBuilder().build_plan_prompt(task)


def build_tool_result_message(tool_name: str, result: Any) -> str:
    return PromptBuilder().build_tool_result_message(tool_name, result)


def _filter_tool_contracts(
    agent_config: AgentConfig,
    profile: str,
    tool_contracts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    filtered = []
    for contract in tool_contracts:
        name = contract.get("name")
        if name in _AGENT_HIDDEN_TOOLS:
            continue
        if name == "execute_blender_code" and (
            profile in _PYTHON_RESTRICTED_PROFILES or not agent_config.allow_python_tools
        ):
            continue
        if isinstance(name, str) and name in _EXTERNAL_ASSET_TOOLS and not _allows_external_assets(profile):
            continue
        filtered.append(contract)
    return filtered


def _allows_external_assets(profile: str) -> bool:
    return profile in {"full"}


def _strip_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[redacted]" if _is_secret_key(str(key)) else _strip_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_strip_secrets(item) for item in value]
    return value


def _strip_secret_text(value: str) -> str:
    lines = []
    for line in value.splitlines():
        lowered = line.lower()
        if any(secret in lowered for secret in _SECRET_KEYS):
            lines.append("[redacted]")
        else:
            lines.append(line)
    return "\n".join(lines)


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(secret in lowered for secret in _SECRET_KEYS)


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True)
