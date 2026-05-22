from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Any

from benchmark.agent.errors import AgentRuntimeError
from benchmark.agent.llm.base import LlmClient, LlmMessage, LlmResponse, LlmToolCall, parse_json_from_content
from benchmark.agent.models import AgentConfig, AgentStepType, AgentTrace, ToolCallStatus
from benchmark.agent.prompts import PromptBuilder, _AGENT_HIDDEN_TOOLS
from benchmark.agent.tool_context import AgentToolContext, ToolSchemaProvider
from benchmark.agent.tool_executor import ToolExecutor, _TOOL_ALIASES
from benchmark.mcp.profiles import McpProfile, get_allowed_tools


class DirectToolCallingStrategy:
    """Single-cycle strategy that lets the LLM choose tool calls directly."""

    def __init__(
        self,
        *,
        prompt_builder: PromptBuilder | None = None,
        tool_schema_provider: ToolSchemaProvider | None = None,
    ) -> None:
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.tool_schema_provider = tool_schema_provider or ToolSchemaProvider()

    def run(
        self,
        task: dict[str, Any],
        agent_config: AgentConfig,
        llm_client: LlmClient | None,
        tool_executor: ToolExecutor,
        tool_context: AgentToolContext,
        output_dir: Path,
    ) -> AgentTrace:
        if llm_client is None:
            raise AgentRuntimeError("Direct tool-calling strategy requires an LLM client")

        started_at = datetime.datetime.now(datetime.timezone.utc)
        task_id = str(task.get("id") or tool_context.task_id or "unknown")
        trace = AgentTrace(
            run_id=tool_context.run_id or str(uuid.uuid4()),
            task_id=task_id,
            agent_id=agent_config.agent_id,
            strategy=agent_config.strategy,
            model=agent_config.llm.model if agent_config.llm else None,
            started_at=started_at,
            metadata={"output_dir": str(output_dir)},
        )

        tool_contracts = [
            contract for contract in self.tool_schema_provider.get_tools_for_profile(agent_config.mcp_profile)
            if contract.name not in _AGENT_HIDDEN_TOOLS
        ]
        tool_contract_dicts = [contract.model_dump(mode="json") for contract in tool_contracts]
        allowed_tool_names = _allowed_tool_names_for_profile(agent_config.mcp_profile)
        tool_schemas = [
            self.tool_schema_provider.to_openai_tool_schema(contract)
            for contract in tool_contracts
        ]
        messages = [
            LlmMessage(
                role="system",
                content=self.prompt_builder.build_system_prompt(
                    agent_config,
                    agent_config.mcp_profile,
                    tool_contract_dicts,
                ),
            ),
            LlmMessage(role="user", content=self.prompt_builder.build_task_prompt(task)),
        ]

        llm_started_at = datetime.datetime.now(datetime.timezone.utc)
        response = llm_client.complete(
            messages,
            tools=tool_schemas,
            timeout_sec=agent_config.step_timeout_sec,
        )
        llm_finished_at = datetime.datetime.now(datetime.timezone.utc)
        parse_result = _extract_tool_calls(response)
        trace = trace.add_step(
            AgentStepType.LLM_CALL,
            raw_llm_response=_response_to_raw(response),
            started_at=llm_started_at,
            finished_at=llm_finished_at,
            duration_sec=(llm_finished_at - llm_started_at).total_seconds(),
            metadata={
                "tool_schema_count": len(tool_schemas),
                **({"direct_parse_format": parse_result.format_hint} if parse_result.format_hint else {}),
            },
        )

        tool_calls = parse_result.tool_calls
        if not tool_calls:
            retry_started = datetime.datetime.now(datetime.timezone.utc)
            retry_response = llm_client.complete(
                [
                    *messages,
                    LlmMessage(role="assistant", content=response.content or ""),
                    LlmMessage(
                        role="user",
                        content="Return exactly one tool call or one JSON action.",
                    ),
                ],
                tools=tool_schemas,
                timeout_sec=agent_config.step_timeout_sec,
            )
            retry_finished = datetime.datetime.now(datetime.timezone.utc)
            retry_parse_result = _extract_tool_calls(retry_response)
            trace = trace.add_step(
                AgentStepType.LLM_CALL,
                raw_llm_response=_response_to_raw(retry_response),
                started_at=retry_started,
                finished_at=retry_finished,
                duration_sec=(retry_finished - retry_started).total_seconds(),
                metadata={
                    "direct_no_action_retry": True,
                    **(
                        {"direct_parse_format": retry_parse_result.format_hint}
                        if retry_parse_result.format_hint
                        else {}
                    ),
                },
            )
            response = retry_response
            parse_result = retry_parse_result
            tool_calls = parse_result.tool_calls

        max_tool_steps = max(agent_config.max_steps - len(trace.steps) - 1, 0)
        error_message = None
        structured_error = None
        executed_count = 0
        for tool_call in tool_calls[:max_tool_steps]:
            validation_error = _validate_tool_call(tool_call, allowed_tool_names)
            if validation_error is not None:
                from benchmark.runner.controlled_errors import controlled_error_payload

                error_message = validation_error
                structured_error = controlled_error_payload(
                    validation_error,
                    source="agent",
                    failure_stage="tool_call",
                )
                trace = trace.add_step(
                    AgentStepType.ERROR,
                    error=validation_error,
                    tool_name=tool_call.name or None,
                    tool_arguments=tool_call.arguments,
                    metadata={
                        "original_action": tool_call.raw,
                        "raw_llm_response": _response_to_raw(response),
                        "direct_parse_format": parse_result.format_hint,
                        "direct_error_type": "InvalidToolCall",
                    },
                )
                break

            tool_result = tool_executor.call_tool(tool_call.name, tool_call.arguments)
            executed_count += 1
            trace = trace.add_step(
                AgentStepType.TOOL_CALL,
                action="call_tool",
                tool_name=tool_result.name,
                tool_arguments=tool_call.arguments,
                observation=tool_result.result,
                error=tool_result.error,
                started_at=tool_result.started_at,
                finished_at=tool_result.finished_at,
                duration_sec=tool_result.duration_sec,
                metadata={"tool_call_id": tool_call.id},
            )
            if tool_result.status != ToolCallStatus.SUCCEEDED:
                error_message = tool_result.error or f"Tool failed: {tool_result.name}"
                break

        if len(tool_calls) > max_tool_steps and error_message is None:
            error_message = "Direct tool-calling strategy reached max_steps before executing all tool calls"

        success = error_message is None
        final_message = None
        if success and tool_calls:
            final_message = f"Executed {executed_count} tool call(s)."
        if not tool_calls:
            final_message = "DirectNoAction: no tool call or JSON action returned by LLM."
            success = False
            error_message = "DirectNoAction"

        if error_message == "DirectNoAction":
            from benchmark.runner.controlled_errors import controlled_error_payload

            structured_error = controlled_error_payload("DirectNoAction", source="agent")
            trace = trace.add_step(
                AgentStepType.ERROR,
                observation=final_message,
                error=error_message,
                metadata={"direct_error_type": "DirectNoAction"},
            )
        elif error_message is None:
            trace = trace.add_step(
                AgentStepType.FINAL,
                observation=final_message,
            )
        elif structured_error is None:
            trace = trace.add_step(
                AgentStepType.ERROR,
                observation=final_message,
                error=error_message,
            )

        finished_at = datetime.datetime.now(datetime.timezone.utc)
        return trace.model_copy(
            update={
                "success": success,
                "error": error_message,
                "final_message": final_message,
                "finished_at": finished_at,
                "duration_sec": (finished_at - started_at).total_seconds(),
                "structured_error": structured_error,
            }
        )


class _ExtractToolCallsResult:
    __slots__ = ("tool_calls", "format_hint")

    def __init__(self, tool_calls: list[LlmToolCall], *, format_hint: str | None = None) -> None:
        self.tool_calls = tool_calls
        self.format_hint = format_hint


def _extract_tool_calls(response: LlmResponse) -> _ExtractToolCallsResult:
    if response.tool_calls:
        repaired: list[LlmToolCall] = []
        for index, call in enumerate(response.tool_calls):
            normalized = _repair_native_tool_call(call, fallback_id=f"native-{index}")
            if normalized is not None:
                repaired.append(normalized)
        if repaired:
            return _ExtractToolCallsResult(repaired, format_hint="native_tool_calls")

    parsed = response.json_action()
    format_hint = "json_content"
    if parsed is None:
        parsed = parse_json_from_content(response.content)
        format_hint = "parsed_json_content" if parsed is not None else None
    if parsed is None:
        return _ExtractToolCallsResult([])

    actions, hint = _normalize_json_action_payload(parsed)
    calls: list[LlmToolCall] = []
    for index, action in enumerate(actions):
        tool_call = _tool_call_from_action(action, fallback_id=f"json-{index}")
        if tool_call is not None:
            calls.append(tool_call)
    return _ExtractToolCallsResult(calls, format_hint=hint or format_hint)


def _repair_native_tool_call(call: LlmToolCall, *, fallback_id: str) -> LlmToolCall | None:
    name = (call.name or "").strip()
    if name.startswith("{") or name.startswith("["):
        parsed = parse_json_from_content(name)
        if isinstance(parsed, dict):
            actions, _ = _normalize_json_action_payload(parsed)
            if actions:
                repaired = _tool_call_from_action(actions[0], fallback_id=fallback_id)
                if repaired is not None:
                    return repaired
    if not name or name.lower() == "none":
        return None
    return call


def _normalize_json_action_payload(obj: Any) -> tuple[list[dict[str, Any]], str | None]:
    if not isinstance(obj, dict):
        return [], None

    for list_key, hint in (("tool_calls", "json_tool_calls"), ("actions", "json_actions")):
        items = obj.get(list_key)
        if isinstance(items, list):
            actions = [item for item in items if isinstance(item, dict)]
            return actions, hint

    nested = obj.get("action")
    if isinstance(nested, dict):
        return [nested], "nested_action"

    if _action_has_tool_name(obj):
        return [obj], "flat_action"

    return [], None


def _action_has_tool_name(action: dict[str, Any]) -> bool:
    return _extract_tool_name(action) is not None


def _extract_tool_name(action: dict[str, Any]) -> str | None:
    for key in ("tool_name", "name", "tool"):
        value = action.get(key)
        if isinstance(value, str) and value.strip() and value.strip().lower() != "none":
            return value.strip()

    function = action.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name.strip() and name.strip().lower() != "none":
            return name.strip()
    return None


def _tool_call_from_action(action: dict[str, Any], *, fallback_id: str) -> LlmToolCall | None:
    name = _extract_tool_name(action)
    if name is None:
        return None

    arguments = action.get("arguments")
    if arguments is None:
        function = action.get("function")
        if isinstance(function, dict):
            arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            import json

            parsed_args = json.loads(arguments)
            arguments = parsed_args if isinstance(parsed_args, dict) else {}
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}

    return LlmToolCall(
        id=str(action.get("id") or fallback_id),
        name=name,
        arguments=arguments,
        raw=action,
    )


def _allowed_tool_names_for_profile(mcp_profile: str) -> set[str] | None:
    try:
        profile = McpProfile(mcp_profile)
    except ValueError:
        profile = McpProfile.FULL
    allowed = get_allowed_tools(profile)
    if allowed is None:
        return None
    return set(allowed)


def _validate_tool_call(tool_call: LlmToolCall, allowed_tool_names: set[str] | None) -> str | None:
    name = (tool_call.name or "").strip()
    if not name or name.lower() == "none":
        return f"InvalidToolCall: empty or missing tool name (raw={tool_call.raw!r})"
    resolved_name = _TOOL_ALIASES.get(name, name)
    if allowed_tool_names is not None and resolved_name not in allowed_tool_names and name not in allowed_tool_names:
        return f"InvalidToolCall: tool {name!r} is not allowed in this profile"
    if not isinstance(tool_call.arguments, dict):
        return f"InvalidToolCall: arguments for {name!r} must be an object"
    return None


def _response_to_raw(response: LlmResponse) -> dict[str, Any]:
    return response.model_dump(mode="json", exclude_none=True)
