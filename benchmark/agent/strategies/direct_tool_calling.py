from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Any

from benchmark.agent.errors import AgentRuntimeError
from benchmark.agent.llm.base import LlmClient, LlmMessage, LlmResponse, LlmToolCall
from benchmark.agent.models import AgentConfig, AgentStepType, AgentTrace, ToolCallStatus
from benchmark.agent.prompts import PromptBuilder, _AGENT_HIDDEN_TOOLS
from benchmark.agent.tool_context import AgentToolContext, ToolSchemaProvider
from benchmark.agent.tool_executor import ToolExecutor


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
        trace = trace.add_step(
            AgentStepType.LLM_CALL,
            raw_llm_response=_response_to_raw(response),
            started_at=llm_started_at,
            finished_at=llm_finished_at,
            duration_sec=(llm_finished_at - llm_started_at).total_seconds(),
            metadata={"tool_schema_count": len(tool_schemas)},
        )

        tool_calls = _extract_tool_calls(response)
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
            trace = trace.add_step(
                AgentStepType.LLM_CALL,
                raw_llm_response=_response_to_raw(retry_response),
                started_at=retry_started,
                finished_at=retry_finished,
                duration_sec=(retry_finished - retry_started).total_seconds(),
                metadata={"direct_no_action_retry": True},
            )
            response = retry_response
            tool_calls = _extract_tool_calls(response)

        max_tool_steps = max(agent_config.max_steps - len(trace.steps) - 1, 0)
        error_message = None
        executed_count = 0
        for tool_call in tool_calls[:max_tool_steps]:
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

        trace = trace.add_step(
            AgentStepType.FINAL if success else AgentStepType.ERROR,
            observation=final_message,
            error=error_message,
            metadata={"direct_error_type": "DirectNoAction"} if error_message == "DirectNoAction" else {},
        )
        finished_at = datetime.datetime.now(datetime.timezone.utc)
        structured_error = None
        if error_message == "DirectNoAction":
            from benchmark.runner.controlled_errors import controlled_error_payload

            structured_error = controlled_error_payload("DirectNoAction", source="agent")
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


def _extract_tool_calls(response: LlmResponse) -> list[LlmToolCall]:
    if response.tool_calls:
        return response.tool_calls

    action = response.json_action()
    if not action:
        return []
    if isinstance(action.get("tool_calls"), list):
        calls = []
        for index, item in enumerate(action["tool_calls"]):
            if isinstance(item, dict):
                calls.append(_tool_call_from_action(item, fallback_id=f"json-{index}"))
        return calls
    return [_tool_call_from_action(action, fallback_id="json-0")]


def _tool_call_from_action(action: dict[str, Any], *, fallback_id: str) -> LlmToolCall:
    name = action.get("tool_name") or action.get("name")
    arguments = action.get("arguments") or {}
    return LlmToolCall(
        id=str(action.get("id") or fallback_id),
        name=str(name),
        arguments=arguments if isinstance(arguments, dict) else {},
        raw=action,
    )


def _response_to_raw(response: LlmResponse) -> dict[str, Any]:
    return response.model_dump(mode="json", exclude_none=True)
