from __future__ import annotations

import datetime
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from benchmark.agent.errors import AgentRuntimeError

log = logging.getLogger(__name__)
from benchmark.agent.llm.base import LlmClient, LlmMessage, LlmResponse
from benchmark.agent.models import AgentConfig, AgentStepType, AgentTrace, ToolCallStatus
from benchmark.agent.prompts import PromptBuilder
from benchmark.agent.tool_context import AgentToolContext, ToolSchemaProvider
from benchmark.agent.tool_executor import ToolExecutor


class ReactStrategy:
    """Iterative ReAct strategy: thought/action, tool observation, repeat."""

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
            raise AgentRuntimeError("ReAct strategy requires an LLM client")

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

        tool_contracts = self.tool_schema_provider.get_tools_for_profile(agent_config.mcp_profile)
        tool_contract_dicts = [contract.model_dump(mode="json") for contract in tool_contracts]
        tool_schemas = [
            self.tool_schema_provider.to_openai_tool_schema(contract)
            for contract in tool_contracts
        ]
        observations: list[str | dict[str, Any]] = []
        system_message = LlmMessage(
            role="system",
            content=self.prompt_builder.build_system_prompt(
                agent_config,
                agent_config.mcp_profile,
                tool_contract_dicts,
            ),
        )

        final_message = None
        error_message = None
        success = False

        log.info("[react:%s] starting task=%s max_steps=%d profile=%s", trace.run_id[:8], task_id, agent_config.max_steps, agent_config.mcp_profile)

        while len(trace.steps) < agent_config.max_steps:
            step_num = len(trace.steps) + 1
            log.info("[react:%s] step %d/%d — calling LLM", trace.run_id[:8], step_num, agent_config.max_steps)
            messages = [
                system_message,
                LlmMessage(
                    role="user",
                    content=self.prompt_builder.build_react_prompt_context(task, observations),
                ),
            ]
            llm_started_at = datetime.datetime.now(datetime.timezone.utc)
            response = llm_client.complete(
                messages,
                tools=tool_schemas,
                timeout_sec=agent_config.step_timeout_sec,
            )
            llm_finished_at = datetime.datetime.now(datetime.timezone.utc)
            action = _parse_react_action(response)
            trace = trace.add_step(
                AgentStepType.LLM_CALL,
                thought=action.thought,
                raw_llm_response=response.model_dump(mode="json", exclude_none=True),
                started_at=llm_started_at,
                finished_at=llm_finished_at,
                duration_sec=(llm_finished_at - llm_started_at).total_seconds(),
                metadata={"tool_schema_count": len(tool_schemas)},
            )

            if action.final_answer is not None:
                log.info("[react:%s] step %d — final answer received", trace.run_id[:8], step_num)
                final_message = action.final_answer
                success = True
                trace = trace.add_step(AgentStepType.FINAL, observation=final_message)
                break

            if action.tool_name is None:
                error_message = "ReAct response did not include action or final_answer"
                trace = trace.add_step(AgentStepType.ERROR, error=error_message)
                break

            if len(trace.steps) >= agent_config.max_steps:
                error_message = "ReAct strategy reached max_steps before executing next action"
                break

            log.info("[react:%s] step %d — tool_call: %s args=%s", trace.run_id[:8], step_num, action.tool_name, list((action.arguments or {}).keys()))
            tool_result = tool_executor.call_tool(action.tool_name, action.arguments)
            observation = tool_result.result if tool_result.error is None else {"error": tool_result.error}
            if tool_result.error:
                log.warning("[react:%s] step %d — tool %s error: %s", trace.run_id[:8], step_num, action.tool_name, tool_result.error)
            else:
                log.info("[react:%s] step %d — tool %s ok (%.2fs)", trace.run_id[:8], step_num, action.tool_name, tool_result.duration_sec or 0)
            observations.append(
                {
                    "tool": tool_result.name,
                    "arguments": action.arguments,
                    "observation": observation,
                }
            )
            trace = trace.add_step(
                AgentStepType.TOOL_CALL,
                thought=action.thought,
                action="call_tool",
                tool_name=tool_result.name,
                tool_arguments=action.arguments,
                observation=observation,
                error=tool_result.error,
                started_at=tool_result.started_at,
                finished_at=tool_result.finished_at,
                duration_sec=tool_result.duration_sec,
            )
            if tool_result.status != ToolCallStatus.SUCCEEDED:
                error_message = tool_result.error or f"Tool failed: {tool_result.name}"
                trace = trace.add_step(AgentStepType.ERROR, observation=observation, error=error_message)
                break

        if final_message is None and error_message is None:
            error_message = "ReAct strategy reached max_steps"
            trace = trace.add_step(AgentStepType.ERROR, error=error_message)

        finished_at = datetime.datetime.now(datetime.timezone.utc)
        return trace.model_copy(
            update={
                "success": success,
                "error": error_message,
                "final_message": final_message,
                "finished_at": finished_at,
                "duration_sec": (finished_at - started_at).total_seconds(),
            }
        )


class _ReactAction:
    def __init__(
        self,
        *,
        thought: str | None = None,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        final_answer: str | None = None,
    ) -> None:
        self.thought = thought
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.final_answer = final_answer


def _parse_react_action(response: LlmResponse) -> _ReactAction:
    action = response.json_action()
    if isinstance(action, dict):
        if "final_answer" in action:
            return _ReactAction(
                thought=_optional_str(action.get("thought")),
                final_answer=str(action["final_answer"]),
            )
        nested_action = action.get("action")
        if isinstance(nested_action, dict):
            return _ReactAction(
                thought=_optional_str(action.get("thought")),
                tool_name=_optional_str(nested_action.get("tool") or nested_action.get("tool_name")),
                arguments=nested_action.get("arguments") if isinstance(nested_action.get("arguments"), dict) else {},
            )
        if action.get("tool_name") or action.get("name"):
            arguments = action.get("arguments")
            return _ReactAction(
                thought=_optional_str(action.get("thought")),
                tool_name=_optional_str(action.get("tool_name") or action.get("name")),
                arguments=arguments if isinstance(arguments, dict) else {},
            )

    if response.tool_calls:
        first = response.tool_calls[0]
        return _ReactAction(tool_name=first.name, arguments=first.arguments)

    if response.content:
        if _is_pseudo_tool_call(response.content):
            # Model wrote tool calls as plain text instead of structured calls.
            # Treating this as a final answer would produce a false pass.
            log.warning("ReAct: response looks like pseudo tool call in plain text — not accepting as final_answer")
            return _ReactAction()
        return _ReactAction(final_answer=response.content)
    return _ReactAction()


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


# Detects model responses that look like pseudo tool calls written as plain text
# instead of structured tool_call JSON.  Example pattern the model produces:
#   "Tool: bma_create_object\nArguments: {...}"
_PSEUDO_TOOL_RE = re.compile(
    r"^\s*(Tool|Action|Arguments?|Observation)\s*:",
    re.MULTILINE | re.IGNORECASE,
)


def _is_pseudo_tool_call(content: str) -> bool:
    """Return True if content looks like text-serialised tool calls, not a genuine final answer."""
    return bool(_PSEUDO_TOOL_RE.search(content))
