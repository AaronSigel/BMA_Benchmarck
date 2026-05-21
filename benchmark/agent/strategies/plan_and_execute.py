from __future__ import annotations

import datetime
import json
import re
import uuid
from pathlib import Path
from typing import Any

from benchmark.agent.errors import AgentRuntimeError, LlmResponseParseError
from benchmark.agent.llm.base import LlmClient, LlmMessage, LlmResponse
from benchmark.agent.models import AgentConfig, AgentStepType, AgentTrace, ToolCallStatus
from benchmark.agent.prompts import PromptBuilder, _AGENT_HIDDEN_TOOLS
from benchmark.agent.tool_context import AgentToolContext, ToolSchemaProvider
from benchmark.agent.tool_executor import ToolExecutor


class PlanAndExecuteStrategy:
    """Plan once with the LLM, then execute each planned tool call in order."""

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
            raise AgentRuntimeError("Plan-and-execute strategy requires an LLM client")

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
        system_prompt = self.prompt_builder.build_system_prompt(
            agent_config,
            agent_config.mcp_profile,
            tool_contract_dicts,
        )
        plan_prompt = self.prompt_builder.build_plan_prompt(task)
        llm_started_at = datetime.datetime.now(datetime.timezone.utc)
        response = llm_client.complete(
            [
                LlmMessage(role="system", content=system_prompt),
                LlmMessage(role="user", content=plan_prompt),
            ],
            tools=None,
            timeout_sec=agent_config.step_timeout_sec,
        )
        llm_finished_at = datetime.datetime.now(datetime.timezone.utc)
        try:
            plan = _parse_plan(response)
        except LlmResponseParseError as error:
            trace = trace.add_step(
                AgentStepType.ERROR,
                error=str(error),
                raw_llm_response=response.model_dump(mode="json", exclude_none=True),
                started_at=llm_started_at,
                finished_at=llm_finished_at,
                duration_sec=(llm_finished_at - llm_started_at).total_seconds(),
            )
            finished_at = datetime.datetime.now(datetime.timezone.utc)
            return trace.model_copy(
                update={
                    "success": False,
                    "error": str(error),
                    "finished_at": finished_at,
                    "duration_sec": (finished_at - started_at).total_seconds(),
                }
            )
        trace = trace.add_step(
            AgentStepType.PLAN,
            action="plan",
            observation={"plan": [step.raw for step in plan]},
            raw_llm_response=response.model_dump(mode="json", exclude_none=True),
            started_at=llm_started_at,
            finished_at=llm_finished_at,
            duration_sec=(llm_finished_at - llm_started_at).total_seconds(),
        )

        error_message = None
        executed = 0
        max_execution_steps = max(agent_config.max_steps - len(trace.steps) - 1, 0)
        for plan_step in plan[:max_execution_steps]:
            result = tool_executor.call_tool(plan_step.tool, plan_step.arguments)
            executed += 1
            trace = trace.add_step(
                AgentStepType.TOOL_CALL,
                thought=plan_step.description,
                action="call_tool",
                tool_name=result.name,
                tool_arguments=plan_step.arguments,
                observation=result.result if result.error is None else {"error": result.error},
                error=result.error,
                started_at=result.started_at,
                finished_at=result.finished_at,
                duration_sec=result.duration_sec,
                metadata={"plan_step": plan_step.step},
            )
            if result.status != ToolCallStatus.SUCCEEDED:
                error_message = result.error or f"Tool failed: {result.name}"
                break

        if len(plan) > max_execution_steps and error_message is None:
            error_message = "Plan-and-execute strategy reached max_steps before executing all plan steps"

        success = error_message is None
        final_message = f"Executed {executed} plan step(s)." if success else None
        trace = trace.add_step(
            AgentStepType.FINAL if success else AgentStepType.ERROR,
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
            }
        )


class _PlanStep:
    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.step = raw["step"]
        self.description = str(raw["description"])
        self.tool = str(raw["tool"])
        self.arguments = raw["arguments"]


def _parse_plan(response: LlmResponse) -> list[_PlanStep]:
    action = _json_action(response)
    if action is None:
        raise LlmResponseParseError(
            "Plan-and-execute response must be a JSON object with a plan list",
            fragment=response.content,
            raw_response=response.raw_response,
        )
    raw_plan = action.get("plan") or action.get("steps")
    if not isinstance(raw_plan, list) or not raw_plan:
        raise LlmResponseParseError(
            "Plan-and-execute response must contain a non-empty plan list",
            fragment=response.content,
            raw_response=response.raw_response,
        )

    steps = []
    for index, item in enumerate(raw_plan):
        if not isinstance(item, dict):
            raise _invalid_plan(response, f"plan[{index}] must be an object")
        if not isinstance(item.get("step"), int):
            raise _invalid_plan(response, f"plan[{index}].step must be an integer")
        if not isinstance(item.get("description"), str) or not item["description"].strip():
            raise _invalid_plan(response, f"plan[{index}].description must be a non-empty string")
        if not isinstance(item.get("tool"), str) or not item["tool"].strip():
            raise _invalid_plan(response, f"plan[{index}].tool must be a non-empty string")
        if not isinstance(item.get("arguments"), dict):
            raise _invalid_plan(response, f"plan[{index}].arguments must be an object")
        steps.append(_PlanStep(item))
    return sorted(steps, key=lambda item: item.step)


def _json_action(response: LlmResponse) -> dict[str, Any] | None:
    action = response.json_action()
    if isinstance(action, dict):
        return action

    parsed = _parse_json_from_content(response.content)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"plan": parsed}
    return None


_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)


def _parse_json_from_content(content: str | None) -> Any:
    if content is None:
        return None
    text = content.strip()
    fence = _JSON_FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value
    return None


def _invalid_plan(response: LlmResponse, message: str) -> LlmResponseParseError:
    return LlmResponseParseError(
        message,
        fragment=response.content,
        raw_response=response.raw_response,
    )
