from __future__ import annotations

import datetime
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Callable

from benchmark.agent.errors import AgentRuntimeError
from benchmark.agent.llm.base import LlmClient, LlmMessage, LlmResponse
from benchmark.agent.models import AgentConfig, AgentStepType, AgentTrace, ToolCallStatus
from benchmark.agent.prompts import PromptBuilder, _AGENT_HIDDEN_TOOLS
from benchmark.agent.tool_context import AgentToolContext, ToolSchemaProvider
from benchmark.agent.tool_executor import ToolExecutor
from benchmark.mcp.tool_contract import TOOL_CONTRACT_MAP, ToolCategory

log = logging.getLogger(__name__)


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
        # Injected by AgentRuntime when stop_after_scene_passed=True.
        # Called with a snapshot output path; returns (scene_ok, score_or_None).
        self.scene_validator_fn: Callable[[Path], tuple[bool, float | None]] | None = None

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

        tool_contracts = [
            contract for contract in self.tool_schema_provider.get_tools_for_profile(agent_config.mcp_profile)
            if contract.name not in _AGENT_HIDDEN_TOOLS
        ]
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
        previous_action_signature: str | None = None
        consecutive_repeated_actions = 0
        created_object_names: set[str] = set()
        duplicate_object_count = 0
        repeated_action_count = 0
        wasted_step_count = 0
        no_progress_step_count = 0
        consecutive_no_progress = 0
        previous_scene_score: float | None = None
        mutation_steps = 0
        mutation_preceded_by_inspection = 0
        last_tool_was_inspection = False

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

            signature = f"{action.tool_name}:{action.arguments}"
            if agent_config.detect_repeated_actions and signature == previous_action_signature:
                consecutive_repeated_actions += 1
                repeated_action_count += 1
                wasted_step_count += 1
                if consecutive_repeated_actions == 1:
                    # First repeat: inject a repair hint and let the agent try again
                    repair_hint = (
                        f"The previous action ({action.tool_name}) was repeated and did not "
                        "improve the scene. Choose a different tool or finish if the scene is complete."
                    )
                    observations.append({
                        "tool": "__repair__",
                        "arguments": {},
                        "observation": {"repair_hint": repair_hint},
                    })
                    trace = trace.add_step(
                        AgentStepType.OBSERVATION,
                        observation=repair_hint,
                        metadata={"repair_attempt": True, "repeated_action": True},
                    )
                    log.info(
                        "[react:%s] step %d — repair hint injected for repeated action: %s",
                        trace.run_id[:8], step_num, action.tool_name,
                    )
                    continue
                # Second consecutive repeat: stop
                error_message = f"ReAct repeated the same action: {action.tool_name}"
                trace = trace.add_step(
                    AgentStepType.ERROR,
                    error=error_message,
                    metadata={"repeated_action": True, "wasted_step": True},
                )
                break
            else:
                consecutive_repeated_actions = 0
            previous_action_signature = signature

            if agent_config.detect_duplicate_objects and action.tool_name == "bma_create_object":
                object_name = action.arguments.get("name")
                if isinstance(object_name, str) and object_name and object_name in created_object_names:
                    wasted_step_count += 1
                    repair_hint = (
                        f"Object '{object_name}' already exists. "
                        "Do not create it again; update its transform or material instead."
                    )
                    observations.append(
                        {
                            "tool": action.tool_name,
                            "arguments": action.arguments,
                            "observation": {
                                "object_already_exists_handled": object_name,
                                "repair_hint": repair_hint,
                            },
                        }
                    )
                    trace = trace.add_step(
                        AgentStepType.OBSERVATION,
                        observation=repair_hint,
                        metadata={"object_already_exists_handled": object_name, "wasted_step": True, "repair_attempt": True},
                    )
                    continue
                if isinstance(object_name, str) and object_name:
                    created_object_names.add(object_name)

            log.info("[react:%s] step %d — tool_call: %s args=%s", trace.run_id[:8], step_num, action.tool_name, list((action.arguments or {}).keys()))
            _tool_category = (TOOL_CONTRACT_MAP.get(action.tool_name).category if action.tool_name in TOOL_CONTRACT_MAP else ToolCategory.OTHER)
            _is_mutation = _tool_category in {ToolCategory.OBJECT, ToolCategory.TRANSFORM, ToolCategory.MATERIAL, ToolCategory.LIGHT, ToolCategory.CAMERA, ToolCategory.EXPORT}
            _is_inspection = _tool_category == ToolCategory.INSPECTION
            if _is_mutation:
                mutation_steps += 1
                if last_tool_was_inspection:
                    mutation_preceded_by_inspection += 1
            last_tool_was_inspection = _is_inspection
            tool_result = tool_executor.call_tool(action.tool_name, action.arguments)
            observation = tool_result.result if tool_result.error is None else {"error": tool_result.error}
            if (
                agent_config.detect_duplicate_objects
                and action.tool_name == "bma_create_object"
                and isinstance(action.arguments.get("name"), str)
                and isinstance(observation, dict)
            ):
                result_payload = observation.get("result") if isinstance(observation.get("result"), dict) else observation
                actual_name = result_payload.get("name") if isinstance(result_payload, dict) else None
                requested_name = action.arguments["name"]
                if isinstance(actual_name, str) and actual_name != requested_name and actual_name.startswith(f"{requested_name}."):
                    duplicate_object_count += 1
                    wasted_step_count += 1
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
            _should_check_scene = (
                (agent_config.stop_after_scene_passed or agent_config.detect_no_progress)
                and self.scene_validator_fn is not None
            )
            if _should_check_scene:
                snap_path = output_dir / f"mid_step_{step_num}_snapshot.json"
                try:
                    scene_ok, current_score = self.scene_validator_fn(snap_path)
                except Exception as _ve:
                    log.debug("[react:%s] step %d — scene check failed: %s", trace.run_id[:8], step_num, _ve)
                    scene_ok, current_score = False, None

                if agent_config.stop_after_scene_passed and scene_ok:
                    log.info("[react:%s] step %d — scene passed, stopping early", trace.run_id[:8], step_num)
                    success = True
                    final_message = "scene_passed_early_stop"
                    trace = trace.add_step(AgentStepType.FINAL, observation="scene_passed_early_stop")
                    break

                if agent_config.detect_no_progress and current_score is not None:
                    score_improved = previous_scene_score is None or current_score > previous_scene_score
                    if not score_improved and not tool_result.error:
                        consecutive_no_progress += 1
                        no_progress_step_count += 1
                        log.debug(
                            "[react:%s] step %d — no progress (score %.3f, prev %.3f), consecutive=%d",
                            trace.run_id[:8], step_num, current_score, previous_scene_score or 0.0, consecutive_no_progress,
                        )
                        if consecutive_no_progress >= agent_config.no_progress_limit:
                            error_message = f"no_progress_detected: score did not improve for {consecutive_no_progress} consecutive steps"
                            trace = trace.add_step(
                                AgentStepType.ERROR,
                                error=error_message,
                                metadata={"no_progress_detected": True, "no_progress_step_count": no_progress_step_count},
                            )
                            break
                    else:
                        consecutive_no_progress = 0
                    previous_scene_score = current_score

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
                "metadata": {
                    **trace.metadata,
                    "duplicate_object_count": duplicate_object_count,
                    "repeated_action_count": repeated_action_count,
                    "wasted_step_count": wasted_step_count,
                    "no_progress_step_count": no_progress_step_count,
                    "inspection_before_mutation_rate": (
                        mutation_preceded_by_inspection / mutation_steps
                        if mutation_steps > 0 else "not_available"
                    ),
                },
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
            # Try to parse it rather than failing the step outright.
            parsed = _parse_pseudo_tool_call(response.content)
            if parsed is not None:
                log.warning(
                    "ReAct: parsed pseudo tool call in plain text — tool=%s", parsed.tool_name
                )
                return parsed
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

# Matches LangChain-style: "Action: tool_name" / "Action Input: {...}"
_ACTION_NAME_RE = re.compile(r"^\s*Action\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"^\s*Action\s+Input\s*:\s*(\{.*)", re.MULTILINE | re.IGNORECASE | re.DOTALL)
# Matches "Tool: tool_name" / "Arguments: {...}"
_TOOL_NAME_RE = re.compile(r"^\s*Tool\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_TOOL_ARGS_RE = re.compile(r"^\s*Arguments?\s*:\s*(\{.*)", re.MULTILINE | re.IGNORECASE | re.DOTALL)
# Thought extraction
_THOUGHT_RE = re.compile(r"^\s*Thought\s*:\s*(.+?)(?=^\s*(?:Action|Tool)\s*:)", re.MULTILINE | re.IGNORECASE | re.DOTALL)


def _is_pseudo_tool_call(content: str) -> bool:
    """Return True if content looks like text-serialised tool calls, not a genuine final answer."""
    return bool(_PSEUDO_TOOL_RE.search(content))


def _extract_json_obj(text: str) -> dict[str, Any] | None:
    """Extract the first complete JSON object from text."""
    import json as _json
    depth = 0
    start = text.find("{")
    if start == -1:
        return None
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return _json.loads(text[start : i + 1])
                except _json.JSONDecodeError:
                    return None
    return None


def _parse_pseudo_tool_call(content: str) -> "_ReactAction | None":
    """Try to extract a tool name and arguments from a plain-text ReAct response."""
    # LangChain style: Action / Action Input
    name_match = _ACTION_NAME_RE.search(content)
    if name_match:
        tool_name = name_match.group(1).strip()
        args: dict[str, Any] = {}
        input_match = _ACTION_INPUT_RE.search(content)
        if input_match:
            args = _extract_json_obj(input_match.group(1)) or {}
        thought_match = _THOUGHT_RE.search(content)
        thought = thought_match.group(1).strip() if thought_match else None
        return _ReactAction(thought=thought, tool_name=tool_name, arguments=args)

    # Tool / Arguments style
    name_match2 = _TOOL_NAME_RE.search(content)
    if name_match2:
        tool_name = name_match2.group(1).strip()
        args = {}
        args_match = _TOOL_ARGS_RE.search(content)
        if args_match:
            args = _extract_json_obj(args_match.group(1)) or {}
        thought_match = _THOUGHT_RE.search(content)
        thought = thought_match.group(1).strip() if thought_match else None
        return _ReactAction(thought=thought, tool_name=tool_name, arguments=args)

    return None
