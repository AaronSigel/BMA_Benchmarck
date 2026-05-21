from __future__ import annotations

import datetime
import json
import logging
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

# Default max_steps per task category when max_steps_by_category is not overridden.
_DEFAULT_CATEGORY_MAX_STEPS: dict[str, int] = {
    "geometry": 4,
    "materials": 5,
    "lighting": 5,
    "camera": 4,
    "export": 8,
    "composition": 8,
}

# Issue codes whose presence blocks export.
_EXPORT_BLOCKING_CODES = frozenset({
    "object_missing",
    "object_missing_for_transform",
    "material_missing",
    "object_material_missing",
    "light_missing",
    "camera_missing",
})

# Tools that mutate the scene (require object existence precondition checks).
_OBJECT_REF_TOOLS = frozenset({"bma_set_transform", "bma_assign_material", "bma_set_material"})


class ReactStrategy:
    """Validator-driven iterative ReAct loop.

    Each step:
    1. Inject current validation state + suggested repair into the prompt.
    2. LLM returns a JSON action (thought / action / finish).
    3. Guard checks (export block, duplicate object, repeated action, precondition).
    4. Execute tool.
    5. Run scene validation.
    6. Check issue-level and score-level progress.
    7. Repeat until scene passes, max_steps, or no-progress limit.
    """

    def __init__(
        self,
        *,
        prompt_builder: PromptBuilder | None = None,
        tool_schema_provider: ToolSchemaProvider | None = None,
    ) -> None:
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.tool_schema_provider = tool_schema_provider or ToolSchemaProvider()
        # Injected by AgentRuntime when stop_after_scene_passed or detect_no_progress is True.
        # Returns (scene_ok, score_or_None, SceneValidationResult_or_None).
        self.scene_validator_fn: Callable[[Path], tuple[bool, float | None, Any]] | None = None

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

        # Resolve effective max_steps from task category (P0.6).
        effective_max_steps = _resolve_max_steps(task, agent_config)

        system_message = LlmMessage(
            role="system",
            content=self.prompt_builder.build_system_prompt(
                agent_config,
                agent_config.mcp_profile,
                tool_contract_dicts,
            ),
        )

        # Load BenchmarkTask model once for mapper / checklist (best-effort).
        task_obj = _try_load_task(task)

        observations: list[str | dict[str, Any]] = []
        final_message = None
        error_message = None
        success = False
        react_iteration_count = 0

        # Guard state
        previous_action_signature: str | None = None
        consecutive_repeated_actions = 0
        created_object_names: set[str] = set()

        # Counters for trace metadata
        duplicate_object_count = 0
        repeated_action_count = 0
        wasted_step_count = 0
        no_progress_step_count = 0
        consecutive_no_progress = 0
        blocked_export_count = 0
        repair_step_count = 0
        initial_issue_count: int | None = None
        initial_issue_codes: set[str] = set()

        # Validation / progress state
        previous_scene_score: float | None = None
        previous_issue_codes: set[str] = set()
        last_validation_result: Any = None  # SceneValidationResult | None

        # Inspection tracking
        mutation_steps = 0
        mutation_preceded_by_inspection = 0
        last_tool_was_inspection = False

        # Build first-step checklist (P2.2) — injected before any validation is available.
        initial_step_context: dict[str, Any] | None = None
        if task_obj is not None:
            from benchmark.agent.strategies.issue_action_mapper import build_task_checklist
            initial_step_context = {"task_checklist": build_task_checklist(task_obj)}

        if self.scene_validator_fn is not None:
            try:
                initial_snap_path = output_dir / "initial_react_snapshot.json"
                scene_ok, initial_score, initial_result = self.scene_validator_fn(initial_snap_path)
            except Exception as exc:
                log.debug("[react:%s] initial scene check failed: %s", trace.run_id[:8], exc)
                scene_ok, initial_score, initial_result = False, None, None
            if initial_result is not None:
                last_validation_result = initial_result
                previous_scene_score = initial_score
                previous_issue_codes = {i.code for i in initial_result.issues}
                initial_issue_count = len(previous_issue_codes)
                initial_issue_codes = set(previous_issue_codes)
            if scene_ok:
                log.info("[react:%s] initial scene passed, stopping before LLM call", trace.run_id[:8])
                success = True
                final_message = "scene_passed_initial_stop"
                trace = trace.add_step(AgentStepType.FINAL, observation=final_message)

        log.info(
            "[react:%s] starting task=%s max_steps=%d (effective=%d) profile=%s",
            trace.run_id[:8], task_id, agent_config.max_steps, effective_max_steps, agent_config.mcp_profile,
        )

        while final_message is None and error_message is None and react_iteration_count < effective_max_steps:
            react_iteration_count += 1
            step_num = react_iteration_count

            # Capture pre-step validation state for per-step trace fields (R-10).
            _score_before: float | None = previous_scene_score
            _issue_count_before: int = len(previous_issue_codes)
            _top_issue_dict: dict[str, Any] | None = None
            _suggested_repair_dict: dict[str, Any] | None = None

            # Build validator-driven step context (P0.2).
            step_context = _build_step_context(
                last_validation_result, task_obj, initial_step_context if step_num == 1 else None
            )
            if step_context:
                _top_issue_dict = step_context.get("top_issue")
                _suggested_repair_dict = step_context.get("suggested_repair")

            log.info("[react:%s] step %d/%d — calling LLM", trace.run_id[:8], step_num, effective_max_steps)
            messages = [
                system_message,
                LlmMessage(
                    role="user",
                    content=self.prompt_builder.build_react_prompt_context(task, observations, step_context),
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

            if action.parse_error is not None:
                error_message = "LlmParseError"
                trace = trace.add_step(
                    AgentStepType.ERROR,
                    error=f"LlmParseError: {action.parse_error}",
                    metadata={
                        "react_error_type": "LlmParseError",
                        "react_non_strict_response": True,
                    },
                )
                break

            if action.final_answer is not None:
                # R-05: only allow finish if scene passed or validator unavailable.
                # If no prior validation result exists, run the validator eagerly.
                _finish_allowed = True
                if self.scene_validator_fn is not None:
                    _check_result = last_validation_result
                    if _check_result is None:
                        try:
                            _snap_path = output_dir / f"finish_check_step_{step_num}_snapshot.json"
                            _, _, _check_result = self.scene_validator_fn(_snap_path)
                            if _check_result is not None:
                                last_validation_result = _check_result
                        except Exception:
                            _check_result = None
                    if _check_result is not None and not getattr(_check_result, "passed", False):
                        _finish_allowed = False

                if not _finish_allowed:
                    repair_hint = (
                        "Cannot finish: the scene has not passed validation. "
                        "Fix the remaining issues first, then finish."
                    )
                    observations.append({
                        "tool": "__premature_finish__",
                        "arguments": {},
                        "observation": {"premature_finish_blocked": True, "repair_hint": repair_hint},
                    })
                    trace = trace.add_step(
                        AgentStepType.OBSERVATION,
                        observation=repair_hint,
                        metadata={"premature_finish_blocked": True, "repair_attempt": True},
                    )
                    log.info("[react:%s] step %d — premature finish blocked (scene not passed)", trace.run_id[:8], step_num)
                    continue
                log.info("[react:%s] step %d — final answer received", trace.run_id[:8], step_num)
                final_message = action.final_answer
                success = True
                trace = trace.add_step(AgentStepType.FINAL, observation=final_message)
                break

            if action.tool_name is None:
                error_message = "LlmParseError"
                trace = trace.add_step(
                    AgentStepType.ERROR,
                    error="LlmParseError: ReAct response did not include a strict JSON action or finish",
                    metadata={"react_error_type": "LlmParseError"},
                )
                break

            # --- Guard: repeated action (existing) ---
            signature = f"{action.tool_name}:{action.arguments}"
            if agent_config.detect_repeated_actions and signature == previous_action_signature:
                consecutive_repeated_actions += 1
                repeated_action_count += 1
                wasted_step_count += 1
                if consecutive_repeated_actions == 1:
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
                        "[react:%s] step %d — repair hint for repeated action: %s",
                        trace.run_id[:8], step_num, action.tool_name,
                    )
                    continue
                error_message = "ReactInvalidAction"
                trace = trace.add_step(
                    AgentStepType.ERROR,
                    error=f"ReactInvalidAction: repeated action {action.tool_name}",
                    metadata={"react_error_type": "ReactInvalidAction", "repeated_action": True, "wasted_step": True},
                )
                break
            else:
                consecutive_repeated_actions = 0
            previous_action_signature = signature

            # --- Guard: duplicate object (existing + strengthened) ---
            if agent_config.detect_duplicate_objects and action.tool_name == "bma_create_object":
                object_name = action.arguments.get("name")
                if isinstance(object_name, str) and object_name and object_name in created_object_names:
                    wasted_step_count += 1
                    repair_hint = (
                        f"Object '{object_name}' already exists in the scene. "
                        "Do not create it again. "
                        "If its transform is wrong, use bma_set_transform. "
                        "If its material is missing, use bma_assign_material."
                    )
                    observations.append({
                        "tool": action.tool_name,
                        "arguments": action.arguments,
                        "observation": {
                            "object_already_exists_handled": object_name,
                            "repair_hint": repair_hint,
                        },
                    })
                    trace = trace.add_step(
                        AgentStepType.OBSERVATION,
                        observation=repair_hint,
                        metadata={"object_already_exists_handled": object_name, "wasted_step": True, "repair_attempt": True},
                    )
                    continue
                if isinstance(object_name, str) and object_name:
                    created_object_names.add(object_name)

            # --- Guard: precondition check for transform/material tools (P1.1) ---
            if action.tool_name in _OBJECT_REF_TOOLS and created_object_names:
                target_name = action.arguments.get("object_name")
                if isinstance(target_name, str) and target_name and target_name not in created_object_names:
                    wasted_step_count += 1
                    repair_hint = (
                        f"Cannot call {action.tool_name}: object '{target_name}' has not been created yet. "
                        "Create it first with bma_create_object."
                    )
                    observations.append({
                        "tool": "__precondition_failed__",
                        "arguments": {},
                        "observation": {"precondition_failed": True, "repair_hint": repair_hint},
                    })
                    trace = trace.add_step(
                        AgentStepType.OBSERVATION,
                        observation=repair_hint,
                        metadata={"precondition_failed": True, "wasted_step": True},
                    )
                    continue

            # --- Guard: export block (P0.4) ---
            if action.tool_name == "bma_export_scene" and last_validation_result is not None:
                blocking = [
                    i for i in last_validation_result.issues
                    if i.code in _EXPORT_BLOCKING_CODES
                ]
                if blocking:
                    wasted_step_count += 1
                    blocked_export_count += 1
                    blocking_desc = ", ".join({i.code for i in blocking})
                    repair_hint = (
                        f"Cannot export: scene has unresolved issues ({blocking_desc}). "
                        "Fix all missing objects and materials before exporting."
                    )
                    observations.append({
                        "tool": "__export_blocked__",
                        "arguments": {},
                        "observation": {"export_blocked": True, "blocking_issues": blocking_desc, "repair_hint": repair_hint},
                    })
                    trace = trace.add_step(
                        AgentStepType.OBSERVATION,
                        observation=repair_hint,
                        metadata={"export_blocked": True, "wasted_step": True},
                    )
                    continue

            # --- Execute tool ---
            log.info("[react:%s] step %d — tool_call: %s args=%s", trace.run_id[:8], step_num, action.tool_name, list((action.arguments or {}).keys()))
            _tool_category = (
                TOOL_CONTRACT_MAP.get(action.tool_name).category
                if action.tool_name in TOOL_CONTRACT_MAP else ToolCategory.OTHER
            )
            _is_mutation = _tool_category in {
                ToolCategory.OBJECT, ToolCategory.TRANSFORM, ToolCategory.MATERIAL,
                ToolCategory.LIGHT, ToolCategory.CAMERA, ToolCategory.EXPORT,
            }
            _is_inspection = _tool_category == ToolCategory.INSPECTION
            if _is_mutation:
                mutation_steps += 1
                if last_tool_was_inspection:
                    mutation_preceded_by_inspection += 1
            last_tool_was_inspection = _is_inspection

            tool_result = tool_executor.call_tool(action.tool_name, action.arguments)
            observation = tool_result.result if tool_result.error is None else {"error": tool_result.error}

            # Track actual created object name (may differ from requested due to .001 suffix).
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

            # Update live object name tracking from snapshot tool results.
            if action.tool_name == "bma_get_scene_snapshot" and isinstance(observation, dict):
                obj_list = observation.get("objects") or []
                if isinstance(obj_list, list):
                    for obj_entry in obj_list:
                        if isinstance(obj_entry, dict) and isinstance(obj_entry.get("name"), str):
                            created_object_names.add(obj_entry["name"])

            if tool_result.error:
                log.warning("[react:%s] step %d — tool %s error: %s", trace.run_id[:8], step_num, action.tool_name, tool_result.error)
            else:
                log.info("[react:%s] step %d — tool %s ok (%.2fs)", trace.run_id[:8], step_num, action.tool_name, tool_result.duration_sec or 0)

            # Count repair-driven steps (top_issue was present when action was chosen).
            _is_repair_step = _top_issue_dict is not None
            if _is_repair_step:
                repair_step_count += 1

            observations.append({
                "tool": tool_result.name,
                "arguments": action.arguments,
                "observation": observation,
            })
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
                metadata={
                    "react": {
                        "step_index": step_num,
                        "top_issue": _top_issue_dict,
                        "suggested_repair": _suggested_repair_dict,
                        "selected_action": {
                            "tool": action.tool_name,
                            "arguments": action.arguments,
                        },
                        "score_before": _score_before,
                        "score_after": None,  # filled in after validation below
                        "issue_count_before": _issue_count_before,
                        "issue_count_after": None,  # filled in after validation below
                        "made_progress": None,
                        "wasted_step": False,
                    }
                },
            )
            if tool_result.error:
                error_message = tool_result.error
                break

            # --- Scene validation check ---
            _should_check_scene = (
                (agent_config.stop_after_scene_passed or agent_config.detect_no_progress)
                and self.scene_validator_fn is not None
            )
            if _should_check_scene:
                snap_path = output_dir / f"mid_step_{step_num}_snapshot.json"
                try:
                    scene_ok, current_score, val_result = self.scene_validator_fn(snap_path)
                except Exception as _ve:
                    log.debug("[react:%s] step %d — scene check failed: %s", trace.run_id[:8], step_num, _ve)
                    scene_ok, current_score, val_result = False, None, None

                if val_result is not None:
                    last_validation_result = val_result

                if agent_config.stop_after_scene_passed and scene_ok:
                    log.info("[react:%s] step %d — scene passed, stopping early", trace.run_id[:8], step_num)
                    success = True
                    final_message = "scene_passed_early_stop"
                    trace = trace.add_step(AgentStepType.FINAL, observation="scene_passed_early_stop")
                    break

                if agent_config.detect_no_progress and current_score is not None:
                    # Issue-level progress check (P0.5): progress if score improved OR
                    # a priority issue disappeared OR issue count decreased.
                    current_issue_codes = {i.code for i in (val_result.issues if val_result else [])}
                    if initial_issue_count is None:
                        initial_issue_count = len(current_issue_codes)
                        initial_issue_codes = set(current_issue_codes)
                    score_improved = previous_scene_score is None or current_score > previous_scene_score
                    issues_reduced = len(current_issue_codes) < len(previous_issue_codes)
                    priority_resolved = bool(previous_issue_codes - current_issue_codes)
                    made_progress = score_improved or issues_reduced or priority_resolved

                    # Back-fill score_after / issue_count_after / made_progress into last step.
                    if trace.steps:
                        last_step = trace.steps[-1]
                        react_meta = (last_step.metadata or {}).get("react")
                        if react_meta is not None:
                            react_meta["score_after"] = current_score
                            react_meta["issue_count_after"] = len(current_issue_codes)
                            react_meta["made_progress"] = made_progress

                    if not made_progress and not tool_result.error:
                        consecutive_no_progress += 1
                        no_progress_step_count += 1
                        log.debug(
                            "[react:%s] step %d — no progress (score %.3f, prev %.3f, issues %d→%d), consecutive=%d",
                            trace.run_id[:8], step_num, current_score, previous_scene_score or 0.0,
                            len(previous_issue_codes), len(current_issue_codes), consecutive_no_progress,
                        )
                        # P1.2: on first stall inject a repair hint instead of stopping immediately.
                        if consecutive_no_progress == 1 and val_result is not None:
                            hint = _build_no_progress_hint(val_result, task_obj)
                            observations.append({
                                "tool": "__no_progress__",
                                "arguments": {},
                                "observation": {"no_progress": True, "repair_hint": hint},
                            })
                            trace = trace.add_step(
                                AgentStepType.OBSERVATION,
                                observation=hint,
                                metadata={"no_progress_detected": True, "repair_attempt": True},
                            )
                            log.info("[react:%s] step %d — no progress hint injected", trace.run_id[:8], step_num)
                        elif consecutive_no_progress >= agent_config.no_progress_limit:
                            error_message = "ReactNoProgress"
                            trace = trace.add_step(
                                AgentStepType.ERROR,
                                error=error_message,
                                metadata={
                                    "react_error_type": "ReactNoProgress",
                                    "no_progress_detected": True,
                                    "no_progress_step_count": no_progress_step_count,
                                },
                            )
                            break
                    else:
                        consecutive_no_progress = 0

                    previous_scene_score = current_score
                    previous_issue_codes = current_issue_codes

        if final_message is None and error_message is None:
            error_message = "ReactMaxSteps"
            trace = trace.add_step(
                AgentStepType.ERROR,
                error="ReactMaxSteps: max steps reached without scene passing",
                metadata={"react_error_type": "ReactMaxSteps"},
            )

        finished_at = datetime.datetime.now(datetime.timezone.utc)
        _react_error_type = (
            "ReactMaxSteps" if error_message == "ReactMaxSteps"
            else "ReactNoProgress" if error_message == "ReactNoProgress"
            else "ReactInvalidAction" if error_message == "ReactInvalidAction"
            else "LlmParseError" if error_message == "LlmParseError"
            else None
        )
        _resolved_issues = (
            len(initial_issue_codes - previous_issue_codes)
            if initial_issue_codes else 0
        )
        scene_passed_but_agent_error = (
            error_message is not None
            and last_validation_result is not None
            and bool(getattr(last_validation_result, "passed", False))
        )
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
                    "effective_max_steps": effective_max_steps,
                    "scene_passed_but_agent_error": scene_passed_but_agent_error,
                    "react_iterations_total": react_iteration_count,
                    "inspection_before_mutation_rate": (
                        mutation_preceded_by_inspection / mutation_steps
                        if mutation_steps > 0 else "not_available"
                    ),
                    # R-10: ReAct aggregate metrics
                    "react_steps_total": len(trace.steps),
                    "react_repair_steps": repair_step_count,
                    "react_wasted_steps": wasted_step_count,
                    "react_no_progress_count": no_progress_step_count,
                    "react_blocked_export_count": blocked_export_count,
                    "react_max_steps_count": 1 if _react_error_type == "ReactMaxSteps" else 0,
                    "react_error_type": _react_error_type,
                    "react_issue_resolution_rate": (
                        _resolved_issues / initial_issue_count
                        if initial_issue_count and initial_issue_count > 0 else None
                    ),
                },
            }
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_max_steps(task: dict[str, Any], config: AgentConfig) -> int:
    """Return the effective max_steps, taking task category overrides into account."""
    category = str(task.get("category") or "").lower()
    if config.max_steps_by_category and category in config.max_steps_by_category:
        return config.max_steps_by_category[category]
    if category in _DEFAULT_CATEGORY_MAX_STEPS:
        return _DEFAULT_CATEGORY_MAX_STEPS[category]
    return config.max_steps


def _try_load_task(task: dict[str, Any]) -> Any:
    """Parse task dict into BenchmarkTask model; return None on failure."""
    try:
        from benchmark.tasks.models import BenchmarkTask
        return BenchmarkTask.model_validate(task)
    except Exception:
        return None


def _build_step_context(
    val_result: Any,
    task_obj: Any,
    initial_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build the current-scene-state dict injected into each step's prompt."""
    if val_result is None:
        return initial_context  # First step: task checklist

    from benchmark.agent.strategies.issue_action_mapper import select_top_issue, map_issue_to_repair

    issues = val_result.issues
    top_issue = select_top_issue(issues) if issues else None
    suggested_repair = None
    if top_issue is not None and task_obj is not None:
        action = map_issue_to_repair(top_issue, task_obj)
        if action is not None:
            suggested_repair = {
                "tool": action.tool_name,
                "arguments": action.arguments_template,
                "description": action.description,
            }
            if action.requires_prior_step:
                suggested_repair["note"] = (
                    f"After this step, also run: {action.requires_prior_step.tool_name}"
                )

    ctx: dict[str, Any] = {
        "scene_score": round(val_result.total_score, 3),
        "scene_status": val_result.overall_status,
        "remaining_issue_count": len(issues),
    }
    if top_issue is not None:
        ctx["top_issue"] = {
            "code": top_issue.code,
            "message": top_issue.message,
            "severity": top_issue.severity,
        }
    if suggested_repair is not None:
        ctx["suggested_repair"] = suggested_repair
    if initial_context:
        ctx.update(initial_context)
    return ctx


def _build_no_progress_hint(val_result: Any, task_obj: Any) -> str:
    """Build a repair hint to inject when no progress is detected."""
    from benchmark.agent.strategies.issue_action_mapper import select_top_issue, map_issue_to_repair

    issues = val_result.issues if val_result else []
    top_issue = select_top_issue(issues) if issues else None
    if top_issue is None:
        return "No progress detected. Try a different approach or finish if the scene is complete."

    hint = (
        f"No progress detected. The top unresolved issue is '{top_issue.code}': {top_issue.message}."
    )
    if task_obj is not None:
        repair = map_issue_to_repair(top_issue, task_obj)
        if repair is not None:
            hint += f" Suggested repair: use {repair.tool_name} with {repair.arguments_template}."
    return hint


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

class _ReactAction:
    def __init__(
        self,
        *,
        thought: str | None = None,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        final_answer: str | None = None,
        parse_error: str | None = None,
    ) -> None:
        self.thought = thought
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.final_answer = final_answer
        self.parse_error = parse_error


def _parse_react_action(response: LlmResponse) -> _ReactAction:
    if response.tool_calls:
        first = response.tool_calls[0]
        return _ReactAction(tool_name=first.name, arguments=first.arguments)

    if response.content is None or not response.content.strip():
        return _ReactAction(parse_error="empty LLM response")
    try:
        action = json.loads(response.content)
    except json.JSONDecodeError as exc:
        return _ReactAction(parse_error=f"ReactNonStrictResponse: content is not strict JSON: {exc}")
    if not isinstance(action, dict):
        return _ReactAction(parse_error="ReactNonStrictResponse: JSON response must be an object")
    extra_keys = set(action) - {"thought", "action", "finish"}
    if extra_keys:
        return _ReactAction(
            parse_error=f"ReactNonStrictResponse: unsupported top-level keys: {sorted(extra_keys)}"
        )

    finish = action.get("finish")
    thought = _optional_str(action.get("thought"))
    nested_action = action.get("action")

    if finish is True:
        if nested_action is not None:
            return _ReactAction(parse_error="ReactNonStrictResponse: finish=true requires action=null")
        return _ReactAction(thought=thought, final_answer=thought or "finish")

    if finish is not False:
        return _ReactAction(parse_error="ReactNonStrictResponse: finish must be true or false")
    if not isinstance(nested_action, dict):
        return _ReactAction(parse_error="ReactNonStrictResponse: finish=false requires action object")
    extra_action_keys = set(nested_action) - {"tool", "arguments"}
    if extra_action_keys:
        return _ReactAction(
            parse_error=f"ReactNonStrictResponse: unsupported action keys: {sorted(extra_action_keys)}"
        )

    tool_name = _optional_str(nested_action.get("tool"))
    if not tool_name:
        return _ReactAction(parse_error="ReactNonStrictResponse: action.tool is required")
    arguments = nested_action.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return _ReactAction(parse_error="ReactNonStrictResponse: action.arguments must be an object")
    return _ReactAction(thought=thought, tool_name=tool_name, arguments=arguments)


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None
