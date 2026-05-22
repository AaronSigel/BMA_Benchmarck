"""Hybrid plan_execute_react_repair strategy.

Phase 1: PlanAndExecuteStrategy builds the primary scene.
Phase 2: If validation fails, ReactStrategy runs a capped repair loop (max 5 steps).
The two traces are merged into one AgentTrace with hybrid metadata.
"""
from __future__ import annotations

import datetime
import logging
import uuid
from pathlib import Path
from typing import Any

from benchmark.agent.llm.base import LlmClient
from benchmark.agent.models import AgentConfig, AgentStep, AgentStrategyName, AgentTrace
from benchmark.agent.prompts import PromptBuilder
from benchmark.agent.tool_context import AgentToolContext, ToolSchemaProvider
from benchmark.agent.tool_executor import ToolExecutor

log = logging.getLogger(__name__)

_REPAIR_MAX_STEPS = 5


class PlanExecuteReactRepairStrategy:
    """Run plan-and-execute, validate, then repair with a validator-driven ReAct loop if needed."""

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
        started_at = datetime.datetime.now(datetime.timezone.utc)
        task_id = str(task.get("id") or tool_context.task_id or "unknown")
        run_id = tool_context.run_id or str(uuid.uuid4())

        log.info("[hybrid:%s] phase 1 — plan_and_execute", run_id[:8])
        from benchmark.agent.strategies.plan_and_execute import PlanAndExecuteStrategy

        plan_config = agent_config.model_copy(update={"strategy": AgentStrategyName.PLAN_AND_EXECUTE})
        plan_strategy = PlanAndExecuteStrategy(
            prompt_builder=self.prompt_builder,
            tool_schema_provider=self.tool_schema_provider,
        )
        plan_trace = plan_strategy.run(task, plan_config, llm_client, tool_executor, tool_context, output_dir)
        plan_status = _plan_phase_status(plan_trace)

        log.info("[hybrid:%s] phase 2 — post-plan validation", run_id[:8])
        val_result, val_unavailable_reason = _validate_via_tool_with_reason(
            tool_executor, task, output_dir / "hybrid_post_plan_snapshot.json"
        )

        if val_result is None:
            log.info("[hybrid:%s] validation unavailable (%s), returning plan trace", run_id[:8], val_unavailable_reason)
            return _stamp_trace(
                plan_trace,
                agent_config,
                started_at,
                plan_status=plan_status,
                hybrid_repair_used=False,
                repair_unavailable=True,
                repair_unavailable_reason=val_unavailable_reason or "validation_unavailable",
                plan_scene_status="unavailable",
                plan_score=None,
                plan_issue_count=None,
            )

        from benchmark.validation.models import ValidationStatus

        _plan_status = str(
            val_result.overall_status.value
            if hasattr(val_result.overall_status, "value")
            else val_result.overall_status
        )
        _plan_score = val_result.total_score
        _plan_issue_count = len(val_result.issues)

        if val_result.overall_status in {ValidationStatus.PASSED, ValidationStatus.WARNING}:
            log.info(
                "[hybrid:%s] scene passed after plan, no repair needed (score=%.3f)",
                run_id[:8],
                val_result.total_score,
            )
            return _stamp_trace(
                plan_trace,
                agent_config,
                started_at,
                plan_status=plan_status,
                hybrid_repair_used=False,
                repair_not_needed=True,
                plan_scene_status=_plan_status,
                plan_score=_plan_score,
                plan_issue_count=_plan_issue_count,
            )

        log.info(
            "[hybrid:%s] scene needs repair (status=%s, score=%.3f, issues=%d) — starting react repair",
            run_id[:8],
            val_result.overall_status,
            val_result.total_score,
            len(val_result.issues),
        )

        from benchmark.agent.strategies.react import ReactStrategy

        repair_config = _repair_config(agent_config, task, val_result)
        react_strategy = ReactStrategy(
            prompt_builder=self.prompt_builder,
            tool_schema_provider=self.tool_schema_provider,
        )
        react_strategy.scene_validator_fn = _make_tool_validator(tool_executor, task)
        react_strategy.initial_validation_result = val_result

        react_trace = react_strategy.run(task, repair_config, llm_client, tool_executor, tool_context, output_dir)

        repair_issue_count_after = _plan_issue_count
        post_repair_val, _ = _validate_via_tool_with_reason(
            tool_executor, task, output_dir / "hybrid_post_repair_snapshot.json"
        )
        if post_repair_val is not None:
            repair_issue_count_after = len(post_repair_val.issues)

        repair_error_type = react_trace.metadata.get("react_error_type")
        repair_scene_status_after_error = _scene_status_from_validation(post_repair_val)
        repair_score_after = post_repair_val.total_score if post_repair_val is not None else None
        repair_improved_score = (
            repair_score_after is not None
            and _plan_score is not None
            and repair_score_after > _plan_score
        )
        repair_reduced_issue_count = repair_issue_count_after < _plan_issue_count

        merged = _merge_traces(
            plan_trace,
            react_trace,
            agent_config,
            run_id,
            task_id,
            started_at,
            post_repair_val=post_repair_val,
        )

        if post_repair_val is not None and post_repair_val.overall_status in {ValidationStatus.PASSED, ValidationStatus.WARNING}:
            repair_status = "passed"
        elif repair_improved_score or repair_reduced_issue_count:
            repair_status = "improved"
        else:
            repair_status = "failed"

        return _stamp_trace(
            merged,
            agent_config,
            started_at,
            plan_status=plan_status,
            hybrid_repair_used=True,
            plan_scene_status=_plan_status,
            plan_score=_plan_score,
            plan_issue_count=_plan_issue_count,
            repair_started=True,
            repair_start_reason=_repair_start_reason(plan_trace, plan_status),
            repair_result_status=repair_status,
            repair_result_score=repair_score_after if repair_score_after is not None else _plan_score,
            repair_issue_count_before=_plan_issue_count,
            repair_issue_count_after=repair_issue_count_after,
            repair_score_before=_plan_score,
            repair_score_after=repair_score_after,
            repair_improved_score=repair_improved_score,
            repair_reduced_issue_count=repair_reduced_issue_count,
            repair_error_type=repair_error_type if repair_status != "passed" else None,
            repair_scene_status_after_error=repair_scene_status_after_error,
        )


def _repair_config(agent_config: AgentConfig, task: dict[str, Any], val_result: Any | None = None) -> AgentConfig:
    category = str(task.get("category") or "").lower()
    no_progress_limit = agent_config.no_progress_limit or 2
    if category and agent_config.no_progress_limit_by_category.get(category):
        no_progress_limit = agent_config.no_progress_limit_by_category[category]

    detect_no_progress = agent_config.detect_no_progress
    if val_result is not None and detect_no_progress:
        from benchmark.agent.strategies.issue_action_mapper import select_top_issue

        plan_score = getattr(val_result, "total_score", None)
        top_issue = select_top_issue(getattr(val_result, "issues", []) or [])
        top_code = getattr(top_issue, "code", None) if top_issue is not None else None
        if (
            isinstance(plan_score, (int, float))
            and plan_score >= 0.95
            and top_code in {"light_direction_mismatch", "light_rotation_mismatch"}
        ):
            detect_no_progress = False

    return agent_config.model_copy(
        update={
            "strategy": AgentStrategyName.REACT,
            "max_steps": agent_config.max_steps or _REPAIR_MAX_STEPS,
            "max_steps_by_category": dict(agent_config.max_steps_by_category),
            "stop_after_scene_passed": True,
            "detect_no_progress": detect_no_progress,
            "no_progress_limit": no_progress_limit,
            "no_progress_limit_by_category": dict(agent_config.no_progress_limit_by_category),
            "detect_repeated_actions": True,
            "detect_duplicate_objects": True,
        }
    )


def _repair_start_reason(plan_trace: AgentTrace, plan_status: str) -> str:
    if plan_status == "runtime_error":
        return "runtime_error_with_snapshot"
    return "failed_validation"


def _scene_status_from_validation(val_result: Any | None) -> str | None:
    if val_result is None:
        return None
    status = getattr(val_result, "overall_status", None)
    if hasattr(status, "value"):
        return str(status.value)
    return str(status) if status is not None else None


def _plan_phase_status(plan_trace: AgentTrace) -> str:
    if plan_trace.success:
        return "completed"
    if plan_trace.error and "runtime" in str(plan_trace.error).lower():
        return "runtime_error"
    return "failed"


def _validate_via_tool(
    tool_executor: ToolExecutor,
    task: dict[str, Any],
    snap_path: Path,
) -> Any:
    result, _ = _validate_via_tool_with_reason(tool_executor, task, snap_path)
    return result


def _validate_via_tool_with_reason(
    tool_executor: ToolExecutor,
    task: dict[str, Any],
    snap_path: Path,
) -> tuple[Any, str | None]:
    from benchmark.validation.snapshot_normalization import validate_from_tool_result

    result = tool_executor.call_tool("bma_get_scene_snapshot", {})
    return validate_from_tool_result(result, task, snap_path)


def _make_tool_validator(tool_executor: ToolExecutor, task: dict[str, Any]):
    from benchmark.validation.snapshot_normalization import build_scene_validator_fn

    return build_scene_validator_fn(tool_executor, task)


def _merge_traces(
    plan_trace: AgentTrace,
    react_trace: AgentTrace,
    config: AgentConfig,
    run_id: str,
    task_id: str,
    started_at: datetime.datetime,
    *,
    post_repair_val: Any | None = None,
) -> AgentTrace:
    from benchmark.validation.models import ValidationStatus

    offset = len(plan_trace.steps)
    reindexed_react_steps = [
        AgentStep(**{**step.model_dump(), "step_index": step.step_index + offset})
        for step in react_trace.steps
    ]
    all_steps = sorted(plan_trace.steps + reindexed_react_steps, key=lambda s: s.step_index)

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    scene_passed = (
        post_repair_val is not None
        and post_repair_val.overall_status in {ValidationStatus.PASSED, ValidationStatus.WARNING}
    )
    scene_passed_but_agent_error = bool(react_trace.metadata.get("scene_passed_but_agent_error"))

    if scene_passed or scene_passed_but_agent_error:
        success = True
        error = None
    elif post_repair_val is not None:
        success = False
        error = react_trace.error
    elif react_trace.success:
        success = True
        error = None
    else:
        success = False
        error = react_trace.error

    react_meta = {
        key: react_trace.metadata[key]
        for key in (
            "react_repair_steps",
            "deterministic_repair_steps",
            "react_issue_resolution_rate",
            "react_error_type",
            "effective_max_steps",
            "scene_passed_but_agent_error",
            "early_stop_reason",
        )
        if key in react_trace.metadata
    }

    return AgentTrace(
        run_id=run_id,
        task_id=task_id,
        agent_id=config.agent_id,
        strategy=AgentStrategyName.PLAN_EXECUTE_REACT_REPAIR,
        model=config.llm.model if config.llm else None,
        steps=all_steps,
        success=success,
        error=error,
        final_message=react_trace.final_message or plan_trace.final_message,
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=(finished_at - started_at).total_seconds(),
        metadata={
            **plan_trace.metadata,
            **react_meta,
            "plan_step_count": len(plan_trace.steps),
            "repair_step_count": len(react_trace.steps),
        },
    )


def _stamp_trace(
    trace: AgentTrace,
    config: AgentConfig,
    started_at: datetime.datetime,
    *,
    plan_status: str | None = None,
    hybrid_repair_used: bool,
    repair_unavailable: bool = False,
    repair_unavailable_reason: str | None = None,
    repair_not_needed: bool = False,
    plan_scene_status: str | None = None,
    plan_score: float | None = None,
    plan_issue_count: int | None = None,
    repair_started: bool = False,
    repair_start_reason: str | None = None,
    repair_result_status: str | None = None,
    repair_result_score: Any = None,
    repair_issue_count_before: int | None = None,
    repair_issue_count_after: int | None = None,
    repair_score_before: float | None = None,
    repair_score_after: float | None = None,
    repair_improved_score: bool | None = None,
    repair_reduced_issue_count: bool | None = None,
    repair_error_type: str | None = None,
    repair_scene_status_after_error: str | None = None,
) -> AgentTrace:
    finished_at = trace.finished_at or datetime.datetime.now(datetime.timezone.utc)
    hybrid_block: dict[str, Any] = {
        "plan_status": plan_status,
        "plan_scene_status": plan_scene_status,
        "plan_score": plan_score,
        "plan_issue_count": plan_issue_count,
        "repair_started": repair_started,
        "repair_start_reason": repair_start_reason,
        "repair_unavailable": repair_unavailable,
        "repair_unavailable_reason": repair_unavailable_reason,
        "repair_not_needed": repair_not_needed,
        "repair_result_status": repair_result_status,
        "repair_result_score": repair_result_score,
        "repair_issue_count_before": repair_issue_count_before,
        "repair_issue_count_after": repair_issue_count_after,
        "repair_score_before": repair_score_before,
        "repair_score_after": repair_score_after,
        "repair_improved_score": repair_improved_score,
        "repair_reduced_issue_count": repair_reduced_issue_count,
        "repair_error_type": repair_error_type,
        "repair_scene_status_after_error": repair_scene_status_after_error,
    }
    hybrid_meta: dict[str, Any] = {
        "hybrid": hybrid_block,
        "hybrid_repair_used": hybrid_repair_used,
        "repair_unavailable": repair_unavailable,
    }
    if repair_unavailable_reason is not None:
        hybrid_meta["repair_unavailable_reason"] = repair_unavailable_reason
    if plan_scene_status is not None:
        hybrid_meta["plan_scene_status"] = plan_scene_status
    if plan_score is not None:
        hybrid_meta["plan_score"] = plan_score
    if plan_issue_count is not None:
        hybrid_meta["plan_issue_count"] = plan_issue_count
    if repair_not_needed:
        hybrid_meta["repair_not_needed"] = True
        hybrid_meta["early_stop_reason"] = "scene_passed_after_validation"
        hybrid_meta["skipped_llm_after_passed"] = True
    if hybrid_repair_used or repair_started:
        hybrid_meta["repair_started"] = True
        hybrid_meta["repair_start_reason"] = repair_start_reason or "failed_validation"
    if repair_result_status is not None:
        hybrid_meta["repair_result_status"] = repair_result_status
    if repair_result_score is not None:
        hybrid_meta["repair_result_score"] = repair_result_score
    if repair_issue_count_before is not None:
        hybrid_meta["repair_issue_count_before"] = repair_issue_count_before
    if repair_issue_count_after is not None:
        hybrid_meta["repair_issue_count_after"] = repair_issue_count_after
    if repair_score_before is not None:
        hybrid_meta["repair_score_before"] = repair_score_before
    if repair_score_after is not None:
        hybrid_meta["repair_score_after"] = repair_score_after
    if repair_improved_score is not None:
        hybrid_meta["repair_improved_score"] = repair_improved_score
    if repair_reduced_issue_count is not None:
        hybrid_meta["repair_reduced_issue_count"] = repair_reduced_issue_count
    if repair_error_type is not None:
        hybrid_meta["repair_error_type"] = repair_error_type
    if repair_scene_status_after_error is not None:
        hybrid_meta["repair_scene_status_after_error"] = repair_scene_status_after_error
    return trace.model_copy(
        update={
            "strategy": AgentStrategyName.PLAN_EXECUTE_REACT_REPAIR,
            "finished_at": finished_at,
            "duration_sec": (finished_at - started_at).total_seconds(),
            "metadata": {
                **trace.metadata,
                **hybrid_meta,
            },
        },
    )
