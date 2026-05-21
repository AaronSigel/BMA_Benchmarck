"""Hybrid plan_execute_react_repair strategy.

Phase 1: PlanAndExecuteStrategy builds the primary scene.
Phase 2: If validation fails, ReactStrategy runs a capped repair loop (max 5 steps).
The two traces are merged into one AgentTrace with hybrid metadata.
"""
from __future__ import annotations

import datetime
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from benchmark.agent.llm.base import LlmClient
from benchmark.agent.models import AgentConfig, AgentStep, AgentStepType, AgentStrategyName, AgentTrace
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

        # --- Phase 1: plan-and-execute ---
        log.info("[hybrid:%s] phase 1 — plan_and_execute", run_id[:8])
        from benchmark.agent.strategies.plan_and_execute import PlanAndExecuteStrategy

        plan_config = agent_config.model_copy(update={"strategy": AgentStrategyName.PLAN_AND_EXECUTE})
        plan_strategy = PlanAndExecuteStrategy(
            prompt_builder=self.prompt_builder,
            tool_schema_provider=self.tool_schema_provider,
        )
        plan_trace = plan_strategy.run(task, plan_config, llm_client, tool_executor, tool_context, output_dir)

        # --- Phase 2: validate post-plan scene ---
        log.info("[hybrid:%s] phase 2 — post-plan validation", run_id[:8])
        val_result, val_unavailable_reason = _validate_via_tool_with_reason(
            tool_executor, task, output_dir / "hybrid_post_plan_snapshot.json"
        )

        if val_result is None:
            log.info("[hybrid:%s] validation unavailable (%s), returning plan trace", run_id[:8], val_unavailable_reason)
            return _stamp_trace(
                plan_trace, agent_config, started_at,
                hybrid_repair_used=False,
                repair_unavailable=True,
                repair_unavailable_reason=val_unavailable_reason or "validation_unavailable",
                plan_scene_status=None,
                plan_score=None,
                plan_issue_count=None,
            )

        from benchmark.validation.models import ValidationStatus
        _plan_status = str(val_result.overall_status.value if hasattr(val_result.overall_status, "value") else val_result.overall_status)
        _plan_score = val_result.total_score
        _plan_issue_count = len(val_result.issues)

        if val_result.overall_status in {ValidationStatus.PASSED, ValidationStatus.WARNING}:
            log.info("[hybrid:%s] scene passed after plan, no repair needed (score=%.3f)", run_id[:8], val_result.total_score)
            return _stamp_trace(
                plan_trace, agent_config, started_at,
                hybrid_repair_used=False,
                plan_scene_status=_plan_status,
                plan_score=_plan_score,
                plan_issue_count=_plan_issue_count,
            )

        log.info(
            "[hybrid:%s] scene needs repair (status=%s, score=%.3f, issues=%d) — starting react repair",
            run_id[:8], val_result.overall_status, val_result.total_score, len(val_result.issues),
        )

        # --- Phase 3: react repair loop ---
        from benchmark.agent.strategies.react import ReactStrategy

        repair_config = agent_config.model_copy(update={
            "strategy": AgentStrategyName.REACT,
            "max_steps": _REPAIR_MAX_STEPS,
            "stop_after_scene_passed": True,
            "detect_no_progress": True,
            "no_progress_limit": 2,
            "detect_repeated_actions": True,
            "detect_duplicate_objects": True,
        })
        react_strategy = ReactStrategy(
            prompt_builder=self.prompt_builder,
            tool_schema_provider=self.tool_schema_provider,
        )
        # Inject tool-based scene validator so the repair loop can track progress.
        react_strategy.scene_validator_fn = _make_tool_validator(tool_executor, task)

        react_trace = react_strategy.run(task, repair_config, llm_client, tool_executor, tool_context, output_dir)

        # --- Merge traces ---
        merged = _merge_traces(plan_trace, react_trace, agent_config, run_id, task_id, started_at)
        _repair_status = str(
            react_trace.metadata.get("effective_max_steps") and (
                "passed" if react_trace.success else "failed"
            ) or ("passed" if react_trace.success else "failed")
        )
        return _stamp_trace(
            merged, agent_config, started_at,
            hybrid_repair_used=True,
            plan_scene_status=_plan_status,
            plan_score=_plan_score,
            plan_issue_count=_plan_issue_count,
            repair_result_status=_repair_status,
            repair_result_score=react_trace.metadata.get("react_issue_resolution_rate"),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_via_tool(
    tool_executor: ToolExecutor,
    task: dict[str, Any],
    snap_path: Path,
) -> Any:
    """Call bma_get_scene_snapshot and run SceneValidator. Returns SceneValidationResult or None."""
    result, _ = _validate_via_tool_with_reason(tool_executor, task, snap_path)
    return result


def _validate_via_tool_with_reason(
    tool_executor: ToolExecutor,
    task: dict[str, Any],
    snap_path: Path,
) -> tuple[Any, str | None]:
    """Like _validate_via_tool but also returns the reason validation was unavailable."""
    try:
        result = tool_executor.call_tool("bma_get_scene_snapshot", {})
        if result.error:
            return None, "snapshot_tool_failed"
        if not isinstance(result.result, dict):
            return None, "snapshot_invalid_schema"
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        snap_path.write_text(json.dumps(result.result))
        try:
            from benchmark.blender.models import SceneSnapshot
            from benchmark.tasks.models import BenchmarkTask
            from benchmark.validation.scene_validator import SceneValidator
            snapshot = SceneSnapshot.model_validate(result.result)
        except Exception:
            return None, "snapshot_invalid_schema"
        try:
            task_obj = BenchmarkTask.model_validate(task)
        except Exception:
            return None, "task_parse_failed"
        try:
            val_result = SceneValidator().validate(task_obj, snapshot)
            return val_result, None
        except Exception:
            return None, "validation_exception"
    except Exception as exc:
        log.debug("hybrid: validation via tool failed: %s", exc)
        return None, "tool_disabled"


def _make_tool_validator(tool_executor: ToolExecutor, task: dict[str, Any]):
    """Build a scene_validator_fn that uses bma_get_scene_snapshot (no adapter required)."""
    def _fn(snap_path: Path) -> tuple[bool, float | None, Any]:
        try:
            result = tool_executor.call_tool("bma_get_scene_snapshot", {})
            if result.error or not isinstance(result.result, dict):
                return False, None, None
            snap_path.parent.mkdir(parents=True, exist_ok=True)
            snap_path.write_text(json.dumps(result.result))
            from benchmark.blender.models import SceneSnapshot
            from benchmark.tasks.models import BenchmarkTask
            from benchmark.validation.scene_validator import SceneValidator
            from benchmark.validation.models import ValidationStatus
            snapshot = SceneSnapshot.model_validate(result.result)
            task_obj = BenchmarkTask.model_validate(task)
            val_result = SceneValidator().validate(task_obj, snapshot)
            scene_ok = val_result.overall_status in {ValidationStatus.PASSED, ValidationStatus.WARNING}
            return scene_ok, val_result.total_score, val_result
        except Exception as exc:
            log.debug("hybrid: tool validator failed: %s", exc)
            return False, None, None
    return _fn


def _merge_traces(
    plan_trace: AgentTrace,
    react_trace: AgentTrace,
    config: AgentConfig,
    run_id: str,
    task_id: str,
    started_at: datetime.datetime,
) -> AgentTrace:
    """Combine plan_trace steps + react_trace steps into a single AgentTrace."""
    offset = len(plan_trace.steps)
    reindexed_react_steps = [
        AgentStep(**{**step.model_dump(), "step_index": step.step_index + offset})
        for step in react_trace.steps
    ]
    all_steps = sorted(plan_trace.steps + reindexed_react_steps, key=lambda s: s.step_index)

    finished_at = datetime.datetime.now(datetime.timezone.utc)
    success = react_trace.success if react_trace.success is not None else plan_trace.success
    error = react_trace.error if not success else None

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
            "plan_step_count": len(plan_trace.steps),
            "repair_step_count": len(react_trace.steps),
        },
    )


def _stamp_trace(
    trace: AgentTrace,
    config: AgentConfig,
    started_at: datetime.datetime,
    *,
    hybrid_repair_used: bool,
    repair_unavailable: bool = False,
    repair_unavailable_reason: str | None = None,
    plan_scene_status: str | None = None,
    plan_score: float | None = None,
    plan_issue_count: int | None = None,
    repair_result_status: str | None = None,
    repair_result_score: Any = None,
) -> AgentTrace:
    finished_at = trace.finished_at or datetime.datetime.now(datetime.timezone.utc)
    hybrid_meta: dict[str, Any] = {
        "hybrid_repair_used": hybrid_repair_used,
        "repair_unavailable": repair_unavailable,
    }
    if repair_unavailable and repair_unavailable_reason is not None:
        hybrid_meta["repair_unavailable_reason"] = repair_unavailable_reason
    if plan_scene_status is not None:
        hybrid_meta["plan_scene_status"] = plan_scene_status
    if plan_score is not None:
        hybrid_meta["plan_score"] = plan_score
    if plan_issue_count is not None:
        hybrid_meta["plan_issue_count"] = plan_issue_count
    if hybrid_repair_used:
        hybrid_meta["repair_started"] = True
        hybrid_meta["repair_start_reason"] = "plan_failed_validation"
    if repair_result_status is not None:
        hybrid_meta["repair_result_status"] = repair_result_status
    if repair_result_score is not None:
        hybrid_meta["repair_result_score"] = repair_result_score
    return trace.model_copy(
        update={
            "strategy": AgentStrategyName.PLAN_EXECUTE_REACT_REPAIR,
            "finished_at": finished_at,
            "duration_sec": (finished_at - started_at).total_seconds(),
            "metadata": {
                **trace.metadata,
                **hybrid_meta,
            },
        }
    )
