from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from benchmark.agent.models import AgentStepType, AgentTrace
from benchmark.analysis.models import RunAnalysisResult
from benchmark.analysis.tool_metrics import compute_tool_summary, extract_tool_metrics

_STEP_LIMIT_PATTERN = re.compile(r"max_steps|step.?limit", re.I)


# ---------------------------------------------------------------------------
# Summary model
# ---------------------------------------------------------------------------


class AgentMetricsSummary(BaseModel):
    """All agent-level metrics derived from a single AgentTrace."""

    # Step-type counters
    llm_call_count: int = Field(default=0, ge=0)
    planning_step_count: int = Field(default=0, ge=0)
    observation_count: int = Field(default=0, ge=0)
    final_step_present: bool = False

    # Error / retry
    retry_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)

    # Duration
    duration_sec: float | None = Field(default=None, ge=0.0)
    average_step_duration_sec: float | None = Field(default=None, ge=0.0)
    max_step_duration_sec: float | None = Field(default=None, ge=0.0)

    # Control-flow
    step_limit_reached: bool = False
    self_correction_attempts: int = Field(default=0, ge=0)
    tool_error_recovery_count: int = Field(default=0, ge=0)

    # Token usage (optional — only present when the LLM reported usage)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0.0)


# ---------------------------------------------------------------------------
# Token extraction helpers
# ---------------------------------------------------------------------------


def _extract_usage_from_step(step_metadata: dict[str, Any] | None, raw_llm: Any) -> dict[str, int | None]:
    """Try to extract token usage from a step's raw_llm_response dict."""
    usage: dict[str, int | None] = {}
    src: dict[str, Any] | None = None

    if isinstance(raw_llm, dict):
        src = raw_llm.get("usage") or raw_llm
    if src is None and isinstance(step_metadata, dict):
        src = step_metadata.get("usage")

    if isinstance(src, dict):
        usage["prompt_tokens"] = src.get("prompt_tokens") or src.get("input_tokens")
        usage["completion_tokens"] = src.get("completion_tokens") or src.get("output_tokens")
        usage["total_tokens"] = src.get("total_tokens")
        if usage["prompt_tokens"] and usage["completion_tokens"] and not usage["total_tokens"]:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    return usage


def _aggregate_token_usage(trace: AgentTrace) -> tuple[int | None, int | None, int | None]:
    """Sum token usage across all LLM-originated steps."""
    total_prompt: int = 0
    total_completion: int = 0
    found_any = False

    for step in trace.steps:
        if step.step_type not in {AgentStepType.LLM_CALL, AgentStepType.PLAN}:
            continue
        usage = _extract_usage_from_step(step.metadata, step.raw_llm_response)
        p = usage.get("prompt_tokens")
        c = usage.get("completion_tokens")
        if p is not None:
            total_prompt += int(p)
            found_any = True
        if c is not None:
            total_completion += int(c)
            found_any = True

    if not found_any:
        return None, None, None
    total = total_prompt + total_completion
    return total_prompt or None, total_completion or None, total or None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_agent_summary(trace: AgentTrace) -> AgentMetricsSummary:
    """Compute all agent-level metrics from a trace. Never raises."""
    steps = trace.steps

    # Step-type counters
    llm_call_count = 0
    planning_step_count = 0
    observation_count = 0
    final_step_present = False
    error_count = 0

    # Duration accumulators
    step_durations: list[float] = []

    # Control-flow detection
    step_limit_reached = False
    self_correction_attempts = 0
    tool_error_recovery_count = 0

    for i, step in enumerate(steps):
        t = step.step_type

        if t == AgentStepType.LLM_CALL:
            llm_call_count += 1
        elif t == AgentStepType.PLAN:
            planning_step_count += 1
            llm_call_count += 1
        elif t == AgentStepType.OBSERVATION:
            observation_count += 1
        elif t == AgentStepType.FINAL:
            final_step_present = True

        if step.error is not None:
            error_count += 1
            # Self-correction: the next step is an LLM_CALL (agent reflects on error)
            if i + 1 < len(steps) and steps[i + 1].step_type == AgentStepType.LLM_CALL:
                self_correction_attempts += 1
            # Tool-error recovery: tool failed and next step is a tool call
            if t == AgentStepType.TOOL_CALL and i + 1 < len(steps):
                next_step = steps[i + 1]
                if next_step.step_type == AgentStepType.TOOL_CALL:
                    tool_error_recovery_count += 1

        if step.duration_sec is not None:
            step_durations.append(step.duration_sec)

    # Step-limit detection: trace error message or last-step error
    trace_error = trace.error or ""
    if _STEP_LIMIT_PATTERN.search(trace_error):
        step_limit_reached = True
    elif steps:
        last_err = steps[-1].error or ""
        if _STEP_LIMIT_PATTERN.search(last_err):
            step_limit_reached = True

    # Duration aggregates
    avg_step_dur = sum(step_durations) / len(step_durations) if step_durations else None
    max_step_dur = max(step_durations) if step_durations else None

    # Retry count from trace metadata
    retry_count = int(trace.metadata.get("retry_count", 0))

    # Token usage
    prompt_tokens, completion_tokens, total_tokens = _aggregate_token_usage(trace)
    estimated_cost: float | None = None
    if total_tokens is not None:
        # Rough estimate: $2 per 1M tokens (mid-range model cost)
        estimated_cost = total_tokens * 2e-6

    return AgentMetricsSummary(
        llm_call_count=llm_call_count,
        planning_step_count=planning_step_count,
        observation_count=observation_count,
        final_step_present=final_step_present,
        retry_count=retry_count,
        error_count=error_count,
        duration_sec=trace.duration_sec,
        average_step_duration_sec=avg_step_dur,
        max_step_duration_sec=max_step_dur,
        step_limit_reached=step_limit_reached,
        self_correction_attempts=self_correction_attempts,
        tool_error_recovery_count=tool_error_recovery_count,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost=estimated_cost,
    )


# ---------------------------------------------------------------------------
# RunAnalysisResult builder
# ---------------------------------------------------------------------------


def extract_agent_metrics(trace: AgentTrace) -> RunAnalysisResult:
    """Build a RunAnalysisResult from a trace (validation fields left blank)."""
    agent_summary = compute_agent_summary(trace)
    tool_summary = compute_tool_summary(trace)
    per_tool = extract_tool_metrics(trace)

    metrics: dict[str, float | str | int | bool] = {
        # Agent metrics
        "llm_call_count": agent_summary.llm_call_count,
        "planning_step_count": agent_summary.planning_step_count,
        "observation_count": agent_summary.observation_count,
        "final_step_present": agent_summary.final_step_present,
        "retry_count": agent_summary.retry_count,
        "step_limit_reached": agent_summary.step_limit_reached,
        "self_correction_attempts": agent_summary.self_correction_attempts,
        "tool_error_recovery_count": agent_summary.tool_error_recovery_count,
        # Tool metrics
        "tool_call_count": tool_summary.tool_call_count,
        "unique_tool_count": tool_summary.unique_tool_count,
        "invalid_tool_call_count": tool_summary.invalid_tool_call_count,
        "disabled_tool_call_count": tool_summary.disabled_tool_call_count,
        "tool_error_count": tool_summary.tool_error_count,
        "inspection_tool_count": tool_summary.inspection_tool_count,
        "mutation_tool_count": tool_summary.mutation_tool_count,
        "python_tool_call_count": tool_summary.python_tool_call_count,
        "asset_tool_call_count": tool_summary.asset_tool_call_count,
        "tool_repetition_count": tool_summary.tool_repetition_count,
    }

    # Optional numeric metrics
    for key, val in [
        ("average_step_duration_sec", agent_summary.average_step_duration_sec),
        ("max_step_duration_sec", agent_summary.max_step_duration_sec),
        ("average_tool_duration_sec", tool_summary.average_tool_duration_sec),
    ]:
        if val is not None:
            metrics[key] = val

    # Optional token metrics
    for key, val in [
        ("prompt_tokens", agent_summary.prompt_tokens),
        ("completion_tokens", agent_summary.completion_tokens),
        ("total_tokens", agent_summary.total_tokens),
        ("estimated_cost", agent_summary.estimated_cost),
    ]:
        if val is not None:
            metrics[key] = val

    # Per-tool breakdown
    for tm in per_tool:
        metrics[f"tool.{tm.tool_name}.calls"] = tm.total_calls
        metrics[f"tool.{tm.tool_name}.success_rate"] = tm.success_rate

    return RunAnalysisResult(
        run_id=trace.run_id,
        task_id=trace.task_id,
        agent_id=trace.agent_id,
        strategy=trace.strategy.value,
        model=trace.model,
        tool_call_count=tool_summary.tool_call_count,
        invalid_tool_call_count=tool_summary.invalid_tool_call_count,
        trajectory_length=tool_summary.trajectory_length,
        retry_count=agent_summary.retry_count,
        llm_call_count=agent_summary.llm_call_count,
        error_count=agent_summary.error_count,
        duration_sec=trace.duration_sec,
        success=trace.success,
        metrics=metrics,
    )
