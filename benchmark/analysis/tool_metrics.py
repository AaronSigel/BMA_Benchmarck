from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel, Field

from benchmark.agent.models import AgentStepType, AgentTrace
from benchmark.analysis.models import ToolCallMetric
from benchmark.mcp.tool_contract import TOOL_CONTRACT_MAP, ToolCategory

# Pattern that identifies a "tool is disabled / not allowed" error
_DISABLED_PATTERN = re.compile(r"not allowed|is disabled|blocked|not permitted", re.I)

# Tool categories that represent scene-mutating operations
_MUTATION_CATEGORIES: frozenset[ToolCategory] = frozenset({
    ToolCategory.OBJECT,
    ToolCategory.TRANSFORM,
    ToolCategory.MATERIAL,
    ToolCategory.LIGHT,
    ToolCategory.CAMERA,
    ToolCategory.EXPORT,
})


def _category_of(tool_name: str) -> ToolCategory:
    contract = TOOL_CONTRACT_MAP.get(tool_name)
    if contract is None:
        # execute_blender_code may not be in TOOL_CONTRACT_MAP in all configs
        if tool_name == "execute_blender_code":
            return ToolCategory.PYTHON
        return ToolCategory.OTHER
    return contract.category


def _is_disabled_error(error: str) -> bool:
    return bool(_DISABLED_PATTERN.search(error))


# ---------------------------------------------------------------------------
# Summary model
# ---------------------------------------------------------------------------


class ToolCallSummary(BaseModel):
    """All tool-call metrics derived from a single AgentTrace."""

    tool_call_count: int = Field(default=0, ge=0)
    unique_tool_count: int = Field(default=0, ge=0)
    invalid_tool_call_count: int = Field(default=0, ge=0)
    disabled_tool_call_count: int = Field(default=0, ge=0)
    tool_error_count: int = Field(default=0, ge=0)
    trajectory_length: int = Field(default=0, ge=0)
    inspection_tool_count: int = Field(default=0, ge=0)
    mutation_tool_count: int = Field(default=0, ge=0)
    python_tool_call_count: int = Field(default=0, ge=0)
    asset_tool_call_count: int = Field(default=0, ge=0)
    tool_repetition_count: int = Field(default=0, ge=0)
    average_tool_duration_sec: float | None = Field(default=None, ge=0.0)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_tool_summary(trace: AgentTrace) -> ToolCallSummary:
    """Compute all tool-call metrics from a trace.

    Returns a ToolCallSummary with zero values for an empty trace — never raises.
    """
    trajectory_length = len(trace.steps)
    tool_steps = [s for s in trace.steps if s.step_type == AgentStepType.TOOL_CALL]

    if not tool_steps:
        return ToolCallSummary(trajectory_length=trajectory_length)

    tool_call_count = len(tool_steps)
    unique_tools: set[str] = set()
    tool_error_count = 0
    disabled_tool_call_count = 0
    invalid_tool_call_count = 0
    inspection_tool_count = 0
    mutation_tool_count = 0
    python_tool_call_count = 0
    asset_tool_call_count = 0
    tool_repetition_count = 0
    durations: list[float] = []

    prev_tool_name: str | None = None

    for step in tool_steps:
        name = step.tool_name or ""
        unique_tools.add(name)

        # Repetition: same tool called consecutively
        if name == prev_tool_name:
            tool_repetition_count += 1
        prev_tool_name = name

        # Duration
        if step.duration_sec is not None:
            durations.append(step.duration_sec)

        # Error classification
        if step.error:
            tool_error_count += 1
            if _is_disabled_error(step.error):
                disabled_tool_call_count += 1
            else:
                invalid_tool_call_count += 1

        # Category classification
        category = _category_of(name)
        if category == ToolCategory.INSPECTION:
            inspection_tool_count += 1
        elif category in _MUTATION_CATEGORIES:
            mutation_tool_count += 1
        elif category == ToolCategory.PYTHON:
            python_tool_call_count += 1
        elif category == ToolCategory.ASSET:
            asset_tool_call_count += 1

    avg_duration = sum(durations) / len(durations) if durations else None

    return ToolCallSummary(
        tool_call_count=tool_call_count,
        unique_tool_count=len(unique_tools),
        invalid_tool_call_count=invalid_tool_call_count,
        disabled_tool_call_count=disabled_tool_call_count,
        tool_error_count=tool_error_count,
        trajectory_length=trajectory_length,
        inspection_tool_count=inspection_tool_count,
        mutation_tool_count=mutation_tool_count,
        python_tool_call_count=python_tool_call_count,
        asset_tool_call_count=asset_tool_call_count,
        tool_repetition_count=tool_repetition_count,
        average_tool_duration_sec=avg_duration,
    )


# ---------------------------------------------------------------------------
# Per-tool breakdown (used by agent_metrics and report_builder)
# ---------------------------------------------------------------------------


def extract_tool_metrics(trace: AgentTrace) -> list[ToolCallMetric]:
    """Return aggregated per-tool statistics from a trace."""
    totals: dict[str, int] = defaultdict(int)
    succeeded: dict[str, int] = defaultdict(int)
    failed: dict[str, int] = defaultdict(int)
    durations: dict[str, list[float]] = defaultdict(list)

    for step in trace.steps:
        if step.step_type != AgentStepType.TOOL_CALL or not step.tool_name:
            continue
        name = step.tool_name
        totals[name] += 1
        if step.duration_sec is not None:
            durations[name].append(step.duration_sec)
        if step.error:
            failed[name] += 1
        else:
            succeeded[name] += 1

    result: list[ToolCallMetric] = []
    for name in totals:
        total = totals[name]
        s = succeeded[name]
        f = failed[name]
        dur_list = durations[name]
        total_dur = sum(dur_list)
        avg_dur = total_dur / len(dur_list) if dur_list else None
        result.append(
            ToolCallMetric(
                tool_name=name,
                total_calls=total,
                succeeded=s,
                failed=f,
                success_rate=s / total if total > 0 else 0.0,
                avg_duration_sec=avg_dur,
                total_duration_sec=total_dur,
            )
        )
    return sorted(result, key=lambda m: m.tool_name)


# ---------------------------------------------------------------------------
# Convenience counts (kept for backward compatibility)
# ---------------------------------------------------------------------------


def tool_call_count(trace: AgentTrace, tool_name: str | None = None) -> int:
    return sum(
        1
        for step in trace.steps
        if step.step_type == AgentStepType.TOOL_CALL
        and (tool_name is None or step.tool_name == tool_name)
    )


def tool_failure_count(trace: AgentTrace, tool_name: str | None = None) -> int:
    return sum(
        1
        for step in trace.steps
        if step.step_type == AgentStepType.TOOL_CALL
        and step.error is not None
        and (tool_name is None or step.tool_name == tool_name)
    )
