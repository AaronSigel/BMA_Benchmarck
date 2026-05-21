from __future__ import annotations

from pathlib import Path

from benchmark.agent.errors import AgentTraceError
from typing import Any

from benchmark.agent.models import AgentStep, AgentStepType, AgentTrace


def read_agent_trace(path: Path | str) -> AgentTrace:
    trace_path = Path(path)
    try:
        return AgentTrace.model_validate_json(trace_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AgentTraceError(f"Failed to read agent trace {trace_path}: {error}") from error
    except ValueError as error:
        raise AgentTraceError(f"Invalid agent trace in {trace_path}: {error}") from error


def write_agent_trace(trace: AgentTrace, path: Path | str) -> None:
    trace_path = Path(path)
    try:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    except OSError as error:
        raise AgentTraceError(f"Failed to write agent trace {trace_path}: {error}") from error


def summarize_trace(trace: AgentTrace) -> dict[str, Any]:
    tool_steps = [step for step in trace.steps if step.step_type == AgentStepType.TOOL_CALL]
    tool_names = [step.tool_name for step in tool_steps if step.tool_name]
    llm_steps = [step for step in trace.steps if step.step_type in {AgentStepType.LLM_CALL, AgentStepType.PLAN}]
    prompt_tokens = 0
    completion_tokens = 0
    found_tokens = False
    provider_name = None
    provider_cost = 0.0
    found_cost = False
    for step in llm_steps:
        raw = step.raw_llm_response if isinstance(step.raw_llm_response, dict) else {}
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else raw
        if not isinstance(usage, dict):
            continue
        if usage.get("prompt_tokens") is not None or usage.get("input_tokens") is not None:
            prompt_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            found_tokens = True
        if usage.get("completion_tokens") is not None or usage.get("output_tokens") is not None:
            completion_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            found_tokens = True
        if provider_name is None and usage.get("provider_name"):
            provider_name = str(usage.get("provider_name"))
        if usage.get("cost") is not None:
            provider_cost += float(usage.get("cost") or 0.0)
            found_cost = True
    total_tokens = prompt_tokens + completion_tokens if found_tokens else None
    return {
        "run_id": trace.run_id,
        "task_id": trace.task_id,
        "agent_id": trace.agent_id,
        "strategy": trace.strategy.value,
        "success": trace.success,
        "steps_count": len(trace.steps),
        "tool_calls_count": sum(1 for step in trace.steps if step.step_type == AgentStepType.TOOL_CALL),
        "errors_count": sum(1 for step in trace.steps if step.error is not None),
        "duration_sec": trace.duration_sec,
        "metrics": {
            "llm_call_count": len(llm_steps),
            "tool_call_count": len(tool_steps),
            "unique_tool_count": len(set(tool_names)),
            "invalid_tool_call_count": sum(1 for step in tool_steps if step.tool_name is None),
            "disabled_tool_call_count": sum(1 for step in tool_steps if "disabled" in (step.error or "").lower()),
            "tool_error_count": sum(1 for step in tool_steps if step.error is not None),
            "retry_count": int(trace.metadata.get("retry_count", 0)),
            "prompt_tokens": prompt_tokens if found_tokens else None,
            "completion_tokens": completion_tokens if found_tokens else None,
            "total_tokens": total_tokens,
            "provider_name": provider_name,
            "provider_reported_prompt_tokens": prompt_tokens if found_tokens else None,
            "provider_reported_completion_tokens": completion_tokens if found_tokens else None,
            "provider_reported_total_tokens": total_tokens,
            "provider_reported_cost_usd": provider_cost if found_cost else None,
            "provider_cost_available": found_cost,
            "duplicate_object_count": int(trace.metadata.get("duplicate_object_count", 0)),
            "repeated_action_count": int(trace.metadata.get("repeated_action_count", 0)),
            "wasted_step_count": int(trace.metadata.get("wasted_step_count", 0)),
            "no_progress_step_count": int(trace.metadata.get("no_progress_step_count", 0)),
        },
    }


