from __future__ import annotations

from pathlib import Path

from benchmark.agent.errors import AgentTraceError
from typing import Any

from benchmark.agent.models import AgentStep, AgentStepType, AgentTrace


AgentTraceStep = AgentStep


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
    }


def load_agent_trace(path: Path | str) -> AgentTrace:
    return read_agent_trace(path)


def dump_agent_trace(trace: AgentTrace, path: Path | str) -> None:
    write_agent_trace(trace, path)
