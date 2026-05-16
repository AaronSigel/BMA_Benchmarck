from pathlib import Path

from benchmark.agent.models import AgentStep, AgentStepType, AgentStrategyName, AgentTrace
from benchmark.agent.trace import read_agent_trace, summarize_trace, write_agent_trace


def make_trace() -> AgentTrace:
    return AgentTrace(
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        strategy=AgentStrategyName.DIRECT_TOOL_CALLING,
        success=False,
        duration_sec=1.25,
        steps=[
            AgentStep(step_index=0, step_type=AgentStepType.LLM_CALL),
            AgentStep(
                step_index=1,
                step_type=AgentStepType.TOOL_CALL,
                tool_name="get_scene_info",
                observation={"objects": []},
            ),
            AgentStep(step_index=2, step_type=AgentStepType.ERROR, error="failed"),
        ],
    )


def test_write_agent_trace_saves_pretty_json(tmp_path: Path) -> None:
    path = tmp_path / "agent_trace.json"

    write_agent_trace(make_trace(), path)

    content = path.read_text(encoding="utf-8")
    assert content.startswith("{\n")
    assert '  "run_id": "run-1"' in content


def test_read_agent_trace_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "agent_trace.json"
    trace = make_trace()

    write_agent_trace(trace, path)
    loaded = read_agent_trace(path)

    assert loaded == trace


def test_summarize_trace_contains_required_counts() -> None:
    summary = summarize_trace(make_trace())

    assert summary["steps_count"] == 3
    assert summary["tool_calls_count"] == 1
    assert summary["errors_count"] == 1
    assert summary["duration_sec"] == 1.25
