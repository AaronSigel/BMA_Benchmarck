from pathlib import Path

from benchmark.agent.llm import LlmResponse, MockLlmClient
from benchmark.agent.models import AgentConfig, AgentStepType, AgentStrategyName, LlmConfig
from benchmark.agent.runtime import AgentRuntime
from benchmark.agent.strategies import ReactStrategy, create_agent_strategy
from benchmark.agent.tool_context import AgentToolContext
from benchmark.agent.tool_executor import MockToolExecutor


def make_config(max_steps: int = 10) -> AgentConfig:
    return AgentConfig(
        agent_id="agent-1",
        strategy=AgentStrategyName.REACT,
        llm=LlmConfig(provider="mock", model="mock"),
        mcp_profile="minimal",
        max_steps=max_steps,
    )


def test_react_strategy_runs_multi_step_mock_scenario() -> None:
    llm = MockLlmClient(
        [
            LlmResponse(
                content=(
                    '{"thought":"Need to inspect first.",'
                    '"action":{"tool":"get_scene_info","arguments":{}}}'
                )
            ),
            LlmResponse(content='{"final_answer":"Scene inspected."}'),
        ]
    )
    executor = MockToolExecutor(results={"get_scene_info": {"objects": []}})

    trace = ReactStrategy().run(
        {"id": "task-1", "prompt": "Inspect scene"},
        make_config(),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.success is True
    assert trace.final_message == "Scene inspected."
    assert [step.step_type for step in trace.steps] == [
        AgentStepType.LLM_CALL,
        AgentStepType.TOOL_CALL,
        AgentStepType.LLM_CALL,
        AgentStepType.FINAL,
    ]
    assert trace.steps[0].thought == "Need to inspect first."
    assert trace.steps[1].observation == {"objects": []}


def test_react_strategy_stops_infinite_loop_at_max_steps() -> None:
    llm = MockLlmClient(
        [
            LlmResponse(content='{"thought":"Again","action":{"tool":"get_scene_info","arguments":{}}}'),
            LlmResponse(content='{"thought":"Again","action":{"tool":"get_scene_info","arguments":{}}}'),
            LlmResponse(content='{"thought":"Again","action":{"tool":"get_scene_info","arguments":{}}}'),
        ]
    )
    executor = MockToolExecutor(results={"get_scene_info": {"objects": []}})

    trace = ReactStrategy().run(
        {"id": "task-1", "prompt": "Inspect scene"},
        make_config(max_steps=3),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.success is False
    assert trace.error is not None
    assert "max_steps" in trace.error
    assert len(trace.steps) <= 4


def test_react_strategy_records_tool_error_as_observation() -> None:
    llm = MockLlmClient(
        [
            LlmResponse(
                content=(
                    '{"thought":"Try forbidden tool.",'
                    '"action":{"tool":"execute_blender_code","arguments":{"code":"print(1)"}}}'
                )
            )
        ]
    )
    executor = MockToolExecutor(allowed_tools=["get_scene_info"])

    trace = ReactStrategy().run(
        {"id": "task-1", "prompt": "Run code"},
        make_config(),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.success is False
    assert trace.steps[1].tool_name == "execute_blender_code"
    assert trace.steps[1].observation == {"error": trace.steps[1].error}
    assert "not allowed" in (trace.error or "")


def test_react_strategy_final_answer_completes_trace() -> None:
    llm = MockLlmClient([LlmResponse(content='{"final_answer":"Done."}')])

    trace = ReactStrategy().run(
        {"id": "task-1", "prompt": "Inspect scene"},
        make_config(),
        llm,
        MockToolExecutor(),
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.success is True
    assert trace.final_message == "Done."
    assert trace.steps[-1].step_type == AgentStepType.FINAL


def test_runtime_selects_react_strategy() -> None:
    runtime = AgentRuntime(
        make_config(),
        llm_client=MockLlmClient([LlmResponse(content='{"final_answer":"Done."}')]),
        tool_executor=MockToolExecutor(),
    )

    result = runtime.run(task_id="task-1", task={"prompt": "Inspect"})

    assert isinstance(create_agent_strategy(AgentStrategyName.REACT), ReactStrategy)
    assert result.ok is True
    assert result.status.value == "passed"
