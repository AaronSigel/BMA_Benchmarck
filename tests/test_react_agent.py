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
                    '"action":{"tool":"get_scene_info","arguments":{}},"finish":false}'
                )
            ),
            LlmResponse(content='{"thought":"Scene inspected.","action":null,"finish":true}'),
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
            LlmResponse(content='{"thought":"Again","action":{"tool":"get_scene_info","arguments":{}},"finish":false}'),
            LlmResponse(content='{"thought":"Again","action":{"tool":"get_scene_info","arguments":{}},"finish":false}'),
            LlmResponse(content='{"thought":"Again","action":{"tool":"get_scene_info","arguments":{}},"finish":false}'),
        ]
    )
    executor = MockToolExecutor(results={"get_scene_info": {"objects": []}})

    trace = ReactStrategy().run(
        {"id": "task-1", "prompt": "Inspect scene"},
        make_config(max_steps=3).model_copy(update={"detect_repeated_actions": False}),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.success is False
    assert trace.error is not None
    assert "ReactMaxSteps" in trace.error or "max_steps" in trace.error
    assert trace.metadata.get("react_iterations_total") == 3
    assert len(trace.steps) > 3


def test_react_strategy_records_tool_error_as_observation() -> None:
    llm = MockLlmClient(
        [
            LlmResponse(
                content=(
                    '{"thought":"Try forbidden tool.",'
                    '"action":{"tool":"execute_blender_code","arguments":{"code":"print(1)"}},"finish":false}'
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
    llm = MockLlmClient([LlmResponse(content='{"thought":"Done.","action":null,"finish":true}')])

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


def test_react_strategy_rejects_plain_text_pseudo_tool_call() -> None:
    """Pseudo tool calls in plain text are non-strict ReAct responses."""
    llm = MockLlmClient(
        [
            LlmResponse(
                content='Tool: bma_create_object\nArguments: {"name": "Cube", "type": "cube"}'
            ),
            LlmResponse(content='{"thought":"Done.","action":null,"finish":true}'),
        ]
    )

    trace = ReactStrategy().run(
        {"id": "task-1", "prompt": "Create a cube"},
        make_config(),
        llm,
        MockToolExecutor(),
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.success is False
    assert trace.error == "LlmParseError"
    tool_steps = [s for s in trace.steps if s.step_type == AgentStepType.TOOL_CALL]
    assert len(tool_steps) == 0


def test_react_strategy_rejects_fenced_and_mixed_json() -> None:
    responses = [
        '```json\n{"thought":"Create","action":{"tool":"bma_create_object","arguments":{}},"finish":false}\n```',
        'First I will create it. {"thought":"Create","action":{"tool":"bma_create_object","arguments":{}},"finish":false}',
    ]

    for content in responses:
        trace = ReactStrategy().run(
            {"id": "task-1", "prompt": "Create a cube"},
            make_config(),
            MockLlmClient([LlmResponse(content=content)]),
            MockToolExecutor(),
            AgentToolContext(run_id="run-1", task_id="task-1"),
            Path("."),
        )

        assert trace.success is False
        assert trace.error == "LlmParseError"
        assert trace.metadata.get("react_error_type") == "LlmParseError"


def test_runtime_selects_react_strategy() -> None:
    runtime = AgentRuntime(
        make_config(),
        llm_client=MockLlmClient([LlmResponse(content='{"thought":"Done.","action":null,"finish":true}')]),
        tool_executor=MockToolExecutor(),
    )

    result = runtime.run(task_id="task-1", task={"prompt": "Inspect"})

    assert isinstance(create_agent_strategy(AgentStrategyName.REACT), ReactStrategy)
    assert result.ok is True
    assert result.status.value == "passed"
