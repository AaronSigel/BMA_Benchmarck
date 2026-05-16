from pathlib import Path

from benchmark.agent.llm import LlmResponse, LlmToolCall, MockLlmClient
from benchmark.agent.models import AgentConfig, AgentStepType, AgentStrategyName, LlmConfig, ToolCallStatus
from benchmark.agent.runtime import AgentRuntime
from benchmark.agent.strategies import DirectToolCallingStrategy, create_agent_strategy
from benchmark.agent.tool_context import AgentToolContext
from benchmark.agent.tool_executor import MockToolExecutor


def make_config(max_steps: int = 20) -> AgentConfig:
    return AgentConfig(
        agent_id="agent-1",
        strategy=AgentStrategyName.DIRECT_TOOL_CALLING,
        llm=LlmConfig(provider="mock", model="mock"),
        mcp_profile="minimal",
        max_steps=max_steps,
    )


def test_direct_tool_calling_strategy_works_with_mock_llm() -> None:
    strategy = DirectToolCallingStrategy()
    llm = MockLlmClient(
        [
            LlmResponse(
                tool_calls=[
                    LlmToolCall(name="get_scene_info", arguments={}),
                ]
            )
        ]
    )
    executor = MockToolExecutor(results={"get_scene_info": {"objects": []}})

    trace = strategy.run(
        {"id": "task-1", "prompt": "Inspect the scene"},
        make_config(),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.success is True
    assert [step.step_type for step in trace.steps] == [
        AgentStepType.LLM_CALL,
        AgentStepType.TOOL_CALL,
        AgentStepType.FINAL,
    ]
    assert trace.steps[1].tool_name == "get_scene_info"
    assert trace.steps[1].observation == {"objects": []}


def test_direct_tool_calling_strategy_executes_multiple_tool_calls() -> None:
    llm = MockLlmClient(
        [
            LlmResponse(
                tool_calls=[
                    LlmToolCall(name="get_scene_info", arguments={}),
                    LlmToolCall(name="get_object_info", arguments={"object_name": "Cube"}),
                ]
            )
        ]
    )
    executor = MockToolExecutor(
        results={
            "get_scene_info": {"objects": ["Cube"]},
            "get_object_info": {"name": "Cube"},
        }
    )

    trace = DirectToolCallingStrategy().run(
        {"id": "task-1", "prompt": "Inspect Cube"},
        make_config(),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    tool_steps = [step for step in trace.steps if step.step_type == AgentStepType.TOOL_CALL]
    assert trace.success is True
    assert [step.tool_name for step in tool_steps] == ["get_scene_info", "get_object_info"]
    assert [call.name for call in executor.calls] == ["get_scene_info", "get_object_info"]


def test_direct_tool_calling_strategy_supports_json_action() -> None:
    llm = MockLlmClient(
        [
            LlmResponse(
                content='{"tool_name": "get_scene_info", "arguments": {"include": "objects"}}'
            )
        ]
    )
    executor = MockToolExecutor(results={"get_scene_info": {"objects": []}})

    trace = DirectToolCallingStrategy().run(
        {"id": "task-1", "prompt": "Inspect the scene"},
        make_config(),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.success is True
    assert executor.calls[0].arguments == {"include": "objects"}


def test_direct_tool_calling_strategy_blocks_forbidden_tool() -> None:
    llm = MockLlmClient(
        [
            LlmResponse(
                tool_calls=[
                    LlmToolCall(name="execute_blender_code", arguments={"code": "print(1)"}),
                ]
            )
        ]
    )
    executor = MockToolExecutor(allowed_tools=["get_scene_info"])

    trace = DirectToolCallingStrategy().run(
        {"id": "task-1", "prompt": "Run code"},
        make_config(),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.success is False
    assert trace.error is not None
    assert "not allowed" in trace.error
    assert trace.steps[-1].step_type == AgentStepType.ERROR
    assert trace.steps[1].tool_name == "execute_blender_code"


def test_runtime_selects_direct_tool_calling_strategy() -> None:
    runtime = AgentRuntime(
        make_config(),
        llm_client=MockLlmClient(
            [LlmResponse(tool_calls=[LlmToolCall(name="get_scene_info", arguments={})])]
        ),
        tool_executor=MockToolExecutor(results={"get_scene_info": {"objects": []}}),
    )

    result = runtime.run(task_id="task-1", task={"prompt": "Inspect"}, artifacts_dir=None)

    assert isinstance(create_agent_strategy(AgentStrategyName.DIRECT_TOOL_CALLING), DirectToolCallingStrategy)
    assert result.ok is True
    assert result.status.value == "passed"


def test_direct_tool_calling_strategy_respects_max_steps() -> None:
    llm = MockLlmClient(
        [
            LlmResponse(
                tool_calls=[
                    LlmToolCall(name="get_scene_info", arguments={}),
                    LlmToolCall(name="get_object_info", arguments={"object_name": "Cube"}),
                ]
            )
        ]
    )
    executor = MockToolExecutor(results={"get_scene_info": {}, "get_object_info": {}})

    trace = DirectToolCallingStrategy().run(
        {"id": "task-1", "prompt": "Inspect"},
        make_config(max_steps=3),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.success is False
    assert "max_steps" in (trace.error or "")
    assert len([step for step in trace.steps if step.step_type == AgentStepType.TOOL_CALL]) == 1
