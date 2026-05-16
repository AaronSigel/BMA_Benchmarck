from pathlib import Path

import pytest

from benchmark.agent.errors import LlmResponseParseError
from benchmark.agent.llm import LlmResponse, MockLlmClient
from benchmark.agent.models import AgentConfig, AgentStepType, AgentStrategyName, LlmConfig
from benchmark.agent.runtime import AgentRuntime
from benchmark.agent.strategies import PlanAndExecuteStrategy, create_agent_strategy
from benchmark.agent.tool_context import AgentToolContext
from benchmark.agent.tool_executor import MockToolExecutor


def make_config(max_steps: int = 10) -> AgentConfig:
    return AgentConfig(
        agent_id="agent-1",
        strategy=AgentStrategyName.PLAN_AND_EXECUTE,
        llm=LlmConfig(provider="mock", model="mock"),
        mcp_profile="minimal",
        max_steps=max_steps,
    )


def test_plan_execute_strategy_executes_steps_in_order() -> None:
    llm = MockLlmClient(
        [
            LlmResponse(
                content=(
                    '{"plan":['
                    '{"step":2,"description":"Inspect cube","tool":"get_object_info",'
                    '"arguments":{"object_name":"Cube"}},'
                    '{"step":1,"description":"Inspect scene","tool":"get_scene_info",'
                    '"arguments":{}}'
                    "]} "
                )
            )
        ]
    )
    executor = MockToolExecutor(
        results={
            "get_scene_info": {"objects": ["Cube"]},
            "get_object_info": {"name": "Cube"},
        }
    )

    trace = PlanAndExecuteStrategy().run(
        {"id": "task-1", "prompt": "Inspect Cube"},
        make_config(),
        llm,
        executor,
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.success is True
    assert trace.steps[0].step_type == AgentStepType.PLAN
    assert [call.name for call in executor.calls] == ["get_scene_info", "get_object_info"]
    assert [step.metadata.get("plan_step") for step in trace.steps if step.step_type == AgentStepType.TOOL_CALL] == [1, 2]


def test_plan_execute_strategy_invalid_plan_raises_parse_error() -> None:
    llm = MockLlmClient([LlmResponse(content='{"plan":[{"step":1,"tool":"get_scene_info"}]}')])

    with pytest.raises(LlmResponseParseError, match="description"):
        PlanAndExecuteStrategy().run(
            {"id": "task-1", "prompt": "Inspect"},
            make_config(),
            llm,
            MockToolExecutor(),
            AgentToolContext(run_id="run-1", task_id="task-1"),
            Path("."),
        )


def test_plan_execute_strategy_blocks_forbidden_tool() -> None:
    llm = MockLlmClient(
        [
            LlmResponse(
                content=(
                    '{"plan":[{"step":1,"description":"Run code",'
                    '"tool":"execute_blender_code","arguments":{"code":"print(1)"}}]}'
                )
            )
        ]
    )
    executor = MockToolExecutor(allowed_tools=["get_scene_info"])

    trace = PlanAndExecuteStrategy().run(
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
    assert trace.steps[0].step_type == AgentStepType.PLAN
    assert trace.steps[1].tool_name == "execute_blender_code"


def test_plan_execute_strategy_respects_max_steps() -> None:
    llm = MockLlmClient(
        [
            LlmResponse(
                content=(
                    '{"plan":['
                    '{"step":1,"description":"A","tool":"get_scene_info","arguments":{}},'
                    '{"step":2,"description":"B","tool":"get_scene_info","arguments":{}}'
                    "]} "
                )
            )
        ]
    )

    trace = PlanAndExecuteStrategy().run(
        {"id": "task-1", "prompt": "Inspect"},
        make_config(max_steps=3),
        llm,
        MockToolExecutor(results={"get_scene_info": {}}),
        AgentToolContext(run_id="run-1", task_id="task-1"),
        Path("."),
    )

    assert trace.success is False
    assert "max_steps" in (trace.error or "")
    assert len([step for step in trace.steps if step.step_type == AgentStepType.TOOL_CALL]) == 1


def test_runtime_selects_plan_execute_strategy() -> None:
    runtime = AgentRuntime(
        make_config(),
        llm_client=MockLlmClient(
            [
                LlmResponse(
                    content='{"plan":[{"step":1,"description":"Inspect","tool":"get_scene_info","arguments":{}}]}'
                )
            ]
        ),
        tool_executor=MockToolExecutor(results={"get_scene_info": {}}),
    )

    result = runtime.run(task_id="task-1", task={"prompt": "Inspect"})

    assert isinstance(create_agent_strategy(AgentStrategyName.PLAN_AND_EXECUTE), PlanAndExecuteStrategy)
    assert result.ok is True
    assert result.status.value == "passed"
