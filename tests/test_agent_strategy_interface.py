from pathlib import Path
from typing import Any

import pytest

from benchmark.agent.errors import UnsupportedAgentStrategyError
from benchmark.agent.llm import LlmMessage, LlmResponse, MockLlmClient
from benchmark.agent.models import AgentConfig, AgentStrategyName, AgentTrace, LlmConfig
from benchmark.agent.runtime import AgentRuntime
from benchmark.agent.strategies import AgentStrategy, create_agent_strategy
from benchmark.agent.tool_context import AgentToolContext
from benchmark.agent.tool_executor import MockToolExecutor, ToolExecutor


class PassingStrategy:
    def __init__(self) -> None:
        self.called = False

    def run(
        self,
        task: dict[str, Any],
        agent_config: AgentConfig,
        llm_client: MockLlmClient | None,
        tool_executor: ToolExecutor,
        tool_context: AgentToolContext,
        output_dir: Path,
    ) -> AgentTrace:
        self.called = True
        return AgentTrace(
            run_id=tool_context.run_id or "run-1",
            task_id=task["id"],
            agent_id=agent_config.agent_id,
            strategy=agent_config.strategy,
            model=agent_config.llm.model if agent_config.llm else None,
            success=True,
            final_message="done",
        )


def test_agent_strategy_protocol_accepts_common_interface() -> None:
    assert isinstance(PassingStrategy(), AgentStrategy)


def test_create_agent_strategy_selects_known_strategy() -> None:
    assert isinstance(create_agent_strategy(AgentStrategyName.DIRECT_TOOL_CALLING), AgentStrategy)
    assert isinstance(create_agent_strategy(AgentStrategyName.REACT), AgentStrategy)
    assert isinstance(create_agent_strategy(AgentStrategyName.PLAN_AND_EXECUTE), AgentStrategy)
    assert isinstance(create_agent_strategy(AgentStrategyName.REMOTE_AGENT), AgentStrategy)


def test_create_agent_strategy_rejects_unknown_strategy() -> None:
    with pytest.raises(UnsupportedAgentStrategyError, match="Unsupported agent strategy"):
        create_agent_strategy("unknown")


def test_runtime_uses_selected_strategy() -> None:
    strategy = PassingStrategy()
    runtime = AgentRuntime(
        AgentConfig(agent_id="agent-1", strategy=AgentStrategyName.REACT, llm=LlmConfig()),
        strategy=strategy,
        llm_client=MockLlmClient([LlmResponse(content="ok")]),
        tool_executor=MockToolExecutor(),
    )

    result = runtime.run(task_id="task-1", task={"prompt": "Create a cube"})

    assert strategy.called is True
    assert result.ok is True
    assert result.status.value == "passed"
    assert result.task_id == "task-1"


def test_runtime_rejects_unknown_strategy_name() -> None:
    config = AgentConfig(agent_id="agent-1", strategy=AgentStrategyName.REACT, llm=LlmConfig())
    config = config.model_copy(update={"strategy": "unknown"})

    with pytest.raises(UnsupportedAgentStrategyError, match="Unsupported agent strategy"):
        AgentRuntime(config).run(task_id="task-1")

