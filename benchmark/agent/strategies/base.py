from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from benchmark.agent.errors import UnsupportedAgentStrategyError
from benchmark.agent.llm.base import LlmClient
from benchmark.agent.models import AgentConfig, AgentStepType, AgentStrategyName, AgentTrace
from benchmark.agent.tool_context import AgentToolContext
from benchmark.agent.tool_executor import ToolExecutor


@runtime_checkable
class AgentStrategy(Protocol):
    def run(
        self,
        task: dict[str, Any],
        agent_config: AgentConfig,
        llm_client: LlmClient | None,
        tool_executor: ToolExecutor,
        tool_context: AgentToolContext,
        output_dir: Path,
    ) -> AgentTrace:
        """Run an agent strategy and return an execution trace."""


class NotImplementedAgentStrategy:
    def __init__(self, strategy_name: AgentStrategyName) -> None:
        self.strategy_name = strategy_name

    def run(
        self,
        task: dict[str, Any],
        agent_config: AgentConfig,
        llm_client: LlmClient | None,
        tool_executor: ToolExecutor,
        tool_context: AgentToolContext,
        output_dir: Path,
    ) -> AgentTrace:
        started_at = datetime.datetime.now(datetime.timezone.utc)
        finished_at = datetime.datetime.now(datetime.timezone.utc)
        task_id = str(task.get("id") or tool_context.task_id or "unknown")
        trace = AgentTrace(
            run_id=tool_context.run_id or str(uuid.uuid4()),
            task_id=task_id,
            agent_id=agent_config.agent_id,
            strategy=agent_config.strategy,
            model=agent_config.llm.model if agent_config.llm else None,
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=(finished_at - started_at).total_seconds(),
            success=False,
            error=f"Agent strategy is not implemented yet: {self.strategy_name.value}",
            metadata={"output_dir": str(output_dir)},
        )
        return trace.add_step(
            AgentStepType.ERROR,
            error=f"Agent strategy is not implemented yet: {self.strategy_name.value}",
        )


def create_agent_strategy(strategy_name: AgentStrategyName | str) -> AgentStrategy:
    try:
        strategy = strategy_name if isinstance(strategy_name, AgentStrategyName) else AgentStrategyName(strategy_name)
    except ValueError as error:
        raise UnsupportedAgentStrategyError(f"Unsupported agent strategy: {strategy_name}") from error

    if strategy == AgentStrategyName.DIRECT_TOOL_CALLING:
        from benchmark.agent.strategies.direct_tool_calling import DirectToolCallingStrategy

        return DirectToolCallingStrategy()

    if strategy == AgentStrategyName.REACT:
        from benchmark.agent.strategies.react import ReactStrategy

        return ReactStrategy()

    if strategy == AgentStrategyName.PLAN_AND_EXECUTE:
        from benchmark.agent.strategies.plan_and_execute import PlanAndExecuteStrategy

        return PlanAndExecuteStrategy()

    if strategy == AgentStrategyName.REMOTE_AGENT:
        from benchmark.agent.strategies.remote_agent import RemoteAgentStrategy

        return RemoteAgentStrategy()

    raise UnsupportedAgentStrategyError(f"Unsupported agent strategy: {strategy.value}")
