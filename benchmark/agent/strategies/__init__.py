"""Agent strategy interfaces and factories."""

from benchmark.agent.strategies.base import (
    AgentStrategy,
    NotImplementedAgentStrategy,
    create_agent_strategy,
)
from benchmark.agent.strategies.direct_tool_calling import DirectToolCallingStrategy
from benchmark.agent.strategies.plan_and_execute import PlanAndExecuteStrategy
from benchmark.agent.strategies.react import ReactStrategy
from benchmark.agent.strategies.remote_agent import RemoteAgentStrategy

__all__ = [
    "AgentStrategy",
    "DirectToolCallingStrategy",
    "NotImplementedAgentStrategy",
    "PlanAndExecuteStrategy",
    "ReactStrategy",
    "RemoteAgentStrategy",
    "create_agent_strategy",
]
