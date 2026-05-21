from __future__ import annotations

from collections.abc import Callable

from benchmark.agent.errors import UnsupportedAgentStrategyError
from benchmark.agent.models import AgentStrategyName
from benchmark.agent.strategies.base import AgentStrategy

StrategyFactory = Callable[[], AgentStrategy]


class StrategyRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, tuple[str, StrategyFactory]] = {}

    def register(self, name: str | AgentStrategyName, import_path: str, factory: StrategyFactory) -> None:
        key = name.value if isinstance(name, AgentStrategyName) else str(name)
        self._factories[key] = (import_path, factory)

    def create(self, name: str | AgentStrategyName) -> AgentStrategy:
        key = name.value if isinstance(name, AgentStrategyName) else str(name)
        try:
            return self._factories[key][1]()
        except KeyError as error:
            raise UnsupportedAgentStrategyError(f"Unsupported agent strategy: {key}. Available: {', '.join(self.names())}") from error

    def names(self) -> list[str]:
        return sorted(self._factories)

    def entries(self) -> list[dict[str, str]]:
        return [{"name": name, "class": item[0]} for name, item in sorted(self._factories.items())]


STRATEGY_REGISTRY = StrategyRegistry()


def _register_defaults() -> None:
    STRATEGY_REGISTRY.register(
        AgentStrategyName.DIRECT_TOOL_CALLING,
        "benchmark.agent.strategies.direct_tool_calling.DirectToolCallingStrategy",
        lambda: __import__("benchmark.agent.strategies.direct_tool_calling", fromlist=["DirectToolCallingStrategy"]).DirectToolCallingStrategy(),
    )
    STRATEGY_REGISTRY.register(
        AgentStrategyName.REACT,
        "benchmark.agent.strategies.react.ReactStrategy",
        lambda: __import__("benchmark.agent.strategies.react", fromlist=["ReactStrategy"]).ReactStrategy(),
    )
    STRATEGY_REGISTRY.register(
        AgentStrategyName.PLAN_AND_EXECUTE,
        "benchmark.agent.strategies.plan_and_execute.PlanAndExecuteStrategy",
        lambda: __import__("benchmark.agent.strategies.plan_and_execute", fromlist=["PlanAndExecuteStrategy"]).PlanAndExecuteStrategy(),
    )
    STRATEGY_REGISTRY.register(
        AgentStrategyName.REMOTE_AGENT,
        "benchmark.agent.strategies.remote_agent.RemoteAgentStrategy",
        lambda: __import__("benchmark.agent.strategies.remote_agent", fromlist=["RemoteAgentStrategy"]).RemoteAgentStrategy(),
    )
    STRATEGY_REGISTRY.register(
        AgentStrategyName.PLAN_EXECUTE_REACT_REPAIR,
        "benchmark.agent.strategies.plan_execute_react_repair.PlanExecuteReactRepairStrategy",
        lambda: __import__("benchmark.agent.strategies.plan_execute_react_repair", fromlist=["PlanExecuteReactRepairStrategy"]).PlanExecuteReactRepairStrategy(),
    )


_register_defaults()
