from __future__ import annotations

from pathlib import Path

from benchmark.agent.config_loader import load_agent_config
from benchmark.agent.llm import LlmResponse, LlmToolCall, MockLlmClient
from benchmark.agent.models import AgentStrategyName
from benchmark.agent.remote import MockRemoteAgentClient
from benchmark.agent.runtime import AgentRuntime
from benchmark.agent.tool_executor import MockToolExecutor, ToolExecutor
from benchmark.runner.execution import ExecutionBackend, ExecutionResult
from benchmark.runner.models import ExecutionMode, RunConfig
from benchmark.tasks.loader import load_task


class AgentExecutionBackend(ExecutionBackend):
    """Experiment runner backend shell for agent-based runs."""

    mode = ExecutionMode.AGENT_MCP

    def __init__(
        self,
        agent_config_path: Path | str | None = None,
        *,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        self.agent_config_path = Path(agent_config_path) if agent_config_path else None
        self.tool_executor = tool_executor

    def execute(self, config: RunConfig) -> ExecutionResult:
        output_dir = Path(config.agent_output_dir or config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        agent_config_path = config.agent_config_path or self.agent_config_path
        if agent_config_path is None:
            return ExecutionResult(
                ok=False,
                scene_snapshot_path=None,
                artifacts_dir=output_dir,
                error="agent_config_path is required",
            )

        try:
            agent_config = load_agent_config(agent_config_path)
            task = _load_task_payload(config)
        except Exception as error:
            return ExecutionResult(
                ok=False,
                scene_snapshot_path=None,
                artifacts_dir=output_dir,
                error=str(error),
            )

        if config.execution_mode == ExecutionMode.REMOTE_AGENT:
            agent_config = agent_config.model_copy(
                update={"strategy": AgentStrategyName.REMOTE_AGENT}
            )

        result = AgentRuntime(
            agent_config,
            tool_executor=self.tool_executor or _default_tool_executor(agent_config.strategy),
            llm_client=_default_llm_client(agent_config.strategy),
            remote_agent_client=_default_remote_client(agent_config.strategy),
        ).run(
            task_id=config.task_id,
            task=task,
            artifacts_dir=output_dir,
        )
        metadata = {
            **result.metadata,
            "trace_path": str(result.trace_path) if result.trace_path else None,
            "agent_run": result.model_dump(mode="json"),
            "warning": None,
        }
        if result.ok and result.scene_snapshot_path is None:
            metadata["warning"] = "agent did not produce scene_snapshot_path"
        return ExecutionResult(
            ok=result.ok and result.scene_snapshot_path is not None,
            scene_snapshot_path=result.scene_snapshot_path,
            artifacts_dir=output_dir,
            output_files=[result.trace_path] if result.trace_path else [],
            error=result.error or (
                "agent did not produce scene_snapshot_path"
                if result.scene_snapshot_path is None
                else None
            ),
            metadata=metadata,
        )


class RemoteAgentExecutionBackend(AgentExecutionBackend):
    mode = ExecutionMode.REMOTE_AGENT


def _load_task_payload(config: RunConfig) -> dict:
    if config.task_path is None:
        return {"id": config.task_id}
    return load_task(config.task_path).model_dump(mode="json", exclude_none=True)


def _default_tool_executor(strategy: AgentStrategyName) -> ToolExecutor:
    if strategy == AgentStrategyName.REMOTE_AGENT:
        return MockToolExecutor()
    return MockToolExecutor(results={"get_scene_info": {"objects": []}})


def _default_llm_client(strategy: AgentStrategyName) -> MockLlmClient | None:
    if strategy == AgentStrategyName.REMOTE_AGENT:
        return None
    return MockLlmClient(
        [LlmResponse(tool_calls=[LlmToolCall(name="get_scene_info", arguments={})])]
    )


def _default_remote_client(strategy: AgentStrategyName) -> MockRemoteAgentClient | None:
    if strategy != AgentStrategyName.REMOTE_AGENT:
        return None
    return MockRemoteAgentClient()
