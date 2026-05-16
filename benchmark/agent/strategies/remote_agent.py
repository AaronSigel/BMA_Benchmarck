from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Any

from benchmark.agent.errors import AgentRuntimeError, RemoteAgentError
from benchmark.agent.llm.base import LlmClient
from benchmark.agent.models import AgentConfig, AgentStepType, AgentTrace
from benchmark.agent.remote.base import RemoteAgentClient, RemoteAgentRequest, RemoteAgentResponse
from benchmark.agent.remote.factory import create_remote_agent_client
from benchmark.agent.tool_context import AgentToolContext, ToolSchemaProvider
from benchmark.agent.tool_executor import ToolExecutor


class RemoteAgentStrategy:
    """Delegate task execution to a server-side remote agent."""

    def __init__(self, *, tool_schema_provider: ToolSchemaProvider | None = None) -> None:
        self.tool_schema_provider = tool_schema_provider or ToolSchemaProvider()

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
        task_id = str(task.get("id") or tool_context.task_id or "unknown")
        remote_client = _get_remote_client(agent_config, tool_context)
        tool_contracts = [
            contract.model_dump(mode="json")
            for contract in self.tool_schema_provider.get_tools_for_profile(agent_config.mcp_profile)
        ]
        request = RemoteAgentRequest(
            task=task,
            mcp_config_path=_metadata_path(agent_config.metadata.get("mcp_config_path")),
            mcp_profile=agent_config.mcp_profile,
            tool_contracts=tool_contracts,
            output_dir=output_dir,
            metadata={
                "run_id": tool_context.run_id,
                "task_id": task_id,
                "agent_id": agent_config.agent_id,
            },
        )

        try:
            response = remote_client.run_task(request)
        except RemoteAgentError as error:
            return _build_trace(
                agent_config,
                tool_context,
                task_id,
                started_at,
                success=False,
                error=str(error),
                response=None,
            )

        if response.trace is not None:
            return _merge_response_trace(response.trace, agent_config, tool_context, task_id, started_at, response)

        return _build_trace(
            agent_config,
            tool_context,
            task_id,
            started_at,
            success=response.ok,
            error=response.error,
            response=response,
        )


def _get_remote_client(
    agent_config: AgentConfig,
    tool_context: AgentToolContext,
) -> RemoteAgentClient:
    injected = tool_context.metadata.get("remote_agent_client")
    if injected is not None:
        return injected
    if agent_config.remote_agent is None:
        raise AgentRuntimeError("remote_agent config is required for remote_agent strategy")
    return create_remote_agent_client(agent_config.remote_agent)


def _merge_response_trace(
    trace: AgentTrace,
    agent_config: AgentConfig,
    tool_context: AgentToolContext,
    task_id: str,
    started_at: datetime.datetime,
    response: RemoteAgentResponse,
) -> AgentTrace:
    finished_at = trace.finished_at or datetime.datetime.now(datetime.timezone.utc)
    merged = trace.model_copy(
        update={
            "run_id": tool_context.run_id or trace.run_id,
            "task_id": task_id,
            "agent_id": agent_config.agent_id,
            "strategy": agent_config.strategy,
            "started_at": trace.started_at or started_at,
            "finished_at": finished_at,
            "duration_sec": trace.duration_sec
            if trace.duration_sec is not None
            else (finished_at - (trace.started_at or started_at)).total_seconds(),
            "success": response.ok if trace.success is None else trace.success,
            "error": response.error if trace.error is None else trace.error,
            "metadata": {
                **trace.metadata,
                "scene_snapshot_path": str(response.scene_snapshot_path)
                if response.scene_snapshot_path
                else None,
                "artifacts": [artifact.model_dump(mode="json") for artifact in response.artifacts],
            },
        }
    )
    if not merged.steps:
        merged = merged.add_step(
            AgentStepType.FINAL if response.ok else AgentStepType.ERROR,
            observation=response.raw_response,
            error=response.error,
        )
    return merged


def _build_trace(
    agent_config: AgentConfig,
    tool_context: AgentToolContext,
    task_id: str,
    started_at: datetime.datetime,
    *,
    success: bool,
    error: str | None,
    response: RemoteAgentResponse | None,
) -> AgentTrace:
    finished_at = datetime.datetime.now(datetime.timezone.utc)
    trace = AgentTrace(
        run_id=tool_context.run_id or str(uuid.uuid4()),
        task_id=task_id,
        agent_id=agent_config.agent_id,
        strategy=agent_config.strategy,
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=(finished_at - started_at).total_seconds(),
        success=success,
        error=error,
        final_message="Remote agent completed." if success else None,
        metadata={
            "scene_snapshot_path": str(response.scene_snapshot_path)
            if response and response.scene_snapshot_path
            else None,
            "artifacts": [artifact.model_dump(mode="json") for artifact in response.artifacts]
            if response
            else [],
        },
    )
    return trace.add_step(
        AgentStepType.FINAL if success else AgentStepType.ERROR,
        observation=response.raw_response if response else None,
        error=error,
    )


def _metadata_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(str(value))
