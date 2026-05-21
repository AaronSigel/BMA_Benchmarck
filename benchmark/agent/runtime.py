from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from benchmark.agent.artifact_layout import ArtifactLayout, build_artifact_layout
from benchmark.agent.config_loader import load_agent_config as _load_agent_config
from benchmark.agent.errors import AgentError, AgentRuntimeError, UnsupportedAgentStrategyError
from benchmark.agent.llm.base import LlmClient
from benchmark.agent.llm.factory import create_llm_client as _create_llm_client
from benchmark.agent.models import AgentConfig, AgentRunResult, AgentRunStatus, AgentStepType
from benchmark.agent.remote.base import RemoteAgentClient
from benchmark.agent.remote.factory import create_remote_agent_client as _create_remote_agent_client
from benchmark.agent.strategies.base import AgentStrategy, create_agent_strategy
from benchmark.agent.tool_context import AgentToolContext
from benchmark.agent.tool_executor import MockToolExecutor, NoopToolExecutor, ToolExecutor, ToolSchemaProvider
from benchmark.agent.trace import AgentTrace, summarize_trace, write_agent_trace


class AgentRuntime:
    """Lightweight runtime shell for later strategy and backend implementations."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        tool_executor: ToolExecutor | None = None,
        tool_schema_provider: ToolSchemaProvider | None = None,
        llm_client: LlmClient | None = None,
        remote_agent_client: RemoteAgentClient | None = None,
        strategy: AgentStrategy | None = None,
    ) -> None:
        self.config = config
        self.tool_executor = tool_executor or NoopToolExecutor()
        self.tool_schema_provider = tool_schema_provider or ToolSchemaProvider()
        self.llm_client = llm_client
        self.remote_agent_client = remote_agent_client
        self.strategy = strategy

    def run(
        self,
        *,
        task_id: str | None = None,
        task: dict[str, Any] | None = None,
        artifacts_dir: Path | None = None,
    ) -> AgentRunResult:
        started_at = datetime.datetime.now(datetime.timezone.utc)
        run_id = str(uuid.uuid4())
        resolved_task_id = task_id or "unknown"
        task_data = dict(task or {})
        task_data.setdefault("id", resolved_task_id)
        trace_path = None
        error_message = None
        status = AgentRunStatus.ERROR
        trace: AgentTrace | None = None

        try:
            strategy = self.strategy or self.get_strategy()
            llm_client = None if self.config.strategy.value == "remote_agent" else self.llm_client or self._build_llm_client()
            tool_context = self.build_tool_context(
                run_id=run_id,
                task_id=resolved_task_id,
                artifacts_dir=artifacts_dir,
            )
            remote_agent_client = self.remote_agent_client or self._build_remote_agent_client()
            if remote_agent_client is not None:
                tool_context = tool_context.model_copy(
                    update={
                        "metadata": {
                            **tool_context.metadata,
                            "remote_agent_client": remote_agent_client,
                        }
                    }
                )
            _inject_scene_validator(strategy, self.config, self.tool_executor, task_data)
            trace = strategy.run(
                task_data,
                self.config,
                llm_client,
                self.tool_executor,
                tool_context,
                Path(artifacts_dir or "."),
            )
            status = AgentRunStatus.PASSED if trace.success else AgentRunStatus.ERROR
            error_message = trace.error
        except UnsupportedAgentStrategyError:
            raise
        except AgentError as error:
            error_message = str(error)
            trace = self._error_trace(run_id, resolved_task_id, started_at, error_message)

        if not trace.steps:
            trace = trace.add_step(
                AgentStepType.OBSERVATION,
                observation="runtime_initialized",
                metadata={
                    "tool_schema_count": len(self.tool_schema_provider.list_tool_schemas()),
                    "has_task": task is not None,
                },
            )
        finished_at = datetime.datetime.now(datetime.timezone.utc)
        trace = trace.model_copy(
            update={
                "finished_at": trace.finished_at or finished_at,
                "duration_sec": trace.duration_sec
                if trace.duration_sec is not None
                else (finished_at - started_at).total_seconds(),
            }
        )
        if trace.success is None:
            trace = trace.model_copy(
                update={
                    "success": status == AgentRunStatus.PASSED,
                    "error": error_message,
                }
            )
        if trace.error and error_message is None:
            error_message = trace.error

        if trace.error and isinstance(trace.error, str) and trace.structured_error is None:
            from benchmark.runner.controlled_errors import controlled_error_payload
            normalized_error = trace.metadata.get("react_error_type") or trace.error
            structured = controlled_error_payload(
                str(normalized_error),
                enrich=True,
                scene_passed_but_agent_error=bool(trace.metadata.get("scene_passed_but_agent_error")),
                early_stop_reason=trace.metadata.get("early_stop_reason"),
                no_progress_reason=trace.metadata.get("no_progress_reason"),
            )
            trace = trace.model_copy(update={"structured_error": structured})

        run_artifacts_dir = artifacts_dir
        if artifacts_dir is not None:
            layout = build_artifact_layout(artifacts_dir, run_id)
            layout.create_dirs()
            run_artifacts_dir = layout.run_dir
            if self.config.trace_enabled:
                trace_path = layout.agent_trace
                write_agent_trace(trace, trace_path)

        finished_at = trace.finished_at or finished_at
        scene_snapshot_path = _scene_snapshot_path_from_trace(trace)
        return AgentRunResult(
            ok=status == AgentRunStatus.PASSED,
            run_id=trace.run_id,
            task_id=trace.task_id,
            agent_id=self.config.agent_id,
            status=status,
            trace_path=trace_path,
            scene_snapshot_path=scene_snapshot_path,
            artifacts_dir=run_artifacts_dir,
            error=error_message,
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=trace.duration_sec,
            summary={
                **summarize_trace(trace),
                **({"structured_error": trace.structured_error} if trace.structured_error else {}),
            },
            metadata={"strategy": self.config.strategy.value, "model": self.config.llm.model if self.config.llm else None},
        )

    def get_strategy(self) -> AgentStrategy:
        return create_agent_strategy(self.config.strategy)

    def _build_llm_client(self) -> LlmClient | None:
        if self.config.llm is None:
            return None
        return create_llm_client(self.config.llm)

    def _build_remote_agent_client(self) -> RemoteAgentClient | None:
        if self.config.remote_agent is None:
            return None
        return create_remote_agent_client(self.config.remote_agent)

    def _error_trace(
        self,
        run_id: str,
        task_id: str,
        started_at: datetime.datetime,
        error_message: str,
    ) -> AgentTrace:
        finished_at = datetime.datetime.now(datetime.timezone.utc)
        trace = AgentTrace(
            run_id=run_id,
            task_id=task_id,
            agent_id=self.config.agent_id,
            strategy=self.config.strategy,
            model=self.config.llm.model if self.config.llm else None,
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=(finished_at - started_at).total_seconds(),
            success=False,
            error=error_message,
            metadata={"strategy": self.config.strategy.value},
        )
        return trace.add_step(
            AgentStepType.ERROR,
            error=error_message,
        )

    def build_tool_context(
        self,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
        artifacts_dir: Path | None = None,
    ) -> AgentToolContext:
        return AgentToolContext(
            run_id=run_id,
            task_id=task_id,
            artifacts_dir=artifacts_dir,
            allowed_tools=self.config.allowed_tools,
        )


def load_agent_config(path: Path | str) -> AgentConfig:
    return _load_agent_config(path)


def create_llm_client(config: Any) -> LlmClient:
    return _create_llm_client(config)


def create_remote_agent_client(config: Any) -> RemoteAgentClient:
    return _create_remote_agent_client(config)


def run_task(
    task: Any,
    agent_config: AgentConfig,
    tool_executor: ToolExecutor | None,
    output_dir: Path | str,
    *,
    llm_client: LlmClient | None = None,
    remote_agent_client: RemoteAgentClient | None = None,
    strategy: AgentStrategy | None = None,
) -> AgentRunResult:
    task_data = _task_to_dict(task)
    output_path = Path(output_dir)
    executor = tool_executor or MockToolExecutor()
    return AgentRuntime(
        agent_config,
        tool_executor=executor,
        llm_client=llm_client,
        remote_agent_client=remote_agent_client,
        strategy=strategy,
    ).run(
        task_id=str(task_data.get("id") or "unknown"),
        task=task_data,
        artifacts_dir=output_path,
    )


def _task_to_dict(task: Any) -> dict[str, Any]:
    if isinstance(task, dict):
        return dict(task)
    if isinstance(task, BaseModel):
        return task.model_dump(mode="json", exclude_none=True)
    if hasattr(task, "model_dump"):
        return task.model_dump(mode="json", exclude_none=True)
    raise AgentRuntimeError("task must be a dict or Pydantic model")


def _scene_snapshot_path_from_trace(trace: AgentTrace) -> Path | None:
    value = trace.metadata.get("scene_snapshot_path")
    if value is None:
        return None
    return Path(str(value))


def _inject_scene_validator(
    strategy: Any,
    config: AgentConfig,
    tool_executor: Any,
    task_data: dict[str, Any],
) -> None:
    """Inject a scene validation callback into ReactStrategy when validation-based stopping is enabled."""
    if not (config.stop_after_scene_passed or config.detect_no_progress):
        return
    from benchmark.agent.strategies.react import ReactStrategy
    if not isinstance(strategy, ReactStrategy):
        return

    from benchmark.validation.snapshot_normalization import build_scene_validator_fn

    strategy.scene_validator_fn = build_scene_validator_fn(tool_executor, task_data)
