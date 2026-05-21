from __future__ import annotations

import logging
import shutil
from pathlib import Path

import yaml
from pydantic import ValidationError

log = logging.getLogger(__name__)

from benchmark.agent.config_loader import load_agent_config
from benchmark.agent.llm import LlmResponse, LlmToolCall, MockLlmClient
from benchmark.agent.models import AgentConfig, AgentStrategyName, LlmProvider
from benchmark.agent.remote import MockRemoteAgentClient
from benchmark.agent.runtime import AgentRuntime
from benchmark.agent.tool_executor import McpToolExecutor, MockToolExecutor, ToolExecutor
from benchmark.blender.models import SceneSnapshot
from benchmark.mcp.config import McpServerConfig
from benchmark.mcp.server_adapter import ExternalBlenderMcpServerAdapter
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
        output_dir = Path(config.output_dir)
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
            agent_config = _apply_run_overrides(agent_config, config)
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

        tool_executor = self.tool_executor or _build_tool_executor(agent_config.strategy, config)
        lifecycle_error, pre_run_snapshot_path = _prepare_blender_scene(tool_executor, output_dir, config.task_id)
        if lifecycle_error is not None:
            return ExecutionResult(
                ok=False,
                scene_snapshot_path=pre_run_snapshot_path,
                artifacts_dir=output_dir,
                error=lifecycle_error,
                metadata={
                    "strategy": agent_config.strategy.value,
                    "agent_id": agent_config.agent_id,
                    "mcp_profile": agent_config.mcp_profile,
                    "pre_run_snapshot_path": str(pre_run_snapshot_path) if pre_run_snapshot_path else None,
                },
            )
        tool_executor = _wrap_export_paths(tool_executor, output_dir, task)
        log.info("[task:%s] strategy=%s profile=%s executor=%s", config.task_id, agent_config.strategy, agent_config.mcp_profile, type(tool_executor).__name__)
        try:
            result = AgentRuntime(
                agent_config,
                tool_executor=tool_executor,
                llm_client=_default_llm_client(agent_config),
                remote_agent_client=_default_remote_client(agent_config.strategy),
            ).run(
                task_id=config.task_id,
                task=task,
                artifacts_dir=output_dir,
            )
        except Exception as error:
            scene_snapshot_path = _auto_capture_snapshot(tool_executor, output_dir)
            return ExecutionResult(
                ok=False,
                scene_snapshot_path=scene_snapshot_path,
                artifacts_dir=output_dir,
                error=str(error),
                metadata={
                    "strategy": agent_config.strategy.value,
                    "agent_id": agent_config.agent_id,
                    "mcp_profile": agent_config.mcp_profile,
                    "pre_run_snapshot_path": str(pre_run_snapshot_path) if pre_run_snapshot_path else None,
                },
            )

        # Always try to capture a scene snapshot — even on error — so partial
        # validation can run and the scene state is preserved for debugging.
        scene_snapshot_path = result.scene_snapshot_path
        captured_post_run_snapshot = False
        if scene_snapshot_path is None:
            status_label = "success" if result.ok else "error"
            log.info("[task:%s] auto-capturing scene snapshot after %s", config.task_id, status_label)
            scene_snapshot_path = _auto_capture_snapshot(tool_executor, output_dir)
            if scene_snapshot_path:
                captured_post_run_snapshot = True
                log.info("[task:%s] scene snapshot captured: %s", config.task_id, scene_snapshot_path)
            else:
                log.warning("[task:%s] auto-capture failed: no snapshot produced", config.task_id)

        trace_path = _copy_trace_to_run_root(result.trace_path, output_dir)
        agent_run_result = result
        if trace_path is not None and trace_path != result.trace_path:
            agent_run_result = result.model_copy(update={"trace_path": trace_path})
        if scene_snapshot_path is not None and result.scene_snapshot_path != scene_snapshot_path:
            summary = dict(result.summary)
            execution_summary = summary.get("execution")
            if isinstance(execution_summary, dict):
                summary["execution"] = {
                    **execution_summary,
                    "scene_snapshot_path": str(scene_snapshot_path),
                }
            agent_run_result = result.model_copy(
                update={
                    "scene_snapshot_path": scene_snapshot_path,
                    "summary": summary,
                }
            )

        metadata = {
            **result.metadata,
            "trace_path": str(trace_path) if trace_path else None,
            "nested_trace_path": str(result.trace_path) if result.trace_path else None,
            "agent_run": agent_run_result.model_dump(mode="json"),
            "warning": None,
            "mcp_profile": agent_config.mcp_profile,
            "agent_id": agent_config.agent_id,
            "strategy": agent_config.strategy.value,
            "pre_run_snapshot_path": str(pre_run_snapshot_path) if pre_run_snapshot_path else None,
            "post_run_snapshot_captured": captured_post_run_snapshot,
        }
        if result.ok and scene_snapshot_path is None:
            metadata["warning"] = "agent did not produce scene_snapshot_path"
        return ExecutionResult(
            ok=result.ok and scene_snapshot_path is not None,
            scene_snapshot_path=scene_snapshot_path,
            artifacts_dir=output_dir,
            output_files=[trace_path] if trace_path else [],
            error=result.error or (
                "agent did not produce scene_snapshot_path"
                if scene_snapshot_path is None
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


def _copy_trace_to_run_root(trace_path: Path | None, output_dir: Path) -> Path | None:
    if trace_path is None or not trace_path.exists():
        return trace_path
    destination = output_dir / "agent_trace.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.resolve() != destination.resolve():
        shutil.copy2(trace_path, destination)
    return destination


def _apply_run_overrides(agent_config: AgentConfig, config: RunConfig) -> AgentConfig:
    updates = {}
    if config.mcp_profile:
        updates["mcp_profile"] = config.mcp_profile
    model_id = config.metadata.get("model_id")
    if (
        isinstance(model_id, str)
        and model_id
        and model_id != "default"
        and agent_config.llm is not None
    ):
        updates["llm"] = agent_config.llm.model_copy(update={"model": model_id})
    strategy_limits = config.metadata.get("strategy_limits")
    if isinstance(strategy_limits, dict):
        limits = strategy_limits.get(agent_config.strategy.value)
        if isinstance(limits, dict):
            if isinstance(limits.get("max_steps"), int):
                updates["max_steps"] = limits["max_steps"]
            for key in (
                "stop_after_scene_passed",
                "detect_repeated_actions",
                "detect_duplicate_objects",
                "detect_no_progress",
                "no_progress_limit",
                "repeated_action_mode",
            ):
                if key in limits:
                    updates[key] = limits[key]
    if not updates:
        return agent_config
    return agent_config.model_copy(update=updates)


def _auto_capture_snapshot(tool_executor: ToolExecutor, output_dir: Path) -> Path | None:
    """Capture a full SceneSnapshot as harness infrastructure."""
    snapshot_path = output_dir / "scene_snapshot.json"
    return _capture_snapshot_to_path(tool_executor, snapshot_path)


def _prepare_blender_scene(
    tool_executor: ToolExecutor,
    output_dir: Path,
    task_id: str,
) -> tuple[str | None, Path | None]:
    """Reset Blender and assert the pre-run scene is empty for MCP-backed runs."""
    mcp_executor = _mcp_executor(tool_executor)
    if mcp_executor is None:
        return None, None

    log.info("[task:%s] resetting Blender scene before run", task_id)
    reset_result = mcp_executor.adapter.reset_scene()
    if "warning" in reset_result:
        warning = str(reset_result["warning"])
        log.warning("[task:%s] scene reset failed: %s", task_id, warning)
        return f"scene reset failed: {warning}", None

    pre_run_snapshot_path = output_dir / "pre_run_scene_snapshot.json"
    captured_path = _capture_snapshot_to_path(mcp_executor, pre_run_snapshot_path)
    if captured_path is None:
        return "pre-run scene snapshot could not be collected", None

    try:
        snapshot = SceneSnapshot.model_validate_json(captured_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        return f"pre-run scene snapshot is invalid: {error}", captured_path

    object_count = len(snapshot.objects)
    if object_count != 0:
        return f"pre-run scene is not clean after reset: object_count={object_count}", captured_path

    log.info("[task:%s] pre-run scene is clean", task_id)
    return None, captured_path


def _capture_snapshot_to_path(tool_executor: ToolExecutor, snapshot_path: Path) -> Path | None:
    mcp_executor = _mcp_executor(tool_executor)
    if mcp_executor is None:
        return None
    try:
        result = mcp_executor.adapter.collect_scene_snapshot(snapshot_path)
    except Exception as error:
        log.warning("scene snapshot collection failed: %s", error)
        return None
    if isinstance(result, dict) and "warning" in result:
        log.warning("scene snapshot collection warning: %s", result["warning"])
        return None
    if snapshot_path.exists():
        return snapshot_path
    return None


class _ExportPathFixingExecutor:
    """Wraps a ToolExecutor and rewrites relative bma_export_scene filepaths."""

    def __init__(self, wrapped: ToolExecutor, artifacts_dir: Path, task: dict | None = None) -> None:
        self._wrapped = wrapped
        self._artifacts_dir = artifacts_dir
        self._expected_exports = _expected_export_paths(artifacts_dir, task)

    def call_tool(
        self,
        tool_name,
        arguments=None,
    ):
        if tool_name == "bma_export_scene" and isinstance(arguments, dict):
            fp = arguments.get("filepath", "")
            export_path = self._resolve_export_path(arguments, fp)
            if export_path is not None:
                arguments = {**arguments, "filepath": str(export_path)}
                log.info("[export_fix] rewritten filepath → %s", arguments["filepath"])
            export_path = Path(arguments.get("filepath", "")) if arguments.get("filepath") else None
            if export_path is not None:
                export_path.parent.mkdir(parents=True, exist_ok=True)
                if export_path.exists():
                    export_path.unlink()
        return self._wrapped.call_tool(tool_name, arguments)

    def assert_tool_allowed(self, tool_name: str) -> None:
        return self._wrapped.assert_tool_allowed(tool_name)

    def normalize_tool_result(self, result) -> dict:
        return self._wrapped.normalize_tool_result(result)

    def _resolve_export_path(self, arguments: dict, filepath: object) -> Path | None:
        export_format = str(arguments.get("format") or "").lower()
        filename = arguments.get("filename")
        if filename:
            return _run_export_path(self._artifacts_dir, filename)
        if filepath:
            return _run_export_path(self._artifacts_dir, filepath)
        if export_format and export_format in self._expected_exports:
            return self._expected_exports[export_format]
        return None


def _wrap_export_paths(tool_executor: ToolExecutor, artifacts_dir: Path, task: dict | None = None) -> ToolExecutor:
    if not isinstance(tool_executor, McpToolExecutor):
        return tool_executor
    return _ExportPathFixingExecutor(tool_executor, artifacts_dir, task)  # type: ignore[return-value]


def _run_export_path(artifacts_dir: Path, requested: object) -> Path | None:
    requested_path = Path(str(requested))
    if requested_path.is_absolute():
        return requested_path
    requested_text = str(requested_path).replace("\\", "/")
    if "/" in requested_text:
        return artifacts_dir / requested_path
    suffix = requested_path.suffix.lower()
    if suffix == ".glb":
        return artifacts_dir / "exports" / requested_path.name
    if suffix == ".blend":
        return artifacts_dir / requested_path.name
    return artifacts_dir / "exports" / requested_path.name


def _expected_export_paths(artifacts_dir: Path, task: dict | None) -> dict[str, Path]:
    expected: dict[str, Path] = {}
    scene = task.get("expected_scene") if isinstance(task, dict) else None
    exports = scene.get("exports") if isinstance(scene, dict) else None
    if not isinstance(exports, list):
        return expected
    for export in exports:
        if not isinstance(export, dict):
            continue
        export_format = str(export.get("format") or "").lower()
        if not export_format:
            continue
        filename = export.get("filename")
        if filename:
            expected[export_format] = _run_export_path(artifacts_dir, filename) or artifacts_dir / str(filename)
        elif export_format == "blend":
            expected[export_format] = artifacts_dir / "result.blend"
        elif export_format in {"glb", "gltf", "fbx"}:
            expected[export_format] = artifacts_dir / "exports" / f"result.{export_format}"
    return expected


def _mcp_executor(tool_executor: ToolExecutor) -> McpToolExecutor | None:
    if isinstance(tool_executor, McpToolExecutor):
        return tool_executor
    wrapped = getattr(tool_executor, "_wrapped", None)
    if isinstance(wrapped, McpToolExecutor):
        return wrapped
    return None


def _build_tool_executor(strategy: AgentStrategyName, config: RunConfig) -> ToolExecutor:
    if strategy == AgentStrategyName.REMOTE_AGENT:
        return MockToolExecutor()
    if config.mcp_config_path is not None and config.mcp_config_path.exists():
        raw = yaml.safe_load(config.mcp_config_path.read_text(encoding="utf-8")) or {}
        mcp_config = McpServerConfig(**{k: v for k, v in raw.items() if k != "env"})
        adapter = ExternalBlenderMcpServerAdapter(mcp_config)
        return McpToolExecutor(adapter, profile=config.mcp_profile or mcp_config.profile)
    return MockToolExecutor(results={"get_scene_info": {"objects": []}})


def _default_llm_client(agent_config: AgentConfig) -> MockLlmClient | None:
    if agent_config.strategy == AgentStrategyName.REMOTE_AGENT:
        return None
    # If a real LLM provider is configured, return None so AgentRuntime builds
    # the client from config via the factory.
    if agent_config.llm is not None and agent_config.llm.provider != LlmProvider.MOCK:
        return None
    return MockLlmClient(
        [LlmResponse(tool_calls=[LlmToolCall(name="get_scene_info", arguments={})])]
    )


def _default_remote_client(strategy: AgentStrategyName) -> MockRemoteAgentClient | None:
    if strategy != AgentStrategyName.REMOTE_AGENT:
        return None
    return MockRemoteAgentClient()
