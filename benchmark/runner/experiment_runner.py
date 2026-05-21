import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

from pydantic import ValidationError

from benchmark.blender.models import SceneSnapshot
from benchmark.metrics.validation_metrics import metrics_from_validation_result
from benchmark.runner.execution import (
    BlenderSmokeBackend,
    ExecutionBackend,
    ExternalSnapshotBackend,
    ReplayBackend,
)
from benchmark.runner.models import AgentStatus, ExecutionMode, RunConfig, RunResult, RunStatus, SceneStatus
from benchmark.runner.paths import RunArtifactLayout
from benchmark.tasks.loader import load_task
from benchmark.tasks.registry import TaskRegistry
from benchmark.validation.models import SceneValidationResult, ValidationStatus
from benchmark.validation.scene_validator import SceneValidator
from benchmark.agent.execution_backend import AgentExecutionBackend, RemoteAgentExecutionBackend
from benchmark.agent.models import AgentStrategyName, AgentTrace
from benchmark.agent.trace import write_agent_trace
from benchmark.runner.artifact_manifest import write_run_artifact_manifest
from benchmark.runner.controlled_errors import controlled_error_payload


class ExperimentRunner:
    def __init__(
        self,
        task_registry: TaskRegistry | None = None,
        backends: dict[ExecutionMode, ExecutionBackend] | None = None,
    ) -> None:
        self.task_registry = task_registry
        self.backends = backends or {
            ExecutionMode.EXTERNAL_SNAPSHOT: ExternalSnapshotBackend(),
            ExecutionMode.REPLAY: ReplayBackend(),
            ExecutionMode.BLENDER_SMOKE: BlenderSmokeBackend(),
            ExecutionMode.AGENT_MCP: AgentExecutionBackend(),
            ExecutionMode.REMOTE_AGENT: RemoteAgentExecutionBackend(),
        }

    def run(self, config: RunConfig) -> RunResult:
        started_at = _now_utc()
        started_perf = time.perf_counter()
        layout = RunArtifactLayout.from_run_output_dir(config.output_dir, config.run_id)
        layout.ensure()

        try:
            task = self._load_task(config)
            backend = self._backend_for(config.execution_mode)
        except Exception as error:
            return self._finish_error(config, layout, started_at, started_perf, str(error))

        log.info("[run:%s] task=%s mode=%s", config.run_id[:8], config.task_id, config.execution_mode.value)
        execution_config = config.model_copy(update={"output_dir": layout.run_dir()})
        execution_result = backend.execute(execution_config)
        if not execution_result.ok:
            log.warning("[run:%s] execution failed: %s", config.run_id[:8], execution_result.error)
            return self._finish_error(
                config,
                layout,
                started_at,
                started_perf,
                execution_result.error or "execution failed",
                scene_snapshot_path=execution_result.scene_snapshot_path,
                artifacts_dir=execution_result.artifacts_dir,
                summary={"execution": execution_result.metadata},
            )

        if execution_result.scene_snapshot_path is None:
            log.warning("[run:%s] no scene_snapshot_path", config.run_id[:8])
            return self._finish_error(
                config,
                layout,
                started_at,
                started_perf,
                "execution did not return scene_snapshot_path",
                artifacts_dir=execution_result.artifacts_dir,
                summary={"execution": execution_result.metadata},
            )

        try:
            snapshot_path = _copy_snapshot_to_layout(
                execution_result.scene_snapshot_path,
                layout.scene_snapshot_json(),
            )
            snapshot = _load_snapshot(snapshot_path)
            validation_result = SceneValidator().validate(
                task,
                snapshot,
                artifacts_dir=execution_result.artifacts_dir,
            )
            validation_result_path = layout.validation_result_json()
            _write_validation_result(validation_result, validation_result_path)
            metrics = metrics_from_validation_result(config.run_id, config.task_id, validation_result)
            _write_metrics(metrics, layout.metrics_json())
        except Exception as error:
            return self._finish_error(
                config,
                layout,
                started_at,
                started_perf,
                str(error),
                scene_snapshot_path=execution_result.scene_snapshot_path,
                artifacts_dir=execution_result.artifacts_dir,
                summary={"execution": execution_result.metadata},
            )

        agent_status = _agent_status_from_execution(execution_result.metadata, None, config.execution_mode)
        scene_status = _scene_status_from_validation(validation_result.overall_status)
        status = _combined_run_status(agent_status, scene_status)
        result = RunResult(
            run_id=config.run_id,
            task_id=config.task_id,
            status=status,
            run_status=status,
            agent_status=agent_status,
            scene_status=scene_status,
            execution_mode=config.execution_mode,
            validation_result_path=validation_result_path,
            scene_snapshot_path=snapshot_path,
            artifacts_dir=layout.run_dir(),
            total_score=validation_result.total_score,
            overall_status=validation_result.overall_status.value,
            started_at=started_at,
            finished_at=_now_utc(),
            duration_sec=time.perf_counter() - started_perf,
            error=None,
            summary={
                "validation": validation_result.summary,
                "execution": execution_result.metadata,
                **_run_metadata_summary(config, execution_result.metadata),
            },
        )
        _write_run_result(result, layout.run_result_json())
        write_run_artifact_manifest(result, layout)
        return result

    def _load_task(self, config: RunConfig):
        if config.task_path is not None:
            return load_task(config.task_path)
        if self.task_registry is None:
            raise ValueError("task_path is required when task_registry is not configured")
        return self.task_registry.get(config.task_id)

    def _backend_for(self, mode: ExecutionMode) -> ExecutionBackend:
        try:
            return self.backends[mode]
        except KeyError as error:
            raise ValueError(f"Execution backend is not configured for mode: {mode.value}") from error

    def _finish_error(
        self,
        config: RunConfig,
        layout: RunArtifactLayout,
        started_at: str,
        started_perf: float,
        error: str,
        scene_snapshot_path: Path | None = None,
        artifacts_dir: Path | None = None,
        summary: dict | None = None,
    ) -> RunResult:
        structured_error = controlled_error_payload(error)
        validation_result_path: Path | None = None
        total_score: float | None = None
        overall_status: str | None = None
        scene_status = SceneStatus.NOT_AVAILABLE
        if scene_snapshot_path is not None and scene_snapshot_path.exists():
            try:
                scene_snapshot_path = _copy_snapshot_to_layout(
                    scene_snapshot_path,
                    layout.scene_snapshot_json(),
                )
            except OSError:
                pass
            # Run partial validation so error runs still have a scene snapshot analysis.
            try:
                task = self._load_task(config)
                snapshot = _load_snapshot(scene_snapshot_path)
                validation_result = SceneValidator().validate(
                    task,
                    snapshot,
                    artifacts_dir=artifacts_dir or layout.run_dir(),
                )
                validation_result_path = layout.validation_result_json()
                _write_validation_result(validation_result, validation_result_path)
                metrics = metrics_from_validation_result(config.run_id, config.task_id, validation_result)
                _write_metrics(metrics, layout.metrics_json())
                total_score = validation_result.total_score
                overall_status = validation_result.overall_status.value
                scene_status = _scene_status_from_validation(validation_result.overall_status)
                log.info("[run:%s] partial validation written (run still error)", config.run_id[:8])
            except Exception as val_error:
                log.warning("[run:%s] partial validation failed: %s", config.run_id[:8], val_error)
        agent_status = _agent_status_from_execution(summary, error, config.execution_mode)
        run_status = RunStatus.ERROR
        if config.execution_mode in {ExecutionMode.AGENT_MCP, ExecutionMode.REMOTE_AGENT} and scene_status is SceneStatus.PASSED and agent_status in {
            AgentStatus.MAX_STEPS_REACHED,
            AgentStatus.INVALID_RESPONSE,
            AgentStatus.REPEATED_ACTION_DETECTED,
            AgentStatus.DUPLICATE_OBJECT_DETECTED,
            AgentStatus.NO_PROGRESS_DETECTED,
            AgentStatus.RUNTIME_ERROR,
        }:
            agent_status = AgentStatus.COMPLETED_AFTER_SCENE_PASSED
            run_status = RunStatus.PASSED
        _error_summary: dict = {**(summary or {})}
        if validation_result_path is not None and "validation" not in _error_summary:
            try:
                _vr = SceneValidationResult.model_validate_json(
                    validation_result_path.read_text(encoding="utf-8")
                )
                _error_summary["validation"] = _vr.summary
            except Exception:
                pass
        result = RunResult(
            run_id=config.run_id,
            task_id=config.task_id,
            status=run_status,
            run_status=run_status,
            agent_status=agent_status,
            scene_status=scene_status,
            execution_mode=config.execution_mode,
            validation_result_path=validation_result_path,
            scene_snapshot_path=scene_snapshot_path,
            artifacts_dir=layout.run_dir(),
            total_score=total_score,
            overall_status=overall_status,
            started_at=started_at,
            finished_at=_now_utc(),
            duration_sec=time.perf_counter() - started_perf,
            error=error,
            structured_error=structured_error,
            summary={
                **_error_summary,
                **_run_metadata_summary(config, summary),
                "structured_error": structured_error,
            },
        )
        if not layout.metrics_json().exists():
            _write_metrics([], layout.metrics_json())
        _write_run_result(result, layout.run_result_json())
        _write_not_available_markers(layout, structured_error, scene_snapshot_path, validation_result_path)
        _write_stub_trace_if_needed(config, layout, structured_error, started_at, result.finished_at)
        write_run_artifact_manifest(result, layout, structured_error=structured_error)
        return result


def _load_snapshot(path: Path) -> SceneSnapshot:
    try:
        return SceneSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise OSError(f"Failed to read scene snapshot {path}: {error}") from error
    except ValidationError as error:
        raise ValueError(f"Invalid SceneSnapshot in {path}: {error}") from error


def _write_validation_result(result: SceneValidationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def _write_run_result(result: RunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def _write_metrics(metrics: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [metric.model_dump(mode="json") for metric in metrics]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_not_available_markers(
    layout: RunArtifactLayout,
    structured_error: dict,
    scene_snapshot_path: Path | None,
    validation_result_path: Path | None,
) -> None:
    if scene_snapshot_path is None and not layout.scene_snapshot_json().exists():
        _write_not_available_marker(layout.run_dir() / "scene_snapshot_not_available.json", structured_error)
    if validation_result_path is None and not layout.validation_result_json().exists():
        _write_not_available_marker(layout.run_dir() / "validation_result_not_available.json", structured_error)


def _write_not_available_marker(path: Path, structured_error: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "available": False,
        "reason": structured_error.get("error_type", "UnclassifiedError"),
        "message": structured_error.get("message", ""),
        "failure_stage": structured_error.get("failure_stage"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_stub_trace_if_needed(
    config: RunConfig,
    layout: RunArtifactLayout,
    structured_error: dict,
    started_at: str,
    finished_at: str | None,
) -> None:
    if config.execution_mode not in {ExecutionMode.AGENT_MCP, ExecutionMode.REMOTE_AGENT}:
        return
    trace_path = layout.run_dir() / "agent_trace.json"
    if trace_path.exists():
        return
    strategy = _strategy_name(config.metadata.get("agent_strategy"))
    trace = AgentTrace(
        run_id=config.run_id,
        task_id=config.task_id,
        agent_id=str(config.metadata.get("agent_id") or config.agent_config_path or "unknown"),
        strategy=strategy,
        model=_effective_model(config),
        steps=[],
        success=False,
        error=structured_error,
        structured_error=structured_error,
        final_message=None,
        started_at=started_at,
        finished_at=finished_at,
        metadata={
            "stub_trace": True,
            "failure_stage": structured_error.get("failure_stage"),
            "mcp_profile": config.mcp_profile,
        },
    )
    write_agent_trace(trace, trace_path)


def _strategy_name(value: object) -> AgentStrategyName:
    try:
        return AgentStrategyName(str(value or AgentStrategyName.DIRECT_TOOL_CALLING.value))
    except ValueError:
        return AgentStrategyName.DIRECT_TOOL_CALLING


def _copy_snapshot_to_layout(source: Path, destination: Path) -> Path:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _run_status_from_validation(status: ValidationStatus) -> RunStatus:
    if status is ValidationStatus.PASSED:
        return RunStatus.PASSED
    return RunStatus.FAILED


def _scene_status_from_validation(status: ValidationStatus | None) -> SceneStatus:
    if status is ValidationStatus.PASSED or status is ValidationStatus.WARNING:
        return SceneStatus.PASSED
    if status is ValidationStatus.FAILED:
        return SceneStatus.FAILED
    if status is ValidationStatus.SKIPPED:
        return SceneStatus.SKIPPED
    return SceneStatus.NOT_AVAILABLE


def _combined_run_status(agent_status: AgentStatus, scene_status: SceneStatus) -> RunStatus:
    if (
        agent_status in {AgentStatus.COMPLETED, AgentStatus.COMPLETED_AFTER_SCENE_PASSED}
        and scene_status is SceneStatus.PASSED
    ):
        return RunStatus.PASSED
    if scene_status is SceneStatus.FAILED:
        return RunStatus.FAILED
    return RunStatus.ERROR


def _agent_status_from_execution(
    metadata: dict | None,
    error: str | None,
    execution_mode: ExecutionMode | None = None,
) -> AgentStatus:
    if execution_mode not in {ExecutionMode.AGENT_MCP, ExecutionMode.REMOTE_AGENT}:
        return AgentStatus.COMPLETED if error is None else AgentStatus.RUNTIME_ERROR
    agent_run = None
    if isinstance(metadata, dict):
        execution = metadata.get("execution")
        if isinstance(execution, dict):
            agent_run = execution.get("agent_run")
        if agent_run is None:
            agent_run = metadata.get("agent_run")
    if isinstance(agent_run, dict):
        status = agent_run.get("status")
        run_error = str(agent_run.get("error") or "")
        if status == "passed":
            return AgentStatus.COMPLETED
        metadata = agent_run.get("metadata")
        if isinstance(metadata, dict):
            if int(metadata.get("repeated_action_count", 0) or 0) > 0:
                return AgentStatus.REPEATED_ACTION_DETECTED
            if int(metadata.get("duplicate_object_count", 0) or 0) > 0:
                return AgentStatus.DUPLICATE_OBJECT_DETECTED
        if "no_progress_detected" in run_error:
            return AgentStatus.NO_PROGRESS_DETECTED
        if "max_steps" in run_error or "reached max_steps" in run_error:
            return AgentStatus.MAX_STEPS_REACHED
        if "repeated the same action" in run_error:
            return AgentStatus.REPEATED_ACTION_DETECTED
        if "duplicate object" in run_error:
            return AgentStatus.DUPLICATE_OBJECT_DETECTED
        if "did not include action" in run_error or "No tool call" in run_error or "Invalid JSON" in run_error:
            return AgentStatus.INVALID_RESPONSE
        if "Tool" in run_error or "tool" in run_error or "Unknown" in run_error:
            return AgentStatus.TOOL_ERROR
    text = error or ""
    if "no_progress_detected" in text:
        return AgentStatus.NO_PROGRESS_DETECTED
    if "max_steps" in text:
        return AgentStatus.MAX_STEPS_REACHED
    if "No tool call" in text or "Invalid JSON" in text or "did not include action" in text:
        return AgentStatus.INVALID_RESPONSE
    if "Tool" in text or "tool" in text or "Unknown" in text:
        return AgentStatus.TOOL_ERROR
    return AgentStatus.RUNTIME_ERROR


def _run_metadata_summary(config: RunConfig, execution_metadata: dict | None = None) -> dict[str, object]:
    return {
        "agent_id": config.metadata.get("agent_id"),
        "strategy": config.metadata.get("agent_strategy"),
        "mcp_profile": config.metadata.get("mcp_profile") or config.mcp_profile,
        "model": _effective_model(config, execution_metadata),
        "repetition": config.metadata.get("repetition"),
    }


def _effective_model(config: RunConfig, execution_metadata: dict | None = None) -> str | None:
    model_id = config.metadata.get("model_id")
    if isinstance(model_id, str) and model_id and model_id != "default":
        return model_id
    if isinstance(execution_metadata, dict):
        agent_run = execution_metadata.get("agent_run")
        if isinstance(agent_run, dict):
            metadata = agent_run.get("metadata")
            if isinstance(metadata, dict) and metadata.get("model"):
                return str(metadata["model"])
    return None


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
