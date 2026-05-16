import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from benchmark.blender.models import SceneSnapshot
from benchmark.metrics.validation_metrics import metrics_from_validation_result
from benchmark.runner.execution import (
    BlenderSmokeBackend,
    ExecutionBackend,
    ExternalSnapshotBackend,
    ReplayBackend,
)
from benchmark.runner.models import ExecutionMode, RunConfig, RunResult, RunStatus
from benchmark.runner.paths import RunArtifactLayout
from benchmark.tasks.loader import load_task
from benchmark.tasks.registry import TaskRegistry
from benchmark.validation.models import SceneValidationResult, ValidationStatus
from benchmark.validation.scene_validator import SceneValidator
from benchmark.agent.execution_backend import AgentExecutionBackend, RemoteAgentExecutionBackend


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

        execution_config = config.model_copy(update={"output_dir": layout.run_dir()})
        execution_result = backend.execute(execution_config)
        if not execution_result.ok:
            return self._finish_error(
                config,
                layout,
                started_at,
                started_perf,
                execution_result.error or "execution failed",
                scene_snapshot_path=execution_result.scene_snapshot_path,
                summary={"execution": execution_result.metadata},
            )

        if execution_result.scene_snapshot_path is None:
            return self._finish_error(
                config,
                layout,
                started_at,
                started_perf,
                "execution did not return scene_snapshot_path",
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
                summary={"execution": execution_result.metadata},
            )

        status = _run_status_from_validation(validation_result.overall_status)
        result = RunResult(
            run_id=config.run_id,
            task_id=config.task_id,
            status=status,
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
            },
        )
        _write_run_result(result, layout.run_result_json())
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
        summary: dict | None = None,
    ) -> RunResult:
        if scene_snapshot_path is not None and scene_snapshot_path.exists():
            try:
                scene_snapshot_path = _copy_snapshot_to_layout(
                    scene_snapshot_path,
                    layout.scene_snapshot_json(),
                )
            except OSError:
                pass
        result = RunResult(
            run_id=config.run_id,
            task_id=config.task_id,
            status=RunStatus.ERROR,
            execution_mode=config.execution_mode,
            validation_result_path=None,
            scene_snapshot_path=scene_snapshot_path,
            artifacts_dir=layout.run_dir(),
            total_score=None,
            overall_status=None,
            started_at=started_at,
            finished_at=_now_utc(),
            duration_sec=time.perf_counter() - started_perf,
            error=error,
            summary=summary or {},
        )
        _write_run_result(result, layout.run_result_json())
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


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
