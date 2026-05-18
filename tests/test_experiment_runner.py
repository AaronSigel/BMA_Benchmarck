import json
from pathlib import Path

from benchmark.runner.config_loader import load_run_config
from benchmark.runner.execution import ExecutionBackend, ExecutionResult
from benchmark.runner.experiment_runner import ExperimentRunner
from benchmark.runner.models import ExecutionMode, RunConfig, RunResult, RunStatus
from benchmark.runner.paths import RunArtifactLayout
from benchmark.validation.models import SceneValidationResult, ValidationStatus

RUNNER_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "runner"


class PassingSceneValidator:
    def validate(self, task, snapshot, artifacts_dir=None) -> SceneValidationResult:
        return SceneValidationResult(
            task_id=task.id,
            overall_status=ValidationStatus.PASSED,
            total_score=1.0,
            validators=[],
            issues=[],
            summary={
                "validators_total": 0,
                "validators_run": 0,
                "issues_total": 0,
                "artifacts_dir": str(artifacts_dir),
            },
        )


class ErrorWithSnapshotBackend(ExecutionBackend):
    mode = ExecutionMode.EXTERNAL_SNAPSHOT

    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = snapshot_path

    def execute(self, config: RunConfig) -> ExecutionResult:
        return ExecutionResult(
            ok=False,
            scene_snapshot_path=self.snapshot_path,
            artifacts_dir=config.output_dir,
            error="agent failed after mutating scene",
            metadata={"mode": "test"},
        )


def make_external_snapshot_config(tmp_path: Path, **overrides: object) -> RunConfig:
    config = load_run_config(RUNNER_FIXTURES_DIR / "run_external_snapshot.yaml")
    updates = {"output_dir": tmp_path / "runs"}
    updates.update(overrides)
    return config.model_copy(update=updates)


def load_expected_run_result() -> RunResult:
    return RunResult.model_validate_json(
        (RUNNER_FIXTURES_DIR / "expected_run_result.json").read_text(encoding="utf-8")
    )


def test_experiment_runner_runs_external_snapshot_and_writes_results(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "benchmark.runner.experiment_runner.SceneValidator",
        PassingSceneValidator,
    )
    config = make_external_snapshot_config(tmp_path)
    layout = RunArtifactLayout.from_run_output_dir(config.output_dir, config.run_id)

    result = ExperimentRunner().run(config)
    expected = load_expected_run_result()

    assert result.status is RunStatus.PASSED
    assert result.run_id == expected.run_id
    assert result.task_id == expected.task_id
    assert result.status == expected.status
    assert result.execution_mode == expected.execution_mode
    assert result.total_score == expected.total_score
    assert result.overall_status == expected.overall_status
    assert result.validation_result_path == layout.validation_result_json()
    assert result.scene_snapshot_path == layout.scene_snapshot_json()
    assert layout.validation_result_json().is_file()
    assert layout.run_result_json().is_file()
    assert layout.scene_snapshot_json().is_file()
    assert layout.metrics_json().is_file()
    assert layout.logs_dir().is_dir()

    validation_payload = json.loads(
        layout.validation_result_json().read_text(encoding="utf-8")
    )
    run_payload = json.loads(layout.run_result_json().read_text(encoding="utf-8"))
    assert validation_payload["overall_status"] == "passed"
    assert run_payload["status"] == "passed"
    assert run_payload["validation_result_path"].endswith("validation_result.json")


def test_experiment_runner_returns_error_when_snapshot_is_missing(tmp_path: Path) -> None:
    config = load_run_config(RUNNER_FIXTURES_DIR / "invalid_missing_snapshot.yaml")
    config = config.model_copy(update={"output_dir": tmp_path / "runs"})
    layout = RunArtifactLayout.from_run_output_dir(config.output_dir, config.run_id)

    result = ExperimentRunner().run(config)

    assert result.status is RunStatus.ERROR
    assert result.error is not None
    assert "scene snapshot does not exist" in result.error
    assert result.validation_result_path is None
    assert layout.run_result_json().is_file()


def test_experiment_runner_writes_partial_validation_for_error_with_snapshot(
    tmp_path: Path,
) -> None:
    config = make_external_snapshot_config(tmp_path)
    layout = RunArtifactLayout.from_run_output_dir(config.output_dir, config.run_id)
    snapshot_path = Path("tests/fixtures/validation/valid_geometry_snapshot.json")

    result = ExperimentRunner(
        backends={
            ExecutionMode.EXTERNAL_SNAPSHOT: ErrorWithSnapshotBackend(snapshot_path),
        }
    ).run(config)

    assert result.status is RunStatus.ERROR
    assert result.scene_snapshot_path == layout.scene_snapshot_json()
    assert result.validation_result_path == layout.validation_result_json()
    assert layout.scene_snapshot_json().is_file()
    assert layout.validation_result_json().is_file()


def test_experiment_runner_uses_task_registry_when_task_path_is_absent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from benchmark.tasks.loader import load_task
    from benchmark.tasks.registry import TaskRegistry

    monkeypatch.setattr(
        "benchmark.runner.experiment_runner.SceneValidator",
        PassingSceneValidator,
    )
    task = load_task("tasks/geometry/geometry_001_basic_primitives.yaml")
    config = make_external_snapshot_config(tmp_path, task_path=None)

    result = ExperimentRunner(task_registry=TaskRegistry([task])).run(config)

    assert result.status is RunStatus.PASSED
    assert result.task_id == task.id
