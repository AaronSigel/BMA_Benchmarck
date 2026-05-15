import csv
import json
from pathlib import Path

from benchmark.runner.batch_runner import BatchRunner
from benchmark.runner.config_loader import load_experiment_config
from benchmark.runner.models import (
    ExperimentConfig,
    RunConfig,
    RunResult,
    RunStatus,
)

RUNNER_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "runner"


class StubRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, config: RunConfig) -> RunResult:
        self.calls.append(config.run_id)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = RunResult(
            run_id=config.run_id,
            task_id=config.task_id,
            status=RunStatus.ERROR if "error" in config.run_id else RunStatus.PASSED,
            execution_mode=config.execution_mode,
            validation_result_path=output_dir / "validation_result.json",
            scene_snapshot_path=config.snapshot_path,
            artifacts_dir=config.artifacts_dir,
            total_score=None if "error" in config.run_id else 1.0,
            overall_status=None if "error" in config.run_id else "passed",
            started_at="2026-05-15T10:00:00Z",
            finished_at="2026-05-15T10:00:01Z",
            duration_sec=1.0,
            error="boom" if "error" in config.run_id else None,
            summary={},
        )
        (output_dir / "run_result.json").write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return result


def make_run_config(output_root: Path, run_id: str) -> RunConfig:
    fixture_config = load_experiment_config(
        RUNNER_FIXTURES_DIR / "experiment_external_snapshots.yaml"
    )
    by_run_id = {run.run_id: run for run in fixture_config.runs}
    return by_run_id[run_id].model_copy(update={"output_dir": output_root / run_id})


def test_batch_runner_runs_all_runs_and_writes_experiment_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "experiment"
    config = load_experiment_config(RUNNER_FIXTURES_DIR / "experiment_external_snapshots.yaml")
    config = config.model_copy(
        update={
            "runs": [
                run.model_copy(update={"output_dir": output_root / run.run_id})
                for run in config.runs
            ]
        }
    )
    runner = StubRunner()

    result = BatchRunner(runner=runner).run_experiment(config)

    assert runner.calls == ["geometry_001_external", "geometry_001_external_2"]
    assert [run.status for run in result.runs] == [
        RunStatus.PASSED,
        RunStatus.PASSED,
    ]
    assert result.summary["total_runs"] == 2
    assert result.summary["passed_runs"] == 2
    assert result.summary["error_runs"] == 0
    assert result.summary["average_score"] == 1.0

    experiment_result_path = output_root / "experiment_result.json"
    summary_json_path = output_root / "summary.json"
    summary_csv_path = output_root / "summary.csv"
    assert experiment_result_path.is_file()
    assert summary_json_path.is_file()
    assert summary_csv_path.is_file()

    experiment_payload = json.loads(experiment_result_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    assert experiment_payload["experiment_id"] == "runner_fixture_external_snapshots"
    assert len(experiment_payload["runs"]) == 2
    assert summary_payload["total_runs"] == 2

    with summary_csv_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    with (RUNNER_FIXTURES_DIR / "expected_summary.csv").open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        expected_rows = list(csv.DictReader(file))
    stable_columns = [
        "run_id",
        "task_id",
        "status",
        "execution_mode",
        "total_score",
        "overall_status",
        "duration_sec",
        "error",
    ]
    assert [
        {column: row[column] for column in stable_columns}
        for row in rows
    ] == [
        {column: row[column] for column in stable_columns}
        for row in expected_rows
    ]


def test_batch_runner_keeps_run_alias(tmp_path: Path) -> None:
    output_root = tmp_path / "experiment"
    config = ExperimentConfig(
        experiment_id="local_batch",
        runs=[make_run_config(output_root, "geometry_001_external")],
    )

    result = BatchRunner(runner=StubRunner()).run(config)

    assert result.experiment_id == "local_batch"
    assert (output_root / "experiment_result.json").is_file()


def test_batch_runner_error_run_does_not_stop_experiment(tmp_path: Path) -> None:
    output_root = tmp_path / "experiment"
    success_config = make_run_config(output_root, "geometry_001_external")
    error_config = success_config.model_copy(
        update={
            "run_id": "geometry_001_error",
            "output_dir": output_root / "geometry_001_error",
        }
    )
    final_config = make_run_config(output_root, "geometry_001_external_2")
    config = ExperimentConfig(
        experiment_id="local_batch_with_error",
        runs=[success_config, error_config, final_config],
    )
    runner = StubRunner()

    result = BatchRunner(runner=runner).run_experiment(config)

    assert runner.calls == [
        "geometry_001_external",
        "geometry_001_error",
        "geometry_001_external_2",
    ]
    assert [run.status for run in result.runs] == [
        RunStatus.PASSED,
        RunStatus.ERROR,
        RunStatus.PASSED,
    ]
    assert result.summary["total_runs"] == 3
    assert result.summary["passed_runs"] == 2
    assert result.summary["error_runs"] == 1
    assert (output_root / "experiment_result.json").is_file()
    assert (output_root / "summary.csv").is_file()
