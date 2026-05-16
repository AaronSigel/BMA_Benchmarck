from pathlib import Path

import yaml

from benchmark.runner import cli
from benchmark.runner.models import ExperimentResult
from benchmark.runner.paths import RunArtifactLayout
from benchmark.validation.models import SceneValidationResult, ValidationStatus


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
            },
        )


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def run_config(output_dir: Path, run_id: str = "geometry_001_external") -> dict:
    return {
        "run_id": run_id,
        "task_id": "geometry_001_basic_primitives",
        "execution_mode": "external_snapshot",
        "task_path": "tasks/geometry/geometry_001_basic_primitives.yaml",
        "snapshot_path": "artifacts/blender_smoke/scene_snapshot.json",
        "artifacts_dir": "artifacts/blender_smoke",
        "output_dir": str(output_dir),
    }


def write_smoke_matrix(path: Path, output_root: Path) -> None:
    write_yaml(
        path,
        {
            "matrix_id": "runner_cli_matrix",
            "tasks": {"ids": ["geometry_001_basic_primitives"]},
            "agents": {"ids": ["mock_agent"]},
            "mcp_profiles": ["minimal"],
            "execution_modes": ["external_snapshot"],
            "repetitions": 1,
            "output_root": str(output_root),
            "metadata": {
                "snapshot_path": "tests/fixtures/validation/valid_geometry_snapshot.json",
                "artifacts_dir": "tests/fixtures/validation",
            },
        },
    )


def test_runner_cli_run_creates_run_result(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(
        "benchmark.runner.experiment_runner.SceneValidator",
        PassingSceneValidator,
    )
    config_path = tmp_path / "run.yaml"
    output_dir = tmp_path / "run_output"
    write_yaml(config_path, run_config(output_dir))
    layout = RunArtifactLayout.from_run_output_dir(output_dir, "geometry_001_external")

    exit_code = cli.main(["run", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert layout.run_result_json().is_file()
    assert layout.validation_result_json().is_file()
    assert layout.scene_snapshot_json().is_file()
    assert layout.metrics_json().is_file()
    assert "run_id: geometry_001_external" in captured.out
    assert "status: passed" in captured.out
    assert "overall_status: passed" in captured.out


def test_runner_cli_experiment_creates_outputs(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(
        "benchmark.runner.experiment_runner.SceneValidator",
        PassingSceneValidator,
    )
    output_root = tmp_path / "experiment"
    config_path = tmp_path / "experiment.yaml"
    write_yaml(
        config_path,
        {
            "experiment_id": "local_cli_experiment",
            "runs": [
                run_config(output_root / "geometry_001_external"),
                run_config(
                    output_root / "geometry_001_external_2",
                    "geometry_001_external_2",
                ),
            ],
        },
    )

    exit_code = cli.main(["experiment", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert (output_root / "experiment_result.json").is_file()
    assert (output_root / "summary.csv").is_file()
    assert (output_root / "metrics.csv").is_file()
    assert "experiment_id: local_cli_experiment" in captured.out
    assert "total_runs: 2" in captured.out
    assert "passed: 2" in captured.out
    assert "average_score: 1.000" in captured.out


def test_runner_cli_summarize_reads_experiment_result(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(
        "benchmark.runner.experiment_runner.SceneValidator",
        PassingSceneValidator,
    )
    output_root = tmp_path / "experiment"
    config_path = tmp_path / "experiment.yaml"
    write_yaml(
        config_path,
        {
            "experiment_id": "local_cli_experiment",
            "runs": [run_config(output_root / "geometry_001_external")],
        },
    )
    assert cli.main(["experiment", "--config", str(config_path)]) == 0
    capsys.readouterr()

    exit_code = cli.main(
        ["summarize", "--results", str(output_root / "experiment_result.json")]
    )

    captured = capsys.readouterr()
    result = ExperimentResult.model_validate_json(
        (output_root / "experiment_result.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert result.experiment_id == "local_cli_experiment"
    assert "experiment_id: local_cli_experiment" in captured.out
    assert "total_runs: 1" in captured.out
    assert "passed: 1" in captured.out


def test_runner_cli_summarize_reports_missing_result(capsys, tmp_path: Path) -> None:
    exit_code = cli.main(
        ["summarize", "--results", str(tmp_path / "missing_experiment_result.json")]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "ERROR:" in captured.out
    assert "Failed to read experiment result" in captured.out


def test_runner_cli_matrix_runs_stage8_smoke_matrix(tmp_path: Path, capsys) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    output_root = tmp_path / "matrix_out"
    write_smoke_matrix(matrix_path, output_root)

    exit_code = cli.main(["matrix", "--config", str(matrix_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "experiment_id: runner_cli_matrix" in captured.out
    assert "total_runs: 1" in captured.out
    assert (output_root / "manifest.json").is_file()
    assert (output_root / "experiment_result.json").is_file()


def test_runner_cli_matrix_analyze_uses_stage7_analysis(tmp_path: Path, capsys) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    output_root = tmp_path / "matrix_out"
    write_smoke_matrix(matrix_path, output_root)

    exit_code = cli.main(["matrix", "--config", str(matrix_path), "--analyze"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "experiment_id: matrix_out" in captured.out
    assert (output_root / "experiment_analysis.json").is_file()


def test_runner_cli_matrix_report_builds_stage7_reports(tmp_path: Path, capsys) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    output_root = tmp_path / "matrix_out"
    write_smoke_matrix(matrix_path, output_root)

    exit_code = cli.main(["matrix", "--config", str(matrix_path), "--report"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"report: {output_root / 'report.md'}" in captured.out
    assert (output_root / "experiment_analysis.json").is_file()
    assert (output_root / "report.md").is_file()
    assert (output_root / "report.html").is_file()


def test_runner_cli_matrix_readiness_error_returns_one(tmp_path: Path, capsys) -> None:
    matrix_path = tmp_path / "bad_matrix.yaml"
    write_yaml(
        matrix_path,
        {
            "matrix_id": "bad_matrix",
            "tasks": {"ids": ["missing_task"]},
            "agents": {"ids": ["mock_agent"]},
            "mcp_profiles": ["minimal"],
            "execution_modes": ["external_snapshot"],
            "output_root": str(tmp_path / "out"),
        },
    )

    exit_code = cli.main(["matrix", "--config", str(matrix_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "ERROR:" in captured.out
    assert "missing_task" in captured.out
