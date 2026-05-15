import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from benchmark.runner.models import ExperimentResult


def write_tmp_example_config(tmp_path: Path) -> Path:
    data = yaml.safe_load(Path("configs/example_experiment.yaml").read_text(encoding="utf-8"))
    output_root = tmp_path / "runs"
    for run in data["runs"]:
        run["output_dir"] = str(output_root / run["run_id"])

    config_path = tmp_path / "example_experiment.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_path


def run_experiment(config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.runner.cli",
            "experiment",
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_runner_smoke_scenario_recreates_results_without_blender(tmp_path: Path) -> None:
    config_path = write_tmp_example_config(tmp_path)
    output_root = tmp_path / "runs"

    first = run_experiment(config_path)

    assert first.returncode == 0
    assert "experiment_id: local_validation_baseline" in first.stdout
    assert "total_runs: 1" in first.stdout
    assert "passed: 1" in first.stdout
    assert (output_root / "experiment_result.json").is_file()
    assert (output_root / "summary.csv").is_file()
    assert (
        output_root
        / "geometry_001_replay"
        / "validation_result.json"
    ).is_file()
    assert (output_root / "geometry_001_replay" / "run_result.json").is_file()

    result = ExperimentResult.model_validate_json(
        (output_root / "experiment_result.json").read_text(encoding="utf-8")
    )
    assert result.summary["total_runs"] == 1
    assert result.summary["passed_runs"] == 1

    shutil.rmtree(output_root)
    second = run_experiment(config_path)

    assert second.returncode == 0
    assert (output_root / "experiment_result.json").is_file()
    assert (output_root / "summary.csv").is_file()
