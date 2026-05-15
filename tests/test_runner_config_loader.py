from pathlib import Path

import pytest

from benchmark.runner.config_loader import (
    dump_experiment_config,
    dump_run_config,
    load_experiment_config,
    load_run_config,
)
from benchmark.runner.errors import RunnerConfigError
from benchmark.runner.models import ExecutionMode, ExperimentConfig, RunConfig


def make_run_config(**overrides: object) -> RunConfig:
    data = {
        "run_id": "geometry_001_replay",
        "task_id": "geometry_001_basic_primitives",
        "execution_mode": ExecutionMode.EXTERNAL_SNAPSHOT,
        "task_path": Path("tasks/geometry/geometry_001_basic_primitives.yaml"),
        "snapshot_path": Path("artifacts/blender_smoke/scene_snapshot.json"),
        "artifacts_dir": Path("artifacts/blender_smoke"),
        "output_dir": Path("artifacts/runs/geometry_001_replay"),
        "metadata": {"source": "test"},
    }
    data.update(overrides)
    return RunConfig(**data)


def test_example_experiment_config_loads() -> None:
    config = load_experiment_config("configs/example_experiment.yaml")

    assert config.experiment_id == "local_validation_baseline"
    assert len(config.runs) == 1
    assert config.runs[0].execution_mode == ExecutionMode.EXTERNAL_SNAPSHOT
    assert config.runs[0].snapshot_path == Path(
        "tests/fixtures/validation/valid_geometry_snapshot.json"
    )


def test_dump_and_load_run_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    original = make_run_config()

    dump_run_config(original, path)
    loaded = load_run_config(path)

    assert loaded == original


def test_dump_and_load_experiment_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    original = ExperimentConfig(
        experiment_id="local_validation_baseline",
        runs=[make_run_config()],
        metadata={"description": "baseline"},
    )

    dump_experiment_config(original, path)
    loaded = load_experiment_config(path)

    assert loaded == original


def test_invalid_yaml_has_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("experiment_id: [unterminated\n", encoding="utf-8")

    with pytest.raises(RunnerConfigError, match="Failed to parse YAML experiment config"):
        load_experiment_config(path)


def test_top_level_yaml_must_be_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(RunnerConfigError, match="must contain a YAML mapping"):
        load_experiment_config(path)


def test_missing_required_fields_are_reported_by_pydantic(tmp_path: Path) -> None:
    path = tmp_path / "missing.yaml"
    path.write_text(
        "\n".join(
            [
                "experiment_id: local_validation_baseline",
                "runs:",
                "  - run_id: geometry_001_replay",
                "    execution_mode: external_snapshot",
                "    artifacts_dir: artifacts/blender_smoke",
                "    output_dir: artifacts/runs/geometry_001_replay",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(RunnerConfigError) as error:
        load_experiment_config(path)

    assert "Invalid experiment config" in str(error.value)
    assert "task_id" in str(error.value)
