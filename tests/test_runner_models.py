from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.runner.models import (
    ExecutionMode,
    ExperimentConfig,
    RunConfig,
    RunResult,
    RunStatus,
)


def make_run_config(**overrides: object) -> RunConfig:
    data = {
        "run_id": "geometry_001_replay",
        "task_id": "geometry_001_basic_primitives",
        "execution_mode": ExecutionMode.EXTERNAL_SNAPSHOT,
        "task_path": Path("tasks/geometry/geometry_001_basic_primitives.yaml"),
        "snapshot_path": Path("artifacts/blender_smoke/scene_snapshot.json"),
        "artifacts_dir": Path("artifacts/blender_smoke"),
        "output_dir": Path("artifacts/runs/geometry_001_replay"),
    }
    data.update(overrides)
    return RunConfig(**data)


def make_run_result(**overrides: object) -> RunResult:
    data = {
        "run_id": "geometry_001_replay",
        "task_id": "geometry_001_basic_primitives",
        "status": RunStatus.PASSED,
        "execution_mode": ExecutionMode.EXTERNAL_SNAPSHOT,
        "validation_result_path": Path("artifacts/runs/run/validation_result.json"),
        "scene_snapshot_path": Path("artifacts/blender_smoke/scene_snapshot.json"),
        "artifacts_dir": Path("artifacts/blender_smoke"),
        "total_score": 1.0,
        "overall_status": "passed",
        "started_at": "2026-05-15T10:00:00Z",
        "finished_at": "2026-05-15T10:00:01Z",
        "duration_sec": 1.0,
    }
    data.update(overrides)
    return RunResult(**data)


def test_run_config_serializes_to_json() -> None:
    config = make_run_config()

    payload = config.model_dump_json()

    assert '"run_id":"geometry_001_replay"' in payload
    assert '"execution_mode":"external_snapshot"' in payload
    assert '"artifacts_dir":"artifacts/blender_smoke"' in payload


def test_experiment_config_serializes_to_json() -> None:
    config = ExperimentConfig(
        experiment_id="local_validation_baseline",
        runs=[make_run_config()],
        metadata={"description": "baseline"},
    )

    payload = config.model_dump_json()

    assert '"experiment_id":"local_validation_baseline"' in payload
    assert '"runs":[{' in payload
    assert '"description":"baseline"' in payload


def test_experiment_config_rejects_empty_experiment_id() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(experiment_id=" ", runs=[make_run_config()])


def test_run_result_serializes_to_json() -> None:
    result = make_run_result()

    payload = result.model_dump_json()

    assert '"run_id":"geometry_001_replay"' in payload
    assert '"status":"passed"' in payload
    assert '"total_score":1.0' in payload
    assert '"validation_result_path":"artifacts/runs/run/validation_result.json"' in payload


@pytest.mark.parametrize("score", [0.0, 0.5, 1.0, None])
def test_run_result_accepts_valid_total_score(score: float | None) -> None:
    result = make_run_result(total_score=score)

    assert result.total_score == score


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_run_result_rejects_invalid_total_score(score: float) -> None:
    with pytest.raises(ValidationError):
        make_run_result(total_score=score)


@pytest.mark.parametrize("field_name", ["run_id", "task_id"])
def test_run_config_rejects_empty_identifiers(field_name: str) -> None:
    with pytest.raises(ValidationError):
        make_run_config(**{field_name: " "})


@pytest.mark.parametrize("field_name", ["run_id", "task_id"])
def test_run_result_rejects_empty_identifiers(field_name: str) -> None:
    with pytest.raises(ValidationError):
        make_run_result(**{field_name: " "})
