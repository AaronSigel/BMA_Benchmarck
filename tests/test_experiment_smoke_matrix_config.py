from pathlib import Path

from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.matrix import load_matrix
from benchmark.runner.execution import ExternalSnapshotBackend
from benchmark.runner.models import ExecutionMode


def test_smoke_matrix_generates_ci_safe_run_config() -> None:
    matrix_path = Path("configs/matrices/smoke_matrix.yaml")

    config = generate_experiment_config(load_matrix(matrix_path))

    assert config.experiment_id == "smoke_matrix"
    assert len(config.runs) == 1
    run = config.runs[0]
    assert run.run_id == "smoke_matrix__geometry_001_basic_primitives__mock_agent__minimal__r1"
    assert run.task_id == "geometry_001_basic_primitives"
    assert run.execution_mode is ExecutionMode.EXTERNAL_SNAPSHOT
    assert run.snapshot_path == Path("tests/fixtures/validation/valid_geometry_snapshot.json")
    assert run.agent_config_path == Path("configs/agents/mock_agent.yaml")
    assert run.mcp_config_path == Path("configs/mcp/minimal.yaml")
    assert run.output_dir == Path("artifacts/experiments/smoke_matrix") / run.run_id


def test_smoke_matrix_external_snapshot_backend_runs_without_services() -> None:
    config = generate_experiment_config(load_matrix("configs/matrices/smoke_matrix.yaml"))

    result = ExternalSnapshotBackend().execute(config.runs[0])

    assert result.ok is True
    assert result.scene_snapshot_path == Path("tests/fixtures/validation/valid_geometry_snapshot.json")
