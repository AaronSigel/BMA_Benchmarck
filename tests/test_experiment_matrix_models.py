import json

import pytest
import yaml
from pydantic import ValidationError

from benchmark.experiments.models import (
    EnvironmentRequirement,
    ExperimentMatrix,
    GeneratedExperimentManifest,
    MatrixAgentSelector,
    MatrixMcpSelector,
    MatrixModelSelector,
    MatrixRunVariant,
    MatrixTaskSelector,
    ReadinessCheckResult,
)
from benchmark.experiments.matrix import load_matrix
from benchmark.runner.models import ExecutionMode
from benchmark.tasks.models import DifficultyLevel, TaskCategory


def test_experiment_matrix_accepts_minimal_fields() -> None:
    matrix = ExperimentMatrix(
        matrix_id="baseline",
        title="Baseline",
        description="Baseline experiment matrix",
        tasks=MatrixTaskSelector(ids=["geometry_001_basic_primitives"]),
        agents=MatrixAgentSelector(ids=["mock_agent"]),
        mcp_profiles=["minimal"],
        models=MatrixModelSelector(ids=["mock"]),
        execution_modes=[ExecutionMode.EXTERNAL_SNAPSHOT],
        repetitions=1,
        output_root="artifacts/experiments/baseline",
        report_config_path="configs/reports/default_report.yaml",
        metadata={"stage": 8},
    )

    assert matrix.matrix_id == "baseline"
    assert matrix.tasks.ids == ["geometry_001_basic_primitives"]
    assert matrix.repetitions == 1


def test_matrix_id_must_not_be_empty() -> None:
    with pytest.raises(ValidationError, match="matrix_id must not be empty"):
        ExperimentMatrix(matrix_id=" ")


def test_repetitions_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ExperimentMatrix(matrix_id="baseline", repetitions=0)


def test_task_selector_supports_id_category_and_difficulty() -> None:
    selector = MatrixTaskSelector(
        ids=["geometry_001_basic_primitives"],
        categories=["geometry"],
        difficulties=["easy"],
        tags=["primitives"],
    )

    assert selector.ids == ["geometry_001_basic_primitives"]
    assert selector.categories == [TaskCategory.GEOMETRY]
    assert selector.difficulties == [DifficultyLevel.EASY]
    assert selector.tags == ["primitives"]


def test_task_selector_rejects_empty_ids() -> None:
    with pytest.raises(ValidationError, match="list values must not be empty"):
        MatrixTaskSelector(ids=["valid", " "])


def test_supporting_models_validate_and_serialize() -> None:
    variant = MatrixRunVariant(
        task_id="geometry_001_basic_primitives",
        agent_id="mock_agent",
        mcp_profile="minimal",
        model_id="mock",
        execution_mode=ExecutionMode.EXTERNAL_SNAPSHOT,
        repetition=1,
    )
    requirement = EnvironmentRequirement(name="output_root_writable")
    readiness = ReadinessCheckResult(ok=True, requirements=[requirement])
    manifest = GeneratedExperimentManifest(
        matrix_id="smoke",
        generated_at="2026-05-16T00:00:00Z",
        task_ids=[variant.task_id],
        agent_ids=[variant.agent_id],
        mcp_profiles=[variant.mcp_profile],
        models=["mock"],
        execution_modes=[variant.execution_mode],
        repetitions=1,
        env_requirements=[requirement],
        config_hash="abc123",
    )

    assert readiness.ok is True
    assert json.loads(manifest.model_dump_json())["matrix_id"] == "smoke"


def test_mcp_selector_model_exists_for_profile_config_shape() -> None:
    selector = MatrixMcpSelector(profiles=["minimal", "no_python"])

    assert selector.profiles == ["minimal", "no_python"]


def test_matrix_config_serializes_to_json_and_yaml() -> None:
    matrix = ExperimentMatrix(
        matrix_id="smoke_matrix",
        tasks={"categories": ["geometry"], "difficulties": ["easy"]},
        agents={"ids": ["mock_agent"]},
        mcp_profiles=["minimal"],
        models={"ids": ["mock"]},
        execution_modes=["external_snapshot"],
        repetitions=1,
        output_root="artifacts/experiments/smoke_matrix",
    )

    json_payload = matrix.model_dump_json()
    yaml_payload = yaml.safe_dump(matrix.model_dump(mode="json"), sort_keys=False)
    loaded = yaml.safe_load(yaml_payload)

    assert json.loads(json_payload)["matrix_id"] == "smoke_matrix"
    assert loaded["tasks"]["categories"] == ["geometry"]
    assert loaded["execution_modes"] == ["external_snapshot"]


def test_matrix_yaml_loads_into_model(tmp_path) -> None:
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "matrix_id": "loaded_matrix",
                "tasks": {"ids": ["geometry_001_basic_primitives"]},
                "agents": {"ids": ["mock_agent"]},
                "mcp_profiles": ["minimal"],
                "execution_modes": ["external_snapshot"],
                "repetitions": 1,
                "output_root": "artifacts/experiments/loaded_matrix",
            }
        ),
        encoding="utf-8",
    )

    matrix = load_matrix(matrix_path)

    assert matrix.matrix_id == "loaded_matrix"
    assert matrix.tasks.ids == ["geometry_001_basic_primitives"]
    assert matrix.execution_modes == [ExecutionMode.EXTERNAL_SNAPSHOT]
