import json
import socket
from pathlib import Path

from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.matrix import load_matrix
from benchmark.experiments.models import ExperimentMatrix
from benchmark.experiments.readiness import (
    check_experiment_readiness,
    check_matrix_readiness,
    readiness_result_to_json,
    write_readiness_result,
)
from benchmark.runner.models import ExecutionMode


def test_smoke_matrix_readiness_passes_without_external_services() -> None:
    matrix = load_matrix("configs/matrices/smoke_matrix.yaml")

    result = check_matrix_readiness(matrix)

    assert result.ok is True
    assert result.errors == []
    assert not any("API key" in warning for warning in result.warnings)
    assert _requirement_names(result) >= {
        "tasks_found",
        "agent_configs_found",
        "mcp_configs_found",
        "output_root_writable",
    }


def test_smoke_experiment_readiness_passes_without_external_services() -> None:
    config = generate_experiment_config(load_matrix("configs/matrices/smoke_matrix.yaml"))

    result = check_experiment_readiness(config)

    assert result.ok is True
    assert result.errors == []


def test_api_models_matrix_warns_about_missing_api_keys(monkeypatch) -> None:
    for key in ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        monkeypatch.delenv(key, raising=False)

    result = check_matrix_readiness(load_matrix("configs/matrices/api_models_matrix.yaml"))

    assert any("OPENROUTER_API_KEY" in warning for warning in result.warnings)
    assert any("OPENAI_API_KEY" in warning for warning in result.warnings)
    assert any("ANTHROPIC_API_KEY" in warning for warning in result.warnings)
    assert not any("API_KEY" in error for error in result.errors)


def test_missing_api_key_is_error_in_strict_mode(monkeypatch) -> None:
    for key in ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    matrix = load_matrix("configs/matrices/api_models_matrix.yaml")
    matrix.metadata["strict_readiness"] = True

    result = check_matrix_readiness(matrix)

    assert result.ok is False
    assert any("OPENROUTER_API_KEY" in error for error in result.errors)
    assert any("OPENAI_API_KEY" in error for error in result.errors)
    assert any("ANTHROPIC_API_KEY" in error for error in result.errors)


def test_missing_task_is_readiness_error() -> None:
    matrix = ExperimentMatrix(
        matrix_id="missing_task",
        tasks={"ids": ["does_not_exist"]},
        agents={"ids": ["mock_agent"]},
        mcp_profiles=["minimal"],
        execution_modes=[ExecutionMode.EXTERNAL_SNAPSHOT],
        output_root="artifacts/experiments/missing_task_readiness",
    )

    result = check_matrix_readiness(matrix)

    assert result.ok is False
    assert any("does_not_exist" in error for error in result.errors)


def test_e2e_like_matrix_requires_blender_and_mcp(monkeypatch) -> None:
    monkeypatch.delenv("BMA_BLENDER_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    def fail_connect(*args, **kwargs):
        raise OSError("refused")

    monkeypatch.setattr(socket, "create_connection", fail_connect)
    matrix = ExperimentMatrix(
        matrix_id="e2e_like",
        tasks={"ids": ["geometry_001_basic_primitives"]},
        agents={"ids": ["mock_agent"]},
        mcp_profiles=["minimal"],
        execution_modes=[ExecutionMode.AGENT_MCP],
        output_root="artifacts/experiments/e2e_like",
    )

    result = check_matrix_readiness(matrix)

    assert result.ok is False
    assert any("Blender executable not found" in error for error in result.errors)
    assert any("Cannot reach Blender socket" in error for error in result.errors)
    assert {"blender_available", "blender_socket_available"} <= _requirement_names(result)


def test_readiness_result_saves_to_json(tmp_path: Path) -> None:
    result = check_matrix_readiness(load_matrix("configs/matrices/smoke_matrix.yaml"))
    output_path = tmp_path / "readiness.json"

    write_readiness_result(result, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    as_json = json.loads(readiness_result_to_json(result))

    assert payload["ok"] is True
    assert as_json["ok"] is True
    assert payload["metadata"]["matrix_id"] == "smoke_matrix"


def test_output_root_writable_requirement(tmp_path: Path) -> None:
    matrix = ExperimentMatrix(
        matrix_id="writable_output",
        tasks={"ids": ["geometry_001_basic_primitives"]},
        agents={"ids": ["mock_agent"]},
        mcp_profiles=["minimal"],
        execution_modes=[ExecutionMode.EXTERNAL_SNAPSHOT],
        output_root=tmp_path / "writable",
    )

    result = check_matrix_readiness(matrix)

    assert result.ok is True
    assert (tmp_path / "writable").is_dir()
    output_requirement = next(
        requirement
        for requirement in result.requirements
        if requirement.name == "output_root_writable"
    )
    assert output_requirement.description == str(tmp_path / "writable")


def test_missing_report_config_is_readiness_error() -> None:
    matrix = ExperimentMatrix(
        matrix_id="missing_report",
        tasks={"ids": ["geometry_001_basic_primitives"]},
        agents={"ids": ["mock_agent"]},
        mcp_profiles=["minimal"],
        execution_modes=[ExecutionMode.EXTERNAL_SNAPSHOT],
        report_config_path="configs/reports/missing.yaml",
    )

    result = check_matrix_readiness(matrix)

    assert result.ok is False
    assert any("report_config_path does not exist" in error for error in result.errors)


def _requirement_names(result) -> set[str]:
    return {requirement.name for requirement in result.requirements}
