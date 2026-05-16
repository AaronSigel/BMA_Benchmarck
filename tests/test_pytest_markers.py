from pathlib import Path

import pytest
import yaml


E2E_MARKERS = {
    "e2e": "full end-to-end benchmark tests",
    "api_e2e": "E2E requiring external API models",
    "mcp_e2e": "E2E requiring Blender + MCP",
    "remote_agent_e2e": "E2E requiring external hosted agent",
}


def _marker_names(markers: list[str]) -> dict[str, str]:
    names = {}
    for marker in markers:
        name, _, description = marker.partition(":")
        names[name] = description.strip()
    return names


def _addopts_expression(pytestconfig: pytest.Config) -> str:
    addopts = pytestconfig.getini("addopts")
    if isinstance(addopts, str):
        return addopts
    return " ".join(addopts)


def test_e2e_pytest_markers_are_registered(pytestconfig: pytest.Config) -> None:
    markers = _marker_names(pytestconfig.getini("markers"))

    for name, description in E2E_MARKERS.items():
        assert markers[name] == description


def test_default_pytest_addopts_exclude_heavy_e2e_markers(
    pytestconfig: pytest.Config,
) -> None:
    addopts = _addopts_expression(pytestconfig)

    for name in E2E_MARKERS:
        assert f"not {name}" in addopts


def test_opt_in_experiment_matrices_reference_registered_markers() -> None:
    markers = E2E_MARKERS.keys()
    matrices = [
        Path("configs/matrices/api_models_matrix.yaml"),
        Path("configs/matrices/remote_agents_matrix.yaml"),
    ]

    for matrix_path in matrices:
        matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
        assert matrix["metadata"]["pytest_marker"] in markers
