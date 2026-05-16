from pathlib import Path

import pytest
import yaml

from benchmark.experiments.matrix import (
    DEFAULT_MCP_PROFILES,
    SUPPORTED_MCP_PROFILES,
    ExperimentMatrixError,
    load_mcp_profile_pool,
    select_mcp_profiles,
    select_mcp_profiles_by_names,
)
from benchmark.experiments.models import ExperimentMatrix


def write_mcp_config(directory: Path, profile: str) -> None:
    (directory / f"{profile}.yaml").write_text(
        yaml.safe_dump(
            {
                "profile": profile,
                "server_distribution": "upstream",
                "command": "uvx",
                "args": ["blender-mcp"],
                "env": {"BMA_MCP_PROFILE": profile},
            }
        ),
        encoding="utf-8",
    )


def make_profile_pool(tmp_path: Path) -> dict[str, dict]:
    for profile in SUPPORTED_MCP_PROFILES:
        write_mcp_config(tmp_path, profile)
    return load_mcp_profile_pool(tmp_path)


def test_load_mcp_profile_pool_reads_supported_profiles(tmp_path: Path) -> None:
    pool = make_profile_pool(tmp_path)

    assert list(pool) == [
        "full",
        "inspection_enabled",
        "minimal",
        "no_python",
        "python_enabled",
    ]
    assert pool["minimal"]["config_path"] == tmp_path / "minimal.yaml"


def test_smoke_matrix_can_use_only_minimal_profile(tmp_path: Path) -> None:
    matrix = ExperimentMatrix(matrix_id="smoke", mcp_profiles=["minimal"])

    selected = select_mcp_profiles(matrix, make_profile_pool(tmp_path))

    assert [profile["profile"] for profile in selected] == ["minimal"]


def test_baseline_matrix_uses_safe_profiles(tmp_path: Path) -> None:
    matrix = ExperimentMatrix(
        matrix_id="baseline",
        mcp_profiles=["minimal", "no_python", "inspection_enabled"],
    )

    selected = select_mcp_profiles(matrix, make_profile_pool(tmp_path))

    assert [profile["profile"] for profile in selected] == [
        "minimal",
        "no_python",
        "inspection_enabled",
    ]


def test_default_profiles_do_not_include_python_enabled_or_full(tmp_path: Path) -> None:
    selected = select_mcp_profiles(ExperimentMatrix(matrix_id="default"), make_profile_pool(tmp_path))

    assert DEFAULT_MCP_PROFILES == ("minimal", "no_python", "inspection_enabled")
    assert [profile["profile"] for profile in selected] == list(DEFAULT_MCP_PROFILES)
    assert "python_enabled" not in [profile["profile"] for profile in selected]
    assert "full" not in [profile["profile"] for profile in selected]


def test_python_enabled_and_full_are_opt_in(tmp_path: Path) -> None:
    matrix = ExperimentMatrix(
        matrix_id="opt_in",
        mcp_profiles=["python_enabled", "full"],
    )

    selected = select_mcp_profiles(matrix, make_profile_pool(tmp_path))

    assert [profile["profile"] for profile in selected] == ["python_enabled", "full"]


def test_unknown_profile_is_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ExperimentMatrixError, match="Unsupported MCP profile: unsafe"):
        select_mcp_profiles_by_names(make_profile_pool(tmp_path), ["unsafe"])


def test_missing_profile_config_is_configuration_error(tmp_path: Path) -> None:
    write_mcp_config(tmp_path, "minimal")
    pool = load_mcp_profile_pool(tmp_path)

    with pytest.raises(ExperimentMatrixError, match="no_python"):
        select_mcp_profiles_by_names(pool, ["no_python"])


def test_builtin_mcp_profile_pool_supports_stage_8_defaults() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "mcp"
    pool = load_mcp_profile_pool(config_dir)
    selected = select_mcp_profiles(ExperimentMatrix(matrix_id="default"), pool)

    assert set(pool) == set(SUPPORTED_MCP_PROFILES)
    assert [profile["profile"] for profile in selected] == list(DEFAULT_MCP_PROFILES)
