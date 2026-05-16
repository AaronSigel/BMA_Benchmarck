from pathlib import Path

from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.matrix import load_matrix
from benchmark.runner.models import ExecutionMode


BASELINE_TASK_IDS = [
    "geometry_001_basic_primitives",
    "geometry_002_positions",
    "geometry_003_dimensions",
    "geometry_004_rotation",
    "geometry_005_composition",
    "materials_001_basic_colors",
    "materials_002_roughness",
    "materials_003_metallic",
    "materials_004_multiple_objects",
    "materials_005_material_composition",
    "lighting_001_area_light",
    "lighting_002_sun_light",
    "camera_001_front_view",
    "camera_002_top_view",
    "export_001_blend_file",
]


def test_baseline_matrix_has_expected_safe_scope() -> None:
    matrix = load_matrix("configs/matrices/baseline_matrix.yaml")

    assert matrix.tasks.ids == BASELINE_TASK_IDS
    assert matrix.agents.ids == [
        "direct_openrouter",
        "react_openrouter",
        "plan_execute_openrouter",
    ]
    assert matrix.mcp_profiles == ["minimal", "no_python", "inspection_enabled"]
    assert "python_enabled" not in matrix.mcp_profiles
    assert "full" not in matrix.mcp_profiles
    assert matrix.repetitions == 3


def test_baseline_matrix_generates_reproducible_run_count() -> None:
    matrix = load_matrix("configs/matrices/baseline_matrix.yaml")

    first = generate_experiment_config(matrix)
    second = generate_experiment_config(matrix)

    expected_runs = 15 * 3 * 3 * 1 * 3
    assert len(first.runs) == expected_runs
    assert first == second
    assert len({run.run_id for run in first.runs}) == expected_runs
    assert len({run.output_dir for run in first.runs}) == expected_runs
    assert all(run.execution_mode is ExecutionMode.AGENT_MCP for run in first.runs)
    assert all(run.mcp_profile in {"minimal", "no_python", "inspection_enabled"} for run in first.runs)


def test_baseline_matrix_run_ids_are_stable() -> None:
    config = generate_experiment_config(load_matrix("configs/matrices/baseline_matrix.yaml"))

    assert config.runs[0].run_id == (
        "baseline_matrix__geometry_001_basic_primitives__direct_openrouter__minimal__r1"
    )
    assert config.runs[-1].run_id == (
        "baseline_matrix__export_001_blend_file__plan_execute_openrouter__inspection_enabled__r3"
    )
    assert config.runs[0].output_dir == Path("artifacts/experiments/baseline_matrix") / config.runs[0].run_id
