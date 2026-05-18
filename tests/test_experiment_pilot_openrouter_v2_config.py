from pathlib import Path

import yaml

from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.matrix import load_matrix
from benchmark.runner.models import ExecutionMode


def test_pilot_openrouter_v2_generates_exactly_three_no_python_runs() -> None:
    config = generate_experiment_config(load_matrix("configs/matrices/pilot_3run_openrouter_v2.yaml"))

    assert config.experiment_id == "pilot_3run_openrouter_v2"
    assert len(config.runs) == 3
    assert all(run.execution_mode is ExecutionMode.AGENT_MCP for run in config.runs)
    assert all(run.mcp_profile == "no_python" for run in config.runs)
    assert all("__no_python__" in run.run_id for run in config.runs)
    assert {run.task_id for run in config.runs} == {
        "geometry_001_basic_primitives",
        "materials_001_basic_colors",
        "export_002_glb_file",
    }


def test_pilot_openrouter_v2_uses_plan_execute_gemini_agent() -> None:
    config = generate_experiment_config(load_matrix("configs/matrices/pilot_3run_openrouter_v2.yaml"))
    agent_paths = {run.agent_config_path for run in config.runs}

    assert agent_paths == {Path("configs/agents/pilot_plan_openrouter_gemini_flash_lite.yaml")}
    agent = yaml.safe_load(Path("configs/agents/pilot_plan_openrouter_gemini_flash_lite.yaml").read_text())
    assert agent["strategy"] == "plan_and_execute"
    assert agent["mcp_profile"] == "no_python"
    assert agent["llm"]["model"] == "google/gemini-2.5-flash-lite"
    assert agent["max_steps"] == 25
    assert agent["allow_python_tools"] is False


def test_pilot_5category_openrouter_v2_generates_readiness_gate_runs() -> None:
    config = generate_experiment_config(load_matrix("configs/matrices/pilot_5category_openrouter_v2.yaml"))

    assert config.experiment_id == "pilot_5category_openrouter_v2"
    assert len(config.runs) == 5
    assert all(run.execution_mode is ExecutionMode.AGENT_MCP for run in config.runs)
    assert all(run.mcp_profile == "no_python" for run in config.runs)
    assert all(
        run.agent_config_path
        == Path("configs/agents/pilot_plan_openrouter_gemini_flash_lite.yaml")
        for run in config.runs
    )
    assert [run.task_id for run in config.runs] == [
        "geometry_001_basic_primitives",
        "materials_001_basic_colors",
        "lighting_001_area_light",
        "camera_001_front_view",
        "export_002_glb_file",
    ]
