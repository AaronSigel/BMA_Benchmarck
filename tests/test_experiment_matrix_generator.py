from pathlib import Path

import yaml

from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.models import ExperimentMatrix
from benchmark.runner.models import ExecutionMode
from benchmark.tasks.loader import dump_task
from benchmark.tasks.models import (
    BenchmarkTask,
    DifficultyLevel,
    ExpectedScene,
    SuccessCriterion,
    TaskCategory,
)


def make_task(task_id: str) -> BenchmarkTask:
    return BenchmarkTask(
        id=task_id,
        title=f"Task {task_id}",
        category=TaskCategory.GEOMETRY,
        difficulty=DifficultyLevel.EASY,
        prompt=f"Run benchmark task {task_id}.",
        tags=["geometry"],
        allowed_tools=[],
        expected_scene=ExpectedScene(),
        success_criteria=[SuccessCriterion(metric="complete", weight=1.0)],
    )


def write_agent_config(directory: Path, agent_id: str, strategy: str = "direct_tool_calling") -> Path:
    path = directory / f"{agent_id}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "agent_id": agent_id,
                "strategy": strategy,
                "mcp_profile": "minimal",
                "llm": {"provider": "mock", "model": "mock"},
            }
        ),
        encoding="utf-8",
    )
    return path


def write_mcp_config(directory: Path, profile: str) -> Path:
    path = directory / f"{profile}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "profile": profile,
                "server_distribution": "upstream",
                "command": "uvx",
                "args": ["blender-mcp"],
            }
        ),
        encoding="utf-8",
    )
    return path


def make_matrix(tmp_path: Path, repetitions: int = 1) -> ExperimentMatrix:
    tasks_root = tmp_path / "tasks"
    agents_root = tmp_path / "agents"
    mcp_root = tmp_path / "mcp"
    tasks_root.mkdir()
    agents_root.mkdir()
    mcp_root.mkdir()

    dump_task(make_task("geometry_001_basic_primitives"), tasks_root / "geometry_001.yaml")
    write_agent_config(agents_root, "mock_agent")
    write_mcp_config(mcp_root, "minimal")

    return ExperimentMatrix(
        matrix_id="smoke_matrix",
        tasks={"ids": ["geometry_001_basic_primitives"]},
        agents={"ids": ["mock_agent"], "config_root": agents_root},
        mcp_profiles=["minimal"],
        models={"ids": ["mock"]},
        execution_modes=[ExecutionMode.EXTERNAL_SNAPSHOT],
        repetitions=repetitions,
        output_root=tmp_path / "outputs",
        metadata={
            "tasks_root": str(tasks_root),
            "mcp_config_root": str(mcp_root),
            "snapshot_path": str(tmp_path / "scene_snapshot.json"),
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
    )


def test_generate_experiment_config_uses_stable_run_id_format(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path)

    config = generate_experiment_config(matrix)

    assert config.experiment_id == "smoke_matrix"
    assert [run.run_id for run in config.runs] == [
        "smoke_matrix__geometry_001_basic_primitives__mock_agent__minimal__r1"
    ]


def test_generate_experiment_config_populates_runner_paths(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path)

    run = generate_experiment_config(matrix).runs[0]

    assert run.task_path == tmp_path / "tasks" / "geometry_001.yaml"
    assert run.agent_config_path == tmp_path / "agents" / "mock_agent.yaml"
    assert run.mcp_config_path == tmp_path / "mcp" / "minimal.yaml"
    assert run.snapshot_path == tmp_path / "scene_snapshot.json"
    assert run.artifacts_dir == tmp_path / "artifacts"
    assert run.output_dir == tmp_path / "outputs" / run.run_id
    assert run.agent_output_dir == run.output_dir / "agent"


def test_repetitions_create_multiple_runs_for_same_combination(tmp_path: Path) -> None:
    config = generate_experiment_config(make_matrix(tmp_path, repetitions=3))

    assert [run.run_id for run in config.runs] == [
        "smoke_matrix__geometry_001_basic_primitives__mock_agent__minimal__r1",
        "smoke_matrix__geometry_001_basic_primitives__mock_agent__minimal__r2",
        "smoke_matrix__geometry_001_basic_primitives__mock_agent__minimal__r3",
    ]
    assert [run.metadata["repetition"] for run in config.runs] == [1, 2, 3]


def test_same_matrix_generates_same_experiment_config(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path, repetitions=2)

    first = generate_experiment_config(matrix)
    second = generate_experiment_config(matrix)

    assert first == second


def test_output_dir_is_unique_for_each_run(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path, repetitions=2)
    matrix.models.ids.append("mock/variant")

    config = generate_experiment_config(matrix)
    output_dirs = [run.output_dir for run in config.runs]

    assert len(output_dirs) == len(set(output_dirs))
    assert [run.metadata["model_id"] for run in config.runs] == [
        "mock",
        "mock",
        "mock/variant",
        "mock/variant",
    ]
    assert all("mock-variant" in run.run_id or run.metadata["model_id"] == "mock" for run in config.runs)


def test_output_dir_is_unique_when_multiple_execution_modes(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path)
    matrix.execution_modes = [ExecutionMode.EXTERNAL_SNAPSHOT, ExecutionMode.REPLAY]

    config = generate_experiment_config(matrix)

    assert [run.execution_mode for run in config.runs] == [
        ExecutionMode.EXTERNAL_SNAPSHOT,
        ExecutionMode.REPLAY,
    ]
    assert len({run.run_id for run in config.runs}) == 2
    assert len({run.output_dir for run in config.runs}) == 2
    assert config.runs[1].run_id.endswith("__replay")


def test_generator_expands_task_agent_mcp_model_repetition_matrix(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path, repetitions=2)
    dump_task(make_task("geometry_002_positions"), tmp_path / "tasks" / "geometry_002.yaml")
    write_agent_config(tmp_path / "agents", "react_agent", strategy="react")
    write_mcp_config(tmp_path / "mcp", "no_python")
    matrix.tasks.ids.append("geometry_002_positions")
    matrix.agents.ids.append("react_agent")
    matrix.mcp_profiles.append("no_python")
    matrix.models.ids.append("mock-2")

    config = generate_experiment_config(matrix)

    assert len(config.runs) == 2 * 2 * 2 * 2 * 2
    assert len({run.run_id for run in config.runs}) == len(config.runs)
