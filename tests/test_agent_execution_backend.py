from pathlib import Path

import yaml

from benchmark.agent.execution_backend import AgentExecutionBackend, RemoteAgentExecutionBackend
from benchmark.runner.models import ExecutionMode, RunConfig


def make_agent_config(path: Path, *, strategy: str = "direct_tool_calling") -> Path:
    if strategy == "remote_agent":
        data = {
            "agent_id": "remote-agent",
            "strategy": "remote_agent",
            "mcp_profile": "minimal",
            "remote_agent": {"provider": "mock", "agent_id": "remote"},
        }
    else:
        data = {
            "agent_id": "mock-agent",
            "strategy": "direct_tool_calling",
            "mcp_profile": "minimal",
            "llm": {"provider": "mock", "model": "mock"},
        }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def make_run_config(tmp_path: Path, agent_config_path: Path, mode: ExecutionMode) -> RunConfig:
    return RunConfig(
        run_id="run-1",
        task_id="task-1",
        execution_mode=mode,
        artifacts_dir=tmp_path / "artifacts",
        output_dir=tmp_path / "output",
        agent_config_path=agent_config_path,
        agent_output_dir=tmp_path / "agent-output",
    )


def test_runner_models_accept_agent_execution_modes(tmp_path: Path) -> None:
    config = RunConfig(
        run_id="run-1",
        task_id="task-1",
        execution_mode="agent_mcp",
        artifacts_dir=tmp_path / "artifacts",
        output_dir=tmp_path / "output",
        agent_config_path=tmp_path / "agent.yaml",
        agent_output_dir=tmp_path / "agent-output",
        mcp_config_path=Path("configs/mcp/minimal.yaml"),
    )

    assert config.execution_mode == ExecutionMode.AGENT_MCP
    assert config.agent_config_path == tmp_path / "agent.yaml"
    assert config.agent_output_dir == tmp_path / "agent-output"


def test_agent_mcp_requires_agent_config_path(tmp_path: Path) -> None:
    config = RunConfig(
        run_id="run-1",
        task_id="task-1",
        execution_mode=ExecutionMode.AGENT_MCP,
        artifacts_dir=tmp_path / "artifacts",
        output_dir=tmp_path / "output",
    )

    result = AgentExecutionBackend().execute(config)

    assert result.ok is False
    assert result.error == "agent_config_path is required"


def test_mock_agent_mcp_runs_without_blender_or_api_and_writes_trace(tmp_path: Path) -> None:
    agent_config_path = make_agent_config(tmp_path / "agent.yaml")
    config = make_run_config(tmp_path, agent_config_path, ExecutionMode.AGENT_MCP)

    result = AgentExecutionBackend().execute(config)

    assert result.ok is False
    assert result.error == "agent did not produce scene_snapshot_path"
    assert result.metadata["trace_path"] is not None
    assert Path(result.metadata["trace_path"]).exists()
    assert result.output_files == [Path(result.metadata["trace_path"])]


def test_remote_agent_backend_runs_without_api_and_writes_trace(tmp_path: Path) -> None:
    agent_config_path = make_agent_config(tmp_path / "remote-agent.yaml", strategy="remote_agent")
    config = make_run_config(tmp_path, agent_config_path, ExecutionMode.REMOTE_AGENT)

    result = RemoteAgentExecutionBackend().execute(config)

    assert result.ok is False
    assert result.error == "agent did not produce scene_snapshot_path"
    assert result.metadata["trace_path"] is not None
    assert Path(result.metadata["trace_path"]).exists()
    assert result.metadata["agent_run"]["ok"] is True


def test_agent_backend_uses_run_config_agent_config_path(tmp_path: Path) -> None:
    agent_config_path = make_agent_config(tmp_path / "agent.yaml")
    config = make_run_config(tmp_path, agent_config_path, ExecutionMode.AGENT_MCP)

    result = AgentExecutionBackend(agent_config_path=tmp_path / "ignored.yaml").execute(config)

    assert result.metadata["trace_path"] is not None
    assert Path(result.metadata["trace_path"]).exists()
