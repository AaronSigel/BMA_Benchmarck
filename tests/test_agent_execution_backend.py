import json
from pathlib import Path

import yaml

from benchmark.agent.execution_backend import (
    AgentExecutionBackend,
    RemoteAgentExecutionBackend,
    _apply_run_overrides,
    _prepare_blender_scene,
    _wrap_export_paths,
)
from benchmark.agent.config_loader import load_agent_config
from benchmark.agent.models import AgentRunResult, AgentRunStatus
from benchmark.agent.tool_executor import McpToolExecutor
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


def test_run_metadata_overrides_agent_model_and_mcp_profile(tmp_path: Path) -> None:
    agent_path = make_agent_config(tmp_path / "agent.yaml")
    run_config = make_run_config(tmp_path, agent_path, ExecutionMode.AGENT_MCP).model_copy(
        update={
            "mcp_profile": "no_python",
            "metadata": {"model_id": "qwen/qwen3-14b"},
        }
    )

    config = _apply_run_overrides(load_agent_config(agent_path), run_config)

    assert config.mcp_profile == "no_python"
    assert config.llm is not None
    assert config.llm.model == "qwen/qwen3-14b"


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
    assert Path(result.metadata["trace_path"]).is_relative_to(config.output_dir)


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


class FakeHarnessAdapter:
    def __init__(self, *, objects: list[dict] | None = None, reset_result: dict | None = None) -> None:
        self.objects = objects or []
        self.reset_result = reset_result or {"ok": True}
        self.calls: list[tuple[str, dict]] = []

    def reset_scene(self) -> dict:
        return self.reset_result

    def collect_scene_snapshot(self, output_path: Path) -> dict:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_snapshot_payload(self.objects)), encoding="utf-8")
        return {"ok": True}

    def call_tool(self, tool_name: str, arguments: dict | None = None) -> dict:
        self.calls.append((tool_name, arguments or {}))
        return {"ok": True, "arguments": arguments or {}}


def _snapshot_payload(objects: list[dict]) -> dict:
    return {
        "scene_name": "Scene",
        "objects": objects,
        "materials": [],
        "lights": [],
        "cameras": [],
        "collections": ["Collection"],
        "render_settings": {
            "engine": "CYCLES",
            "resolution_x": 1920,
            "resolution_y": 1080,
            "frame_start": 1,
            "frame_end": 1,
            "frame_current": 1,
        },
        "frame_current": 1,
        "blender_version": "4.0.0",
        "created_at": "2026-05-15T12:00:00Z",
    }


def _object_payload(name: str = "Cube") -> dict:
    vector = {"x": 0.0, "y": 0.0, "z": 0.0}
    return {
        "name": name,
        "type": "MESH",
        "primitive_hint": "cube",
        "location": vector,
        "rotation_euler": vector,
        "scale": {"x": 1.0, "y": 1.0, "z": 1.0},
        "dimensions": {"x": 2.0, "y": 2.0, "z": 2.0},
        "material_slots": [],
        "parent": None,
        "collection_names": ["Collection"],
        "vertex_count": None,
        "polygon_count": None,
    }


def test_pre_run_lifecycle_fails_when_reset_fails(tmp_path: Path) -> None:
    executor = McpToolExecutor(
        FakeHarnessAdapter(reset_result={"warning": "socket refused"}),
        profile="no_python",
    )

    error, snapshot_path = _prepare_blender_scene(executor, tmp_path, "task-1")

    assert error == "scene reset failed: socket refused"
    assert snapshot_path is None


def test_pre_run_lifecycle_fails_when_scene_remains_contaminated(tmp_path: Path) -> None:
    executor = McpToolExecutor(
        FakeHarnessAdapter(objects=[_object_payload("OldCube")]),
        profile="no_python",
    )

    error, snapshot_path = _prepare_blender_scene(executor, tmp_path, "task-1")

    assert error == "pre-run scene is not clean after reset: object_count=1"
    assert snapshot_path == tmp_path / "pre_run_scene_snapshot.json"
    assert snapshot_path.exists()


def test_export_wrapper_rewrites_relative_export_path_under_artifacts_dir(tmp_path: Path) -> None:
    adapter = FakeHarnessAdapter()
    executor = McpToolExecutor(adapter, profile="no_python")
    wrapped = _wrap_export_paths(executor, tmp_path)

    result = wrapped.call_tool("bma_export_scene", {"filepath": "exports/result.glb"})

    assert result.error is None
    assert adapter.calls == [
        ("bma_export_scene", {"filepath": str(tmp_path / "exports/result.glb")})
    ]


def test_export_wrapper_uses_task_blend_filename_at_artifact_root(tmp_path: Path) -> None:
    adapter = FakeHarnessAdapter()
    executor = McpToolExecutor(adapter, profile="no_python")
    task = {
        "expected_scene": {
            "exports": [{"format": "blend", "filename": "result.blend", "must_exist": True}]
        }
    }
    wrapped = _wrap_export_paths(executor, tmp_path, task)

    result = wrapped.call_tool("bma_export_scene", {"format": "blend", "filename": "result.blend"})

    assert result.error is None
    assert adapter.calls == [
        (
            "bma_export_scene",
            {
                "format": "blend",
                "filename": "result.blend",
                "filepath": str(tmp_path / "result.blend"),
            },
        )
    ]


def test_export_wrapper_injects_expected_glb_path_when_filepath_omitted(tmp_path: Path) -> None:
    adapter = FakeHarnessAdapter()
    executor = McpToolExecutor(adapter, profile="no_python")
    task = {
        "expected_scene": {
            "exports": [{"format": "glb", "filename": "exports/result.glb", "must_exist": True}]
        }
    }
    wrapped = _wrap_export_paths(executor, tmp_path, task)

    result = wrapped.call_tool("bma_export_scene", {"format": "glb"})

    assert result.error is None
    assert adapter.calls == [
        ("bma_export_scene", {"format": "glb", "filepath": str(tmp_path / "exports/result.glb")})
    ]


def test_agent_run_metadata_includes_auto_captured_snapshot_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    agent_config_path = make_agent_config(tmp_path / "agent.yaml")
    config = make_run_config(tmp_path, agent_config_path, ExecutionMode.AGENT_MCP)
    executor = McpToolExecutor(FakeHarnessAdapter(), profile="no_python")

    class FakeRuntime:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, *, task_id: str, task: dict, artifacts_dir: Path) -> AgentRunResult:
            return AgentRunResult(
                ok=True,
                run_id="agent-run-1",
                task_id=task_id,
                agent_id="mock-agent",
                status=AgentRunStatus.PASSED,
                scene_snapshot_path=None,
                artifacts_dir=artifacts_dir,
                summary={"execution": {"scene_snapshot_path": None}},
            )

    monkeypatch.setattr("benchmark.agent.execution_backend.AgentRuntime", FakeRuntime)

    result = AgentExecutionBackend(tool_executor=executor).execute(config)

    assert result.ok is True
    assert result.scene_snapshot_path == config.output_dir / "scene_snapshot.json"
    expected_snapshot_path = str(config.output_dir / "scene_snapshot.json")
    assert result.metadata["agent_run"]["scene_snapshot_path"] == expected_snapshot_path
    assert (
        result.metadata["agent_run"]["summary"]["execution"]["scene_snapshot_path"]
        == expected_snapshot_path
    )
