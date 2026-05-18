from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.experiments import preflight
from benchmark.mcp.config import McpServerConfig
from benchmark.runner.models import ExecutionMode, ExperimentConfig, RunConfig


def _agent_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="preflight",
        runs=[
            RunConfig(
                run_id="r1",
                task_id="t1",
                execution_mode=ExecutionMode.AGENT_MCP,
                artifacts_dir=tmp_path / "r1",
                output_dir=tmp_path / "r1",
                mcp_config_path=tmp_path / "mcp.yaml",
                mcp_profile="no_python",
            )
        ],
    )


class FakeAdapter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset_count = 0

    def reset_scene(self):
        self.reset_count += 1
        return {"ok": True}

    def call_tool(self, tool_name, params=None):
        if tool_name == "bma_create_object":
            return {
                "name": "ProbeCube",
                "dimensions": [2.0, 2.0, 2.0],
                "primitive_hint": "cube",
            }
        if tool_name == "bma_set_material":
            return {"object": "ProbeCube", "material": "ProbeRed"}
        return {}

    def collect_scene_snapshot(self, output_path):
        Path(output_path).write_text(
            """
{
  "objects": [
    {
      "name": "ProbeCube",
      "dimensions": {"x": 2.0, "y": 2.0, "z": 2.0},
      "primitive_hint": "cube",
      "material_slots": ["ProbeRed"]
    }
  ],
  "materials": [{"name": "ProbeRed"}]
}
""",
            encoding="utf-8",
        )
        return {"ok": True}


def _patch_smoke(monkeypatch: pytest.MonkeyPatch, adapter_cls=FakeAdapter) -> None:
    monkeypatch.setattr(
        preflight,
        "load_mcp_config",
        lambda path: McpServerConfig(profile="no_python", blender_host="localhost", blender_port=9876),
    )
    monkeypatch.setattr(preflight, "ExternalBlenderMcpServerAdapter", adapter_cls)


def test_prepare_output_root_requires_clean_for_existing_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "old.json").write_text("{}", encoding="utf-8")

    with pytest.raises(preflight.PreflightError, match="--clean-output"):
        preflight.prepare_output_root(output_root, clean_output=False)


def test_prepare_output_root_clean_removes_existing_artifacts(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "old.json").write_text("{}", encoding="utf-8")

    meta = preflight.prepare_output_root(output_root, clean_output=True)

    assert output_root.is_dir()
    assert not (output_root / "old.json").exists()
    assert meta["removed_existing_output"] is True


def test_contract_smoke_success_writes_snapshot_and_resets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_smoke(monkeypatch)

    result = preflight.run_contract_smoke_for_experiment(_agent_config(tmp_path), tmp_path)

    assert result["ok"] is True
    assert (tmp_path / "preflight_contract_smoke_snapshot.json").is_file()


def test_contract_smoke_fails_when_dimensions_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class BadDimensionsAdapter(FakeAdapter):
        def collect_scene_snapshot(self, output_path):
            Path(output_path).write_text(
                '{"objects":[{"name":"ProbeCube","dimensions":{"x":1,"y":1,"z":1},'
                '"primitive_hint":"cube","material_slots":["ProbeRed"]}],"materials":[{"name":"ProbeRed"}]}',
                encoding="utf-8",
            )
            return {"ok": True}

    _patch_smoke(monkeypatch, BadDimensionsAdapter)

    with pytest.raises(preflight.PreflightError, match="expected BMA contract"):
        preflight.run_contract_smoke_for_experiment(_agent_config(tmp_path), tmp_path)


def test_contract_smoke_fails_when_material_name_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class BadMaterialAdapter(FakeAdapter):
        def collect_scene_snapshot(self, output_path):
            Path(output_path).write_text(
                '{"objects":[{"name":"ProbeCube","dimensions":{"x":2,"y":2,"z":2},'
                '"primitive_hint":"cube","material_slots":["BMA_ProbeCube_mat"]}],"materials":[{"name":"BMA_ProbeCube_mat"}]}',
                encoding="utf-8",
            )
            return {"ok": True}

    _patch_smoke(monkeypatch, BadMaterialAdapter)

    with pytest.raises(preflight.PreflightError, match="expected BMA contract"):
        preflight.run_contract_smoke_for_experiment(_agent_config(tmp_path), tmp_path)


def test_contract_smoke_fails_when_primitive_hint_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class BadHintAdapter(FakeAdapter):
        def collect_scene_snapshot(self, output_path):
            Path(output_path).write_text(
                '{"objects":[{"name":"ProbeCube","dimensions":{"x":2,"y":2,"z":2},'
                '"primitive_hint":null,"material_slots":["ProbeRed"]}],"materials":[{"name":"ProbeRed"}]}',
                encoding="utf-8",
            )
            return {"ok": True}

    _patch_smoke(monkeypatch, BadHintAdapter)

    with pytest.raises(preflight.PreflightError, match="expected BMA contract"):
        preflight.run_contract_smoke_for_experiment(_agent_config(tmp_path), tmp_path)
