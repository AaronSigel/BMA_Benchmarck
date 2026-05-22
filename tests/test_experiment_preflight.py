from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.agent.errors import LlmClientError
from benchmark.agent.llm.base import LlmResponse, LlmToolCall
from benchmark.experiments import preflight
from benchmark.mcp.config import McpServerConfig
from benchmark.runner.models import ExecutionMode, ExperimentConfig, RunConfig

_OPENROUTER_AGENT = Path("configs/agents/direct_openrouter.yaml")
_MODELS = [
    "google/gemini-2.5-flash-lite",
    "openai/gpt-5-mini",
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen3-coder",
    "mistralai/mistral-small-3.2-24b-instruct",
]


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
            name = (params or {}).get("name", "ProbeCube")
            dims = (params or {}).get("dimensions", [2.0, 2.0, 2.0])
            return {
                "ok": True,
                "name": name,
                "dimensions": dims,
                "primitive_hint": "cube",
            }
        if tool_name == "bma_set_material":
            return {"ok": True, "object": "ProbeCube", "material": "ProbeRed"}
        if tool_name == "bma_export_scene":
            filepath = Path((params or {})["filepath"])
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(b"export")
            return {
                "ok": True,
                "result": {
                    "ok": True,
                    "format": (params or {}).get("format"),
                    "filepath": str(filepath),
                    "exists": True,
                    "file_size_bytes": filepath.stat().st_size,
                },
                "error": None,
            }
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
    assert len(result["export_results"]) == 2


def test_preflight_fails_when_export_response_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyExportAdapter(FakeAdapter):
        def call_tool(self, tool_name, params=None):
            if tool_name == "bma_export_scene":
                return {}
            return super().call_tool(tool_name, params)

    _patch_smoke(monkeypatch, EmptyExportAdapter)

    result = preflight.run_contract_smoke_for_experiment(_agent_config(tmp_path), tmp_path)

    assert result["ok"] is False
    assert "bma_export_scene" in result["error"]


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


def _openrouter_experiment_config(tmp_path: Path, model_ids: list[str]) -> ExperimentConfig:
    runs = [
        RunConfig(
            run_id=f"r{index}",
            task_id="geometry_001_basic_primitives",
            execution_mode=ExecutionMode.AGENT_MCP,
            artifacts_dir=tmp_path / f"r{index}",
            output_dir=tmp_path / f"r{index}",
            mcp_config_path=tmp_path / "mcp.yaml",
            mcp_profile="minimal",
            agent_config_path=_OPENROUTER_AGENT,
            metadata={"model_id": model_id},
        )
        for index, model_id in enumerate(model_ids)
    ]
    return ExperimentConfig(experiment_id="openrouter-smoke", runs=runs)


class _FakeLlmClient:
    def __init__(self, config, responses: dict[str, LlmResponse | Exception]) -> None:
        self.config = config
        self.responses = responses

    def complete(self, messages, tools=None, timeout_sec=None) -> LlmResponse:
        response = self.responses.get(self.config.model)
        if response is None:
            raise LlmClientError(f"model unavailable: {self.config.model}")
        if isinstance(response, Exception):
            raise response
        return response


def _patch_openrouter_llm_client(
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[str, LlmResponse | Exception],
) -> None:
    def factory(config):
        return _FakeLlmClient(config, responses)

    monkeypatch.setattr("benchmark.agent.llm.factory.create_llm_client", factory)


def _tool_call_response() -> LlmResponse:
    return LlmResponse(
        tool_calls=[
            LlmToolCall(id="tc-1", name="bma_get_scene_snapshot", arguments={}),
        ],
        finish_reason="tool_calls",
    )


def test_openrouter_model_smoke_success_with_tool_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    _patch_openrouter_llm_client(monkeypatch, {model_id: _tool_call_response() for model_id in _MODELS})

    result = preflight.run_openrouter_model_smoke_for_experiment(
        _openrouter_experiment_config(tmp_path, _MODELS),
        tmp_path,
        require_tool_call=True,
        smoke_cfg={"mcp_profile": "minimal", "probe_tool": "bma_get_scene_snapshot"},
    )

    assert result["ok"] is True
    assert len(result["models"]) == 5
    assert all(item["ok"] for item in result["models"])
    assert all(item["tool_call_count"] == 1 for item in result["models"])
    assert (tmp_path / "preflight_model_smoke.json").is_file()


def test_openrouter_model_smoke_fails_without_tool_calls_when_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    _patch_openrouter_llm_client(
        monkeypatch,
        {model_id: LlmResponse(content="no tools here", finish_reason="stop") for model_id in _MODELS},
    )

    result = preflight.run_openrouter_model_smoke_for_experiment(
        _openrouter_experiment_config(tmp_path, _MODELS[:1]),
        tmp_path,
        require_tool_call=True,
    )

    assert result["ok"] is False
    assert result["models"][0]["error"] == "model returned no tool calls or JSON action"


def test_openrouter_model_smoke_fails_on_llm_client_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    _patch_openrouter_llm_client(
        monkeypatch,
        {
            _MODELS[0]: _tool_call_response(),
            _MODELS[1]: LlmClientError("OpenRouter HTTP 404: model not found"),
        },
    )

    result = preflight.run_openrouter_model_smoke_for_experiment(
        _openrouter_experiment_config(tmp_path, _MODELS[:2]),
        tmp_path,
        require_tool_call=True,
    )

    assert result["ok"] is False
    assert result["models"][0]["ok"] is True
    assert result["models"][1]["ok"] is False
    assert "404" in result["models"][1]["error"]


def test_openrouter_model_smoke_fails_without_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    result = preflight.run_openrouter_model_smoke_for_experiment(
        _openrouter_experiment_config(tmp_path, _MODELS[:1]),
        tmp_path,
        require_tool_call=False,
    )

    assert result["ok"] is False
    assert result["api_available"] is False
    assert "OPENROUTER_API_KEY" in result["error"]


def test_run_matrix_required_preflight_uses_openrouter_model_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    _patch_openrouter_llm_client(monkeypatch, {model_id: _tool_call_response() for model_id in _MODELS[:1]})
    monkeypatch.setattr(
        preflight,
        "run_profile_preflight_for_experiment",
        lambda config: {"ok": True, "profiles": []},
    )
    monkeypatch.setattr(
        preflight,
        "run_contract_smoke_for_experiment",
        lambda config, output_root: {"ok": True},
    )

    result = preflight.run_matrix_required_preflight(
        _openrouter_experiment_config(tmp_path, _MODELS[:1]),
        tmp_path,
        preflight_cfg={
            "enabled": True,
            "require_model_access_smoke": True,
            "require_tool_calling_smoke": True,
            "model_smoke": {"mcp_profile": "minimal"},
        },
    )

    smoke = result["checks"]["openrouter_model_smoke"]
    assert smoke["ok"] is True
    assert result["ok"] is True
