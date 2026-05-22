from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.matrix import load_matrix
from benchmark.runner.models import ExecutionMode

_FINAL_MODELS = {
    "google/gemini-2.5-flash-lite",
    "openai/gpt-5-mini",
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen3-coder",
    "mistralai/mistral-small-3.2-24b-instruct",
}


def test_model_access_smoke_generates_five_runs() -> None:
    matrix = load_matrix("configs/matrices/model_access_smoke_openrouter_v1.yaml")
    config = generate_experiment_config(matrix)
    assert len(config.runs) == 5
    assert config.experiment_id == "model_access_smoke_openrouter_v1"
    assert all(run.execution_mode is ExecutionMode.AGENT_MCP for run in config.runs)
    assert {run.metadata["model_id"] for run in config.runs} == _FINAL_MODELS
    assert all(run.metadata["agent_id"] == "direct_openrouter" for run in config.runs)
    assert all(run.mcp_profile == "minimal" for run in config.runs)


def test_model_access_smoke_matches_final_generation_profile() -> None:
    final = load_matrix("configs/matrices/final_multimodel_openrouter_v1.yaml")
    smoke = load_matrix("configs/matrices/model_access_smoke_openrouter_v1.yaml")
    final_gp = final.metadata["generation_profile"]
    smoke_gp = smoke.metadata["generation_profile"]
    assert smoke_gp["temperature"] == final_gp["temperature"]
    assert smoke_gp["top_p"] == final_gp["top_p"]
    assert smoke_gp["max_tokens"] == final_gp["max_tokens"]
    assert smoke_gp["reasoning"]["enabled"] is False
