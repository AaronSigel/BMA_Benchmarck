from pathlib import Path

from benchmark.agent.llm import LlmResponse, LlmToolCall, MockLlmClient
from benchmark.agent.models import AgentConfig, AgentStrategyName, LlmConfig, RemoteAgentConfig
from benchmark.agent.remote import MockRemoteAgentClient, RemoteAgentResponse
from benchmark.agent.runtime import (
    AgentRuntime,
    create_llm_client,
    create_remote_agent_client,
    load_agent_config,
    run_task,
)
from benchmark.agent.strategies import create_agent_strategy
from benchmark.agent.strategies import DirectToolCallingStrategy, RemoteAgentStrategy
from benchmark.agent.tool_executor import MockToolExecutor
from benchmark.agent.trace import read_agent_trace


def test_runtime_load_agent_config_wrapper() -> None:
    config = load_agent_config(Path("configs/agents/mock_agent.yaml"))

    assert config.agent_id == "mock_agent"
    assert config.llm is not None
    assert config.llm.provider.value == "mock"


def test_runtime_factory_wrappers() -> None:
    assert create_llm_client(LlmConfig(provider="mock", model="mock")).__class__.__name__ == "MockLlmClient"
    assert (
        create_remote_agent_client(RemoteAgentConfig(provider="mock", agent_id="remote"))
        .__class__.__name__
        == "MockRemoteAgentClient"
    )
    assert isinstance(create_agent_strategy(AgentStrategyName.DIRECT_TOOL_CALLING), DirectToolCallingStrategy)
    assert isinstance(create_agent_strategy(AgentStrategyName.REMOTE_AGENT), RemoteAgentStrategy)


def test_run_task_mock_agent_works_without_api_and_writes_trace(tmp_path: Path) -> None:
    config = AgentConfig(
        agent_id="mock-agent",
        strategy=AgentStrategyName.DIRECT_TOOL_CALLING,
        llm=LlmConfig(provider="mock", model="mock"),
    )

    result = run_task(
        {"id": "task-1", "prompt": "Inspect scene"},
        config,
        MockToolExecutor(results={"get_scene_info": {"objects": []}}),
        tmp_path,
        llm_client=MockLlmClient(
            [LlmResponse(tool_calls=[LlmToolCall(name="get_scene_info", arguments={})])]
        ),
    )

    assert result.ok is True
    assert result.trace_path == result.artifacts_dir / "agent_trace.json"
    assert result.trace_path.exists()
    trace = read_agent_trace(result.trace_path)
    assert trace.success is True
    assert result.summary["steps_count"] == len(trace.steps)
    assert result.summary["tool_calls_count"] == 1


def test_runtime_remote_mock_agent_works_without_api_and_writes_trace(tmp_path: Path) -> None:
    config = AgentConfig(
        agent_id="mock-remote-agent",
        strategy=AgentStrategyName.REMOTE_AGENT,
        llm=None,
        remote_agent=RemoteAgentConfig(provider="mock", agent_id="remote"),
    )
    scene_snapshot_path = tmp_path / "scene_snapshot.json"

    result = run_task(
        {"id": "task-1", "prompt": "Create cube"},
        config,
        None,
        tmp_path,
        remote_agent_client=MockRemoteAgentClient(
            RemoteAgentResponse(ok=True, scene_snapshot_path=scene_snapshot_path)
        ),
    )

    assert result.ok is True
    assert result.scene_snapshot_path == scene_snapshot_path
    assert result.trace_path == result.artifacts_dir / "agent_trace.json"
    assert result.trace_path.exists()


def test_runtime_preserves_errors_in_agent_run_result(tmp_path: Path) -> None:
    config = AgentConfig(
        agent_id="mock-agent",
        strategy=AgentStrategyName.DIRECT_TOOL_CALLING,
        llm=LlmConfig(provider="mock", model="mock"),
    )

    result = AgentRuntime(
        config,
        llm_client=MockLlmClient(error="planned llm failure"),
        tool_executor=MockToolExecutor(),
    ).run(task_id="task-1", task={"prompt": "Inspect"}, artifacts_dir=tmp_path)

    assert result.ok is False
    assert result.error == "planned llm failure"
    assert result.trace_path == result.artifacts_dir / "agent_trace.json"
    assert result.trace_path.exists()
    trace = read_agent_trace(result.trace_path)
    assert trace.error == "planned llm failure"
