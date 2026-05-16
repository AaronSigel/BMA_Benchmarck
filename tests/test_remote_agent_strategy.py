from pathlib import Path

from benchmark.agent.errors import RemoteAgentError
from benchmark.agent.llm import MockLlmClient
from benchmark.agent.models import AgentConfig, AgentStrategyName, AgentTrace, RemoteAgentConfig
from benchmark.agent.remote import MockRemoteAgentClient, RemoteAgentResponse
from benchmark.agent.runtime import AgentRuntime
from benchmark.agent.strategies import RemoteAgentStrategy, create_agent_strategy
from benchmark.agent.tool_context import AgentToolContext
from benchmark.agent.tool_executor import MockToolExecutor


def make_config() -> AgentConfig:
    return AgentConfig(
        agent_id="agent-1",
        strategy=AgentStrategyName.REMOTE_AGENT,
        llm=None,
        remote_agent=RemoteAgentConfig(provider="mock", agent_id="remote"),
        mcp_profile="minimal",
    )


def test_remote_agent_strategy_works_with_mock_remote_client(tmp_path: Path) -> None:
    remote_trace = AgentTrace(
        run_id="remote-run",
        task_id="task-1",
        agent_id="remote-agent",
        strategy=AgentStrategyName.REMOTE_AGENT,
        success=True,
        final_message="done",
    )
    remote_client = MockRemoteAgentClient(
        RemoteAgentResponse(
            ok=True,
            trace=remote_trace,
            scene_snapshot_path=tmp_path / "scene_snapshot.json",
            raw_response={"ok": True},
        )
    )

    trace = RemoteAgentStrategy().run(
        {"id": "task-1", "prompt": "Create a cube"},
        make_config(),
        llm_client=None,
        tool_executor=MockToolExecutor(),
        tool_context=AgentToolContext(
            run_id="run-1",
            task_id="task-1",
            metadata={"remote_agent_client": remote_client},
        ),
        output_dir=tmp_path,
    )

    assert trace.success is True
    assert trace.run_id == "run-1"
    assert trace.agent_id == "agent-1"
    assert remote_client.requests[0].task["prompt"] == "Create a cube"
    assert remote_client.requests[0].mcp_profile == "minimal"
    assert remote_client.requests[0].tool_contracts


def test_remote_agent_strategy_does_not_use_llm_client(tmp_path: Path) -> None:
    remote_client = MockRemoteAgentClient(RemoteAgentResponse(ok=True, raw_response={"ok": True}))
    llm_client = MockLlmClient(error="LLM should not be called")

    trace = RemoteAgentStrategy().run(
        {"id": "task-1", "prompt": "Create a cube"},
        make_config(),
        llm_client=llm_client,
        tool_executor=MockToolExecutor(),
        tool_context=AgentToolContext(
            run_id="run-1",
            task_id="task-1",
            metadata={"remote_agent_client": remote_client},
        ),
        output_dir=tmp_path,
    )

    assert trace.success is True
    assert llm_client.calls == []


def test_remote_agent_error_reaches_agent_run_result(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        make_config(),
        remote_agent_client=MockRemoteAgentClient(error=RemoteAgentError("remote failed")),
    )

    result = runtime.run(
        task_id="task-1",
        task={"prompt": "Create a cube"},
        artifacts_dir=tmp_path,
    )

    assert result.ok is False
    assert result.status.value == "error"
    assert result.error == "remote failed"
    assert result.trace_path is not None


def test_runtime_selects_remote_agent_strategy_without_local_model(tmp_path: Path) -> None:
    runtime = AgentRuntime(
        make_config(),
        remote_agent_client=MockRemoteAgentClient(RemoteAgentResponse(ok=True)),
    )

    result = runtime.run(task_id="task-1", task={"prompt": "Create a cube"}, artifacts_dir=tmp_path)

    assert isinstance(create_agent_strategy(AgentStrategyName.REMOTE_AGENT), RemoteAgentStrategy)
    assert result.ok is True
    assert result.status.value == "passed"
