from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.agent.models import (
    AgentConfig,
    AgentRunResult,
    AgentRunStatus,
    AgentStep,
    AgentStepType,
    AgentStrategyName,
    AgentTrace,
    LlmConfig,
    LlmProvider,
    RemoteAgentConfig,
    RemoteAgentProvider,
    ToolCallRequest,
    ToolCallResult,
    ToolCallStatus,
)
from benchmark.agent.trace import dump_agent_trace, load_agent_trace


def test_agent_model_enums_match_stage_contract() -> None:
    assert [item.value for item in AgentStrategyName] == [
        "direct_tool_calling",
        "react",
        "plan_and_execute",
        "remote_agent",
    ]
    assert [item.value for item in LlmProvider] == [
        "openrouter",
        "openai_compatible",
        "anthropic",
        "mock",
    ]
    assert [item.value for item in RemoteAgentProvider] == [
        "codex",
        "claude_code",
        "generic_http",
        "generic_command",
        "mock",
    ]
    assert [item.value for item in AgentRunStatus] == [
        "pending",
        "running",
        "passed",
        "failed",
        "error",
    ]
    assert [item.value for item in AgentStepType] == [
        "llm_call",
        "tool_call",
        "observation",
        "plan",
        "final",
        "error",
    ]


def test_llm_config_validation() -> None:
    config = LlmConfig(
        provider="mock",
        model="mock-model",
        api_key_env="BMA_TEST_API_KEY",
        extra_headers={"HTTP-Referer": "https://example.test"},
        metadata={"purpose": "unit-test"},
    )
    assert config.provider == LlmProvider.MOCK
    assert config.model == "mock-model"
    assert config.base_url is None
    assert config.api_key_env == "BMA_TEST_API_KEY"
    assert config.temperature == 0.2
    assert config.top_p == 0.9
    assert config.max_tokens == 2048
    assert config.timeout_sec == 120
    assert config.extra_headers == {"HTTP-Referer": "https://example.test"}
    assert config.metadata == {"purpose": "unit-test"}

    with pytest.raises(ValidationError):
        LlmConfig(provider="ollama", model="llama")
    with pytest.raises(ValidationError):
        LlmConfig(model="")
    with pytest.raises(ValidationError):
        LlmConfig(model="mock", temperature=-0.1)
    with pytest.raises(ValidationError):
        LlmConfig(model="mock", temperature=2.1)
    with pytest.raises(ValidationError):
        LlmConfig(model="mock", top_p=-0.1)
    with pytest.raises(ValidationError):
        LlmConfig(model="mock", top_p=1.1)
    with pytest.raises(ValidationError):
        LlmConfig(model="mock", max_tokens=0)
    with pytest.raises(ValidationError):
        LlmConfig(model="mock", api_key="secret")


def test_remote_agent_config_validation() -> None:
    default_config = RemoteAgentConfig(
        provider=RemoteAgentProvider.MOCK,
        agent_id="remote",
        api_key_env="BMA_REMOTE_AGENT_KEY",
        args=["--json"],
        workspace_dir=Path("/tmp/workspace"),
        metadata={"purpose": "unit-test"},
    )
    assert default_config.endpoint_url is None
    assert default_config.command is None
    assert default_config.timeout_sec == 300
    assert default_config.api_key_env == "BMA_REMOTE_AGENT_KEY"
    assert default_config.args == ["--json"]
    assert default_config.workspace_dir == Path("/tmp/workspace")
    assert default_config.metadata == {"purpose": "unit-test"}

    http_config = RemoteAgentConfig(
        provider=RemoteAgentProvider.GENERIC_HTTP,
        agent_id="remote",
        endpoint_url="https://example.test/agent",
    )
    assert http_config.endpoint_url == "https://example.test/agent"

    command_config = RemoteAgentConfig(
        provider=RemoteAgentProvider.GENERIC_COMMAND,
        agent_id="remote",
        command="agent-wrapper",
    )
    assert command_config.command == "agent-wrapper"

    with pytest.raises(ValidationError):
        RemoteAgentConfig(provider=RemoteAgentProvider.GENERIC_HTTP, agent_id="remote")
    with pytest.raises(ValidationError):
        RemoteAgentConfig(provider=RemoteAgentProvider.GENERIC_COMMAND, agent_id="remote")
    with pytest.raises(ValidationError):
        RemoteAgentConfig(provider="mock", agent_id="remote", api_key="secret")
    with pytest.raises(ValidationError):
        RemoteAgentConfig(provider="mock", agent_id="remote", model="local-model")
    with pytest.raises(ValidationError):
        RemoteAgentConfig(provider="mock", agent_id="")
    with pytest.raises(ValidationError):
        RemoteAgentConfig(provider="mock", agent_id="remote", timeout_sec=0)


def test_agent_config_strategy_validation() -> None:
    llm_config = AgentConfig(
        agent_id="agent-1",
        strategy=AgentStrategyName.REACT,
        mcp_profile="minimal",
        llm=LlmConfig(provider=LlmProvider.MOCK, model="mock-model"),
        max_steps=20,
        max_retries=1,
        step_timeout_sec=120,
        allow_python_tools=False,
        allow_inspection_tools=True,
        system_prompt_template="Solve the task.",
        metadata={"purpose": "unit-test"},
    )
    assert llm_config.llm is not None
    assert llm_config.mcp_profile == "minimal"
    assert llm_config.max_steps == 20
    assert llm_config.max_retries == 1
    assert llm_config.step_timeout_sec == 120
    assert llm_config.allow_python_tools is False
    assert llm_config.allow_inspection_tools is True

    remote_config = AgentConfig(
        agent_id="agent-2",
        strategy=AgentStrategyName.REMOTE_AGENT,
        llm=None,
        remote_agent=RemoteAgentConfig(agent_id="remote"),
    )
    assert remote_config.remote_agent is not None

    with pytest.raises(ValidationError):
        AgentConfig(agent_id="agent-0")
    with pytest.raises(ValidationError):
        AgentConfig(agent_id="agent-3", strategy=AgentStrategyName.REACT, llm=None)
    with pytest.raises(ValidationError):
        AgentConfig(agent_id="agent-4", strategy=AgentStrategyName.REMOTE_AGENT, llm=None)
    with pytest.raises(ValidationError):
        AgentConfig(
            agent_id="agent-5",
            strategy=AgentStrategyName.REACT,
            llm=LlmConfig(),
            mcp_profile="",
        )
    with pytest.raises(ValidationError):
        AgentConfig(
            agent_id="agent-6",
            strategy=AgentStrategyName.REACT,
            llm=LlmConfig(),
            max_steps=0,
        )
    with pytest.raises(ValidationError):
        AgentConfig(
            agent_id="agent-7",
            strategy=AgentStrategyName.REACT,
            llm=LlmConfig(),
            allowed_tools=["execute_blender_code"],
        )
    with pytest.raises(ValidationError):
        AgentConfig(
            agent_id="agent-8",
            strategy=AgentStrategyName.REACT,
            llm=LlmConfig(),
            system_prompt_template="Call execute_blender_code when needed.",
        )
    python_tool_config = AgentConfig(
        agent_id="agent-9",
        strategy=AgentStrategyName.REACT,
        llm=LlmConfig(),
        allow_python_tools=True,
        allowed_tools=["execute_blender_code"],
    )
    assert python_tool_config.allow_python_tools is True
    with pytest.raises(ValidationError):
        AgentConfig(
            agent_id="agent-10",
            strategy=AgentStrategyName.REACT,
            llm=LlmConfig(),
            local_model="local-model",
        )


def test_tool_call_models() -> None:
    request = ToolCallRequest(name="get_scene_info", arguments={"include": "objects"})
    result = ToolCallResult(name=request.name, status=ToolCallStatus.SUCCEEDED, result={})

    assert request.name == "get_scene_info"
    assert result.status == ToolCallStatus.SUCCEEDED


def test_agent_trace_sorts_steps_and_round_trips_json(tmp_path: Path) -> None:
    trace = AgentTrace(
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        strategy=AgentStrategyName.DIRECT_TOOL_CALLING,
        model="mock",
        steps=[
            AgentStep(
                step_index=2,
                step_type=AgentStepType.FINAL,
                thought="done",
                action="finish",
                observation="done",
                duration_sec=None,
            ),
            AgentStep(
                step_index=1,
                step_type=AgentStepType.TOOL_CALL,
                thought="inspect scene",
                action="call_tool",
                tool_name="get_scene_info",
                tool_arguments={"include": "objects"},
                observation={"objects": []},
                raw_llm_response={"tool_call": "get_scene_info"},
                duration_sec=0.1,
            ),
        ],
        final_message="done",
        success=True,
        duration_sec=0.2,
        metadata={"purpose": "unit-test"},
    )

    assert [step.step_index for step in trace.steps] == [1, 2]
    assert trace.steps[0].tool_arguments == {"include": "objects"}

    parsed = AgentTrace.model_validate_json(trace.model_dump_json())
    assert parsed == trace
    trace_path = tmp_path / "agent_trace.json"
    dump_agent_trace(trace, trace_path)
    loaded = load_agent_trace(trace_path)

    assert loaded == trace


def test_agent_step_trace_result_reject_negative_duration() -> None:
    with pytest.raises(ValidationError):
        AgentStep(step_index=0, step_type=AgentStepType.LLM_CALL, duration_sec=-0.1)
    with pytest.raises(ValidationError):
        AgentTrace(
            run_id="run-1",
            task_id="task-1",
            agent_id="agent-1",
            strategy=AgentStrategyName.REACT,
            duration_sec=-0.1,
        )
    with pytest.raises(ValidationError):
        AgentRunResult(
            ok=False,
            run_id="run-1",
            task_id="task-1",
            agent_id="agent-1",
            status=AgentRunStatus.ERROR,
            duration_sec=-0.1,
        )


def test_agent_run_result_model() -> None:
    result = AgentRunResult(
        ok=True,
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        trace_path=Path("trace.json"),
        scene_snapshot_path=Path("scene_snapshot.json"),
        artifacts_dir=Path("artifacts/run-1"),
        status=AgentRunStatus.PASSED,
        summary={"score": 1.0},
        duration_sec=None,
    )

    assert result.ok is True
    assert result.trace_path == Path("trace.json")
    assert result.scene_snapshot_path == Path("scene_snapshot.json")
    assert result.artifacts_dir == Path("artifacts/run-1")
    assert result.summary["score"] == 1.0
