from benchmark.agent.errors import (
    AgentConfigError,
    AgentError,
    AgentRuntimeError,
    AgentStepLimitError,
    AgentTimeoutError,
    AgentTraceError,
    LlmClientError,
    LlmResponseParseError,
    RemoteAgentError,
    RemoteAgentTimeoutError,
    ToolInvocationError,
    ToolSchemaError,
    UnsupportedAgentStrategyError,
    UnsupportedLlmProviderError,
)


def test_agent_errors_are_importable() -> None:
    assert issubclass(AgentConfigError, AgentError)
    assert issubclass(AgentRuntimeError, AgentError)
    assert issubclass(AgentTimeoutError, AgentRuntimeError)
    assert issubclass(AgentStepLimitError, AgentRuntimeError)
    assert issubclass(AgentTraceError, AgentError)
    assert issubclass(LlmClientError, AgentError)
    assert issubclass(LlmResponseParseError, LlmClientError)
    assert issubclass(RemoteAgentError, AgentError)
    assert issubclass(RemoteAgentTimeoutError, RemoteAgentError)
    assert issubclass(ToolInvocationError, AgentError)
    assert issubclass(ToolSchemaError, AgentError)
    assert issubclass(UnsupportedAgentStrategyError, AgentConfigError)
    assert issubclass(UnsupportedLlmProviderError, AgentConfigError)


def test_tool_invocation_error_contains_tool_name() -> None:
    error = ToolInvocationError("tool failed", tool_name="get_scene_info")

    assert str(error) == "tool failed"
    assert error.tool_name == "get_scene_info"


def test_llm_response_parse_error_contains_fragment_and_raw_response() -> None:
    raw_response = {"choices": []}
    error = LlmResponseParseError(
        "parse failed",
        fragment="{bad json",
        raw_response=raw_response,
    )

    assert error.fragment == "{bad json"
    assert error.raw_response == raw_response


def test_remote_agent_error_contains_provider_and_agent_id() -> None:
    error = RemoteAgentError("remote failed", provider="codex", agent_id="agent-1")

    assert error.provider == "codex"
    assert error.agent_id == "agent-1"
