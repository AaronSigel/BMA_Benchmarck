class AgentError(Exception):
    """Base exception for agent runtime errors."""


class AgentConfigError(AgentError):
    """Raised when an agent configuration cannot be loaded or validated."""


class AgentRuntimeError(AgentError):
    """Raised when agent execution fails before producing a result."""


class AgentTimeoutError(AgentRuntimeError):
    """Raised when agent execution exceeds its timeout."""


class AgentStepLimitError(AgentRuntimeError):
    """Raised when agent execution exceeds its configured step limit."""


class AgentTraceError(AgentError):
    """Raised when an agent trace cannot be loaded or written."""


class LlmClientError(AgentError):
    """Raised when an LLM provider request fails."""


class LlmResponseParseError(LlmClientError):
    """Raised when an LLM response cannot be parsed into the expected structure."""

    def __init__(
        self,
        message: str,
        *,
        fragment: str | None = None,
        raw_response: object | None = None,
    ) -> None:
        super().__init__(message)
        self.fragment = fragment
        self.raw_response = raw_response


class RemoteAgentError(AgentError):
    """Raised when a remote server-side agent fails."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.agent_id = agent_id


class RemoteAgentTimeoutError(RemoteAgentError):
    """Raised when a remote server-side agent request times out."""


class ToolInvocationError(AgentError):
    """Raised when a tool call cannot be executed."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        details: object | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.details = details


class ToolSchemaError(AgentError):
    """Raised when tool schemas are invalid or unavailable."""


class UnsupportedAgentStrategyError(AgentConfigError):
    """Raised when an agent strategy is not supported by the runtime."""


class UnsupportedLlmProviderError(AgentConfigError):
    """Raised when an LLM provider is not supported by the runtime."""


# Backward-compatible alias for the initial stage-6.1 scaffold.
AgentToolError = ToolInvocationError
