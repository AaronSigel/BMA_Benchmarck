import pytest

from benchmark.agent.errors import RemoteAgentError
from benchmark.agent.models import RemoteAgentConfig, RemoteAgentProvider
from benchmark.agent.remote import (
    GenericRemoteAgentClient,
    MockRemoteAgentClient,
    create_remote_agent_client,
)


def test_create_remote_agent_client_returns_mock() -> None:
    client = create_remote_agent_client(RemoteAgentConfig(provider=RemoteAgentProvider.MOCK))

    assert isinstance(client, MockRemoteAgentClient)


def test_create_remote_agent_client_returns_generic_http_client() -> None:
    config = RemoteAgentConfig(
        provider=RemoteAgentProvider.GENERIC_HTTP,
        agent_id="http-agent",
        endpoint_url="https://remote.example/run",
    )

    client = create_remote_agent_client(config)

    assert isinstance(client, GenericRemoteAgentClient)
    assert client.config == config


def test_create_remote_agent_client_returns_generic_command_client() -> None:
    config = RemoteAgentConfig(
        provider=RemoteAgentProvider.GENERIC_COMMAND,
        agent_id="command-agent",
        command="remote-agent",
    )

    client = create_remote_agent_client(config)

    assert isinstance(client, GenericRemoteAgentClient)
    assert client.config == config


def test_codex_can_map_to_generic_http_via_metadata() -> None:
    client = create_remote_agent_client(
        RemoteAgentConfig(
            provider=RemoteAgentProvider.CODEX,
            agent_id="codex",
            metadata={
                "transport": "generic_http",
                "endpoint_url": "https://remote.example/codex",
                "api_key_env": "CODEX_API_KEY",
            },
        )
    )

    assert isinstance(client, GenericRemoteAgentClient)
    assert client.config.provider == RemoteAgentProvider.GENERIC_HTTP
    assert client.config.endpoint_url == "https://remote.example/codex"
    assert client.config.api_key_env == "CODEX_API_KEY"


def test_claude_code_can_map_to_generic_command_via_metadata() -> None:
    client = create_remote_agent_client(
        RemoteAgentConfig(
            provider=RemoteAgentProvider.CLAUDE_CODE,
            agent_id="claude-code",
            metadata={
                "transport": "generic_command",
                "command": "claude",
                "args": ["--print", "--json"],
            },
        )
    )

    assert isinstance(client, GenericRemoteAgentClient)
    assert client.config.provider == RemoteAgentProvider.GENERIC_COMMAND
    assert client.config.command == "claude"
    assert client.config.args == ["--print", "--json"]


def test_reserved_provider_requires_transport_metadata() -> None:
    with pytest.raises(RemoteAgentError, match="metadata.transport"):
        create_remote_agent_client(
            RemoteAgentConfig(provider=RemoteAgentProvider.CODEX, agent_id="codex")
        )


def test_unknown_remote_agent_provider_raises_error() -> None:
    config = RemoteAgentConfig(provider=RemoteAgentProvider.MOCK).model_copy(
        update={"provider": "unknown"}
    )

    with pytest.raises(RemoteAgentError, match="Unsupported remote agent provider"):
        create_remote_agent_client(config)
