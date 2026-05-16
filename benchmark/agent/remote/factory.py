from __future__ import annotations

from typing import Any

from benchmark.agent.errors import RemoteAgentError
from benchmark.agent.models import RemoteAgentConfig, RemoteAgentProvider
from benchmark.agent.remote.base import RemoteAgentClient
from benchmark.agent.remote.mock_remote_agent import MockRemoteAgentClient
from benchmark.agent.remote.remote_agent_client import GenericRemoteAgentClient


def create_remote_agent_client(config: RemoteAgentConfig) -> RemoteAgentClient:
    if config.provider == RemoteAgentProvider.MOCK:
        return MockRemoteAgentClient()
    if config.provider in {RemoteAgentProvider.GENERIC_HTTP, RemoteAgentProvider.GENERIC_COMMAND}:
        return GenericRemoteAgentClient(config)
    if config.provider in {RemoteAgentProvider.CODEX, RemoteAgentProvider.CLAUDE_CODE}:
        return GenericRemoteAgentClient(_reserved_provider_config(config))
    raise RemoteAgentError(
        f"Unsupported remote agent provider: {config.provider}",
        provider=str(config.provider),
        agent_id=config.agent_id,
    )


def _reserved_provider_config(config: RemoteAgentConfig) -> RemoteAgentConfig:
    transport = config.metadata.get("transport")
    if transport == RemoteAgentProvider.GENERIC_HTTP.value:
        return _copy_with_provider(
            config,
            provider=RemoteAgentProvider.GENERIC_HTTP,
            endpoint_url=_metadata_str(config.metadata, "endpoint_url") or config.endpoint_url,
        )
    if transport == RemoteAgentProvider.GENERIC_COMMAND.value:
        return _copy_with_provider(
            config,
            provider=RemoteAgentProvider.GENERIC_COMMAND,
            command=_metadata_str(config.metadata, "command") or config.command,
            args=_metadata_list(config.metadata, "args") or config.args,
        )
    raise RemoteAgentError(
        (
            f"{config.provider.value} remote agents require metadata.transport="
            "generic_http or generic_command"
        ),
        provider=config.provider.value,
        agent_id=config.agent_id,
    )


def _copy_with_provider(
    config: RemoteAgentConfig,
    *,
    provider: RemoteAgentProvider,
    endpoint_url: str | None = None,
    command: str | None = None,
    args: list[str] | None = None,
) -> RemoteAgentConfig:
    return RemoteAgentConfig(
        provider=provider,
        agent_id=config.agent_id,
        endpoint_url=endpoint_url,
        api_key_env=_metadata_str(config.metadata, "api_key_env") or config.api_key_env,
        command=command,
        args=args or [],
        workspace_dir=config.workspace_dir,
        timeout_sec=config.timeout_sec,
        metadata=config.metadata,
    )


def _metadata_str(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) else None


def _metadata_list(metadata: dict[str, Any], key: str) -> list[str] | None:
    value = metadata.get(key)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return None
