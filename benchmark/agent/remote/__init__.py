"""Provider-neutral remote agent abstractions."""

from benchmark.agent.remote.base import (
    RemoteAgentArtifact,
    RemoteAgentClient,
    RemoteAgentRequest,
    RemoteAgentResponse,
)
from benchmark.agent.remote.factory import create_remote_agent_client
from benchmark.agent.remote.mock_remote_agent import MockRemoteAgentClient
from benchmark.agent.remote.remote_agent_client import GenericRemoteAgentClient

__all__ = [
    "GenericRemoteAgentClient",
    "MockRemoteAgentClient",
    "RemoteAgentArtifact",
    "RemoteAgentClient",
    "RemoteAgentRequest",
    "RemoteAgentResponse",
    "create_remote_agent_client",
]
