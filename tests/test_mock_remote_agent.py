from pathlib import Path

import pytest

from benchmark.agent.errors import RemoteAgentError, RemoteAgentTimeoutError
from benchmark.agent.remote import (
    MockRemoteAgentClient,
    RemoteAgentArtifact,
    RemoteAgentClient,
    RemoteAgentRequest,
    RemoteAgentResponse,
)


def test_mock_remote_agent_returns_configured_response(tmp_path: Path) -> None:
    response = RemoteAgentResponse(
        ok=True,
        scene_snapshot_path=tmp_path / "scene_snapshot.json",
        artifacts=[
            RemoteAgentArtifact(
                name="trace",
                path=tmp_path / "agent_trace.json",
                kind="json",
            )
        ],
    )
    client = MockRemoteAgentClient(response)
    request = RemoteAgentRequest(task={"id": "task-1"}, output_dir=tmp_path)

    actual = client.run_task(request)

    assert isinstance(client, RemoteAgentClient)
    assert actual == response
    assert client.requests == [request]


def test_mock_remote_agent_default_response_is_success(tmp_path: Path) -> None:
    client = MockRemoteAgentClient()

    response = client.run_task(RemoteAgentRequest(task={"id": "task-1"}, output_dir=tmp_path))

    assert response.ok is True


def test_mock_remote_agent_can_raise_error(tmp_path: Path) -> None:
    client = MockRemoteAgentClient(error="planned failure")

    with pytest.raises(RemoteAgentError, match="planned failure"):
        client.run_task(RemoteAgentRequest(task={"id": "task-1"}, output_dir=tmp_path))


def test_mock_remote_agent_can_raise_custom_error(tmp_path: Path) -> None:
    client = MockRemoteAgentClient(error=RemoteAgentError("custom failure"))

    with pytest.raises(RemoteAgentError, match="custom failure"):
        client.run_task(RemoteAgentRequest(task={"id": "task-1"}, output_dir=tmp_path))


def test_mock_remote_agent_can_raise_timeout(tmp_path: Path) -> None:
    client = MockRemoteAgentClient(timeout=True)

    with pytest.raises(RemoteAgentTimeoutError, match="Mock remote agent timeout"):
        client.run_task(RemoteAgentRequest(task={"id": "task-1"}, output_dir=tmp_path))
