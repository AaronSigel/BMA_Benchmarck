import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from benchmark.agent.errors import RemoteAgentError, RemoteAgentTimeoutError
from benchmark.agent.models import RemoteAgentConfig, RemoteAgentProvider
from benchmark.agent.remote import GenericRemoteAgentClient, RemoteAgentRequest


class FakeHttpResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_generic_http_remote_agent_posts_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeHttpResponse:
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))  # type: ignore[union-attr]
        return FakeHttpResponse({"ok": True, "scene_snapshot_path": "scene_snapshot.json"})

    monkeypatch.setenv("REMOTE_AGENT_API_KEY", "secret")
    monkeypatch.setattr("benchmark.agent.remote.remote_agent_client.urllib.request.urlopen", fake_urlopen)
    client = GenericRemoteAgentClient(
        RemoteAgentConfig(
            provider=RemoteAgentProvider.GENERIC_HTTP,
            agent_id="http-agent",
            endpoint_url="https://remote.example/run",
            api_key_env="REMOTE_AGENT_API_KEY",
            timeout_sec=10,
        )
    )

    response = client.run_task(
        RemoteAgentRequest(
            task={"id": "task-1"},
            mcp_config_path=Path("configs/mcp/minimal.yaml"),
            mcp_profile="minimal",
            tool_contracts=[{"name": "get_scene_info"}],
            output_dir=tmp_path,
        )
    )

    assert response.ok is True
    assert captured["url"] == "https://remote.example/run"
    assert captured["timeout"] == 10
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["body"]["task"] == {"id": "task-1"}
    assert captured["body"]["tool_contracts"] == [{"name": "get_scene_info"}]
    assert captured["body"]["mcp_config_path"] == "configs/mcp/minimal.yaml"


def test_generic_command_remote_agent_runs_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["input"] = json.loads(kwargs["input"])
        captured["timeout"] = kwargs["timeout"]
        captured["cwd"] = kwargs["cwd"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"ok": True, "error": None}),
            stderr="",
        )

    monkeypatch.setattr("benchmark.agent.remote.remote_agent_client.subprocess.run", fake_run)
    client = GenericRemoteAgentClient(
        RemoteAgentConfig(
            provider=RemoteAgentProvider.GENERIC_COMMAND,
            agent_id="command-agent",
            command="remote-agent",
            args=["--json"],
            workspace_dir=tmp_path,
            timeout_sec=5,
        )
    )

    response = client.run_task(RemoteAgentRequest(task={"id": "task-1"}, output_dir=tmp_path))

    assert response.ok is True
    assert captured["command"] == ["remote-agent", "--json"]
    assert captured["input"]["task"] == {"id": "task-1"}
    assert captured["timeout"] == 5
    assert captured["cwd"] == tmp_path


def test_generic_command_errors_are_wrapped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="failed")

    monkeypatch.setattr("benchmark.agent.remote.remote_agent_client.subprocess.run", fake_run)
    client = GenericRemoteAgentClient(
        RemoteAgentConfig(
            provider=RemoteAgentProvider.GENERIC_COMMAND,
            agent_id="command-agent",
            command="remote-agent",
        )
    )

    with pytest.raises(RemoteAgentError, match="exited with code 2"):
        client.run_task(RemoteAgentRequest(task={"id": "task-1"}, output_dir=tmp_path))


def test_generic_command_timeout_is_wrapped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout=1)

    monkeypatch.setattr("benchmark.agent.remote.remote_agent_client.subprocess.run", fake_run)
    client = GenericRemoteAgentClient(
        RemoteAgentConfig(
            provider=RemoteAgentProvider.GENERIC_COMMAND,
            agent_id="command-agent",
            command="remote-agent",
        )
    )

    with pytest.raises(RemoteAgentTimeoutError, match="timed out"):
        client.run_task(RemoteAgentRequest(task={"id": "task-1"}, output_dir=tmp_path))


def test_generic_http_errors_are_wrapped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeHttpResponse:
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("benchmark.agent.remote.remote_agent_client.urllib.request.urlopen", fake_urlopen)
    client = GenericRemoteAgentClient(
        RemoteAgentConfig(
            provider=RemoteAgentProvider.GENERIC_HTTP,
            agent_id="http-agent",
            endpoint_url="https://remote.example/run",
        )
    )

    with pytest.raises(RemoteAgentError, match="HTTP request failed"):
        client.run_task(RemoteAgentRequest(task={"id": "task-1"}, output_dir=tmp_path))


def test_invalid_response_json_is_wrapped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="not json", stderr="")

    monkeypatch.setattr("benchmark.agent.remote.remote_agent_client.subprocess.run", fake_run)
    client = GenericRemoteAgentClient(
        RemoteAgentConfig(
            provider=RemoteAgentProvider.GENERIC_COMMAND,
            agent_id="command-agent",
            command="remote-agent",
        )
    )

    with pytest.raises(RemoteAgentError, match="invalid JSON"):
        client.run_task(RemoteAgentRequest(task={"id": "task-1"}, output_dir=tmp_path))
