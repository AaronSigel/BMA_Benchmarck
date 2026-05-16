from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark.agent.models import AgentStepType, AgentStrategyName, AgentTrace
from benchmark.agent.remote import (
    RemoteAgentArtifact,
    RemoteAgentClient,
    RemoteAgentRequest,
    RemoteAgentResponse,
)


class StaticRemoteAgentClient:
    def __init__(self, response: RemoteAgentResponse) -> None:
        self.response = response
        self.requests: list[RemoteAgentRequest] = []

    def run_task(self, request: RemoteAgentRequest) -> RemoteAgentResponse:
        self.requests.append(request)
        return self.response


def test_remote_agent_request_contains_runtime_context(tmp_path: Path) -> None:
    request = RemoteAgentRequest(
        task={"id": "task-1", "prompt": "Create a cube"},
        mcp_config_path=Path("configs/mcp/minimal.yaml"),
        mcp_profile="minimal",
        tool_contracts=[{"name": "get_scene_info"}],
        output_dir=tmp_path,
        metadata={"run_id": "run-1"},
    )

    assert request.task["id"] == "task-1"
    assert request.mcp_config_path == Path("configs/mcp/minimal.yaml")
    assert request.mcp_profile == "minimal"
    assert request.tool_contracts == [{"name": "get_scene_info"}]
    assert request.output_dir == tmp_path
    assert request.metadata == {"run_id": "run-1"}


def test_remote_agent_response_accepts_agent_trace_compatible_dict() -> None:
    response = RemoteAgentResponse(
        ok=True,
        trace={
            "run_id": "run-1",
            "task_id": "task-1",
            "agent_id": "remote-agent",
            "strategy": "remote_agent",
            "steps": [
                {
                    "step_index": 0,
                    "step_type": "final",
                    "observation": "done",
                }
            ],
        },
        scene_snapshot_path=Path("scene_snapshot.json"),
        artifacts=[
            RemoteAgentArtifact(
                name="trace",
                path=Path("agent_trace.json"),
                kind="json",
            )
        ],
        raw_response={"provider": "generic"},
    )

    assert isinstance(response.trace, AgentTrace)
    assert response.trace.strategy == AgentStrategyName.REMOTE_AGENT
    assert response.trace.steps[0].step_type == AgentStepType.FINAL
    assert response.scene_snapshot_path == Path("scene_snapshot.json")
    assert response.artifacts[0].name == "trace"


def test_remote_agent_response_does_not_require_llm_tool_calls() -> None:
    trace = AgentTrace(
        run_id="run-1",
        task_id="task-1",
        agent_id="remote-agent",
        strategy=AgentStrategyName.REMOTE_AGENT,
        final_message="completed by remote agent",
        success=True,
    )
    response = RemoteAgentResponse(ok=True, trace=trace)

    assert response.ok is True
    assert response.trace == trace


def test_remote_agent_client_protocol_is_provider_neutral(tmp_path: Path) -> None:
    client = StaticRemoteAgentClient(RemoteAgentResponse(ok=True))
    request = RemoteAgentRequest(task={"id": "task-1"}, output_dir=tmp_path)

    assert isinstance(client, RemoteAgentClient)
    assert client.run_task(request).ok is True
    assert client.requests == [request]


def test_remote_agent_abstraction_validation(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        RemoteAgentArtifact(name="", path=Path("artifact.json"))
    with pytest.raises(ValidationError):
        RemoteAgentRequest(task={}, output_dir=tmp_path, mcp_profile="")
