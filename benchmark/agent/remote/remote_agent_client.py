from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from typing import Any

from pydantic import ValidationError

from benchmark.agent.errors import RemoteAgentError, RemoteAgentTimeoutError
from benchmark.agent.models import RemoteAgentConfig, RemoteAgentProvider
from benchmark.agent.remote.base import RemoteAgentRequest, RemoteAgentResponse


class GenericRemoteAgentClient:
    """Remote agent client for generic HTTP and command-wrapper integrations."""

    def __init__(self, config: RemoteAgentConfig) -> None:
        self.config = config

    def run_task(self, request: RemoteAgentRequest) -> RemoteAgentResponse:
        if self.config.provider in {RemoteAgentProvider.GENERIC_HTTP, RemoteAgentProvider.CODEX, RemoteAgentProvider.CLAUDE_CODE}:
            return self._run_http(request)
        if self.config.provider == RemoteAgentProvider.GENERIC_COMMAND:
            return self._run_command(request)
        raise RemoteAgentError(
            f"Unsupported remote agent provider: {self.config.provider}",
            provider=str(self.config.provider),
            agent_id=self.config.agent_id,
        )

    def _run_http(self, request: RemoteAgentRequest) -> RemoteAgentResponse:
        if self.config.endpoint_url is None:
            raise RemoteAgentError(
                "endpoint_url is required for generic_http remote agents",
                provider=self.config.provider.value,
                agent_id=self.config.agent_id,
            )

        payload = _request_payload(request)
        headers = {"Content-Type": "application/json"}
        api_key = _read_api_key(self.config)
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        http_request = urllib.request.Request(
            self.config.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                http_request,
                timeout=self.config.timeout_sec,
            ) as response:
                raw = response.read().decode("utf-8")
        except TimeoutError as error:
            raise RemoteAgentTimeoutError(
                f"Remote agent HTTP request timed out: {error}",
                provider=self.config.provider.value,
                agent_id=self.config.agent_id,
            ) from error
        except urllib.error.URLError as error:
            raise RemoteAgentError(
                f"Remote agent HTTP request failed: {error}",
                provider=self.config.provider.value,
                agent_id=self.config.agent_id,
            ) from error
        return _parse_response(raw, self.config)

    def _run_command(self, request: RemoteAgentRequest) -> RemoteAgentResponse:
        if self.config.command is None:
            raise RemoteAgentError(
                "command is required for generic_command remote agents",
                provider=self.config.provider.value,
                agent_id=self.config.agent_id,
            )

        command = [self.config.command, *self.config.args]
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(_request_payload(request)),
                text=True,
                capture_output=True,
                timeout=self.config.timeout_sec,
                cwd=self.config.workspace_dir,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RemoteAgentTimeoutError(
                f"Remote agent command timed out: {error}",
                provider=self.config.provider.value,
                agent_id=self.config.agent_id,
            ) from error
        except OSError as error:
            raise RemoteAgentError(
                f"Remote agent command failed to start: {error}",
                provider=self.config.provider.value,
                agent_id=self.config.agent_id,
            ) from error

        if completed.returncode != 0:
            raise RemoteAgentError(
                f"Remote agent command exited with code {completed.returncode}: {completed.stderr}",
                provider=self.config.provider.value,
                agent_id=self.config.agent_id,
            )
        return _parse_response(completed.stdout, self.config)


def _request_payload(request: RemoteAgentRequest) -> dict[str, Any]:
    return request.model_dump(mode="json", exclude_none=True)


def _parse_response(raw: str, config: RemoteAgentConfig) -> RemoteAgentResponse:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RemoteAgentError(
            f"Remote agent returned invalid JSON: {error}",
            provider=config.provider.value,
            agent_id=config.agent_id,
        ) from error
    try:
        return RemoteAgentResponse.model_validate(data)
    except ValidationError as error:
        raise RemoteAgentError(
            f"Remote agent returned invalid response: {error}",
            provider=config.provider.value,
            agent_id=config.agent_id,
        ) from error


def _read_api_key(config: RemoteAgentConfig) -> str | None:
    if config.api_key_env is None:
        return None
    return os.environ.get(config.api_key_env)
