from __future__ import annotations

import json
import socket
import time
from typing import Any

from benchmark.mcp.config import McpServerConfig
from benchmark.mcp.connection_check import is_blender_socket_available
from benchmark.mcp.errors import BlenderSocketUnavailableError, McpExecutionError

_DEFAULT_RETRIES = 20
_DEFAULT_INTERVAL_SEC = 0.5
_RECV_BUFFER = 8192


def send_blender_socket_command(
    host: str,
    port: int,
    command_type: str,
    params: dict[str, Any] | None = None,
    timeout_sec: float = 10.0,
) -> Any:
    """Send a JSON command to the Blender socket server and return the parsed response.

    Payload format:  {"type": <command_type>, "params": <params>}
    """
    payload = json.dumps({"type": command_type, "params": params or {}}).encode()

    try:
        sock = socket.create_connection((host, port), timeout=timeout_sec)
    except OSError as exc:
        raise BlenderSocketUnavailableError(
            f"Cannot connect to Blender socket at {host}:{port}: {exc}"
        ) from exc

    try:
        sock.settimeout(timeout_sec)
        sock.sendall(payload)
        raw = _recv_all(sock)
    except OSError as exc:
        raise McpExecutionError(
            f"Socket I/O error while calling '{command_type}': {exc}"
        ) from exc
    finally:
        sock.close()

    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise McpExecutionError(
            f"Invalid JSON response from '{command_type}': {exc}"
        ) from exc

    if isinstance(response, dict) and response.get("status") == "error":
        raise McpExecutionError(
            f"Command '{command_type}' returned error: {response.get('error', response)}"
        )

    return response.get("result", response) if isinstance(response, dict) else response


def get_scene_info_via_socket(
    host: str,
    port: int,
    timeout_sec: float = 10.0,
) -> Any:
    """Send {"type": "get_scene_info", "params": {}} and return the parsed result.

    Convenience wrapper around send_blender_socket_command for healthchecks.
    """
    return send_blender_socket_command(
        host=host,
        port=port,
        command_type="get_scene_info",
        params={},
        timeout_sec=timeout_sec,
    )


def wait_for_blender_socket(
    config: McpServerConfig,
    retries: int = _DEFAULT_RETRIES,
    interval_sec: float = _DEFAULT_INTERVAL_SEC,
) -> None:
    """Poll until the Blender socket is reachable or retries are exhausted."""
    for attempt in range(1, retries + 1):
        if is_blender_socket_available(config):
            return
        if attempt < retries:
            time.sleep(interval_sec)

    raise BlenderSocketUnavailableError(
        f"Blender socket at {config.blender_host}:{config.blender_port} did not become "
        f"available after {retries} attempts ({retries * interval_sec:.1f}s)"
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _recv_all(sock: socket.socket, buffer_size: int = _RECV_BUFFER) -> bytes:
    """Read from socket until a complete JSON object is received."""
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(buffer_size)
        if not chunk:
            break
        chunks.append(chunk)
        try:
            json.loads(b"".join(chunks))
            break
        except json.JSONDecodeError:
            continue
    return b"".join(chunks)
