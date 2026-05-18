from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
from pathlib import Path
from typing import Any

from benchmark.mcp.config import McpServerConfig, build_mcp_env
from benchmark.mcp.errors import (
    BlenderSocketUnavailableError,
    McpExecutionError,
    McpServerStartError,
    ToolDisabledError,
)
from benchmark.mcp.profiles import McpProfile, is_tool_allowed
from benchmark.mcp.tool_registry import McpToolRegistry

_RECV_BUFFER = 8192


def _connect_host(host: str) -> str:
    return "127.0.0.1" if host == "localhost" else host


class ExternalBlenderMcpServerAdapter:
    """Adapter for launching and communicating with a blender-mcp server process.

    Supports three distribution modes via McpServerConfig.server_distribution:
      "upstream"  → uvx blender-mcp
      "fork"      → uvx --from git+<package_source> blender-mcp
      "local"     → uvx --from <local-path> blender-mcp
    """

    def __init__(
        self,
        config: McpServerConfig,
        registry: McpToolRegistry | None = None,
    ) -> None:
        self._config = config
        self._registry = registry or McpToolRegistry()

    @property
    def config(self) -> McpServerConfig:
        return self._config

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    def build_command(self, config: McpServerConfig | None = None) -> list[str]:
        """Return the argv list for launching the MCP server.

        Distribution:
          upstream → [config.command, *config.args]   (e.g. uvx blender-mcp)
          fork     → [config.command, --from, <package_source>, blender-mcp]
          local    → [config.command, --from, <package_source>, blender-mcp]
        """
        cfg = config or self._config
        dist = cfg.server_distribution

        if dist == "upstream":
            return [cfg.command, *cfg.args]

        if dist in ("fork", "local"):
            source = cfg.package_source
            if not source:
                raise McpServerStartError(
                    f"server_distribution='{dist}' requires package_source to be set"
                )
            return [cfg.command, "--from", source, "blender-mcp"]

        raise McpServerStartError(f"Unknown server_distribution: '{dist}'")

    def build_env(self, config: McpServerConfig | None = None) -> dict[str, str]:
        """Return the env dict for the server process (OS env merged with BMA vars)."""
        cfg = config or self._config
        env = dict(os.environ)
        env.update(build_mcp_env(cfg))
        return env

    def start(self, config: McpServerConfig | None = None) -> subprocess.Popen:
        """Launch the MCP server subprocess and return the Popen handle."""
        cfg = config or self._config
        cmd = self.build_command(cfg)
        env = self.build_env(cfg)
        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise McpServerStartError(
                f"MCP server executable not found: {cmd[0]!r}. "
                "Is 'uv' / 'uvx' installed?"
            ) from exc
        except OSError as exc:
            raise McpServerStartError(
                f"Failed to start MCP server with command {cmd}: {exc}"
            ) from exc
        return process

    def stop(self, process: subprocess.Popen, timeout: float = 5.0) -> None:
        """Gracefully terminate the server process; SIGKILL on timeout."""
        if process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            process.wait()

    def is_running(self, process: subprocess.Popen) -> bool:
        """Return True if the process is still alive."""
        return process.poll() is None

    # ------------------------------------------------------------------
    # Tool invocation (socket-based)
    # ------------------------------------------------------------------

    def call_tool(self, tool_name: str, params: dict[str, Any] | None = None) -> Any:
        """Call a tool on the Blender MCP socket and return the parsed response."""
        try:
            profile = McpProfile(self._config.profile)
        except ValueError:
            profile = McpProfile.FULL

        if not is_tool_allowed(tool_name, profile):
            raise ToolDisabledError(
                f"Tool '{tool_name}' is not allowed in profile '{self._config.profile}'"
            )

        socket_cmd = _TOOL_TO_SOCKET_CMD.get(tool_name, tool_name)
        payload = json.dumps({"type": socket_cmd, "params": params or {}}).encode()

        try:
            sock = socket.create_connection(
                (_connect_host(self._config.blender_host), self._config.blender_port),
                timeout=self._config.startup_timeout_sec,
            )
        except OSError as exc:
            raise BlenderSocketUnavailableError(
                f"Cannot connect to Blender socket at "
                f"{self._config.blender_host}:{self._config.blender_port}: {exc}"
            ) from exc

        try:
            sock.settimeout(self._config.request_timeout_sec)
            sock.sendall(payload)
            raw = _recv_all(sock)
        except OSError as exc:
            raise McpExecutionError(f"Socket I/O error calling '{tool_name}': {exc}") from exc
        finally:
            sock.close()

        try:
            response = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise McpExecutionError(
                f"Invalid JSON response from '{tool_name}': {exc}"
            ) from exc

        if isinstance(response, dict) and response.get("status") == "error":
            raise McpExecutionError(
                f"Tool '{tool_name}' returned error: {response.get('error', response)}"
            )

        return response.get("result", response) if isinstance(response, dict) else response

    def reset_scene(self) -> dict:
        """Reset the Blender scene unconditionally, bypassing profile restrictions."""
        return self.execute_code_unrestricted(_RESET_SCENE_CODE)

    def collect_scene_snapshot(self, output_path: Path | str) -> dict:
        """Collect a SceneSnapshot as harness infrastructure, bypassing profile restrictions."""
        code = _collect_snapshot_code(Path(output_path))
        return self.execute_code_unrestricted(code)

    def execute_code_unrestricted(self, code: str) -> dict:
        """Run Blender Python for harness-only operations, not model-visible tool use."""
        payload = json.dumps({"type": "execute_code", "params": {"code": code}}).encode()
        try:
            sock = socket.create_connection(
                (_connect_host(self._config.blender_host), self._config.blender_port),
                timeout=self._config.startup_timeout_sec,
            )
            sock.settimeout(self._config.request_timeout_sec)
            sock.sendall(payload)
            raw = _recv_all(sock)
            sock.close()
            response = json.loads(raw)
            return response.get("result", response) if isinstance(response, dict) else {}
        except Exception as exc:
            return {"warning": f"execute_code failed: {exc}"}

    def list_tools(self) -> list[str]:
        try:
            profile = McpProfile(self._config.profile)
        except ValueError:
            profile = McpProfile.FULL
        return [c.name for c in self._registry.list_for_profile(profile)]


# Maps benchmark tool names to the socket command types the Blender addon understands.
# bma_* tools are wrappers in server.py that call these underlying socket commands.
_TOOL_TO_SOCKET_CMD: dict[str, str] = {
    "execute_blender_code": "execute_code",
    "bma_get_scene_info": "get_scene_info",
    "bma_create_object": "create_object",
    "bma_set_transform": "set_transform",
    "bma_set_material": "set_material",
    "bma_create_light": "create_light",
    "bma_create_camera": "create_camera",
    "bma_export_scene": "export_scene",
}


_RESET_SCENE_CODE = """\
import bpy
for obj in list(bpy.data.objects):
    try:
        bpy.data.objects.remove(obj, do_unlink=True)
    except Exception:
        pass
for attr in ("meshes", "materials", "lights", "cameras"):
    col = getattr(bpy.data, attr)
    for block in list(col):
        if getattr(block, "users", 0) == 0:
            try:
                col.remove(block, do_unlink=True)
            except TypeError:
                col.remove(block)
print("reset_scene:ok")
"""


def _collect_snapshot_code(output_path: Path) -> str:
    import inspect as _inspect

    from benchmark.blender.scripts.collect_snapshot import collect_snapshot as _collect_snapshot

    module = _inspect.getmodule(_collect_snapshot)
    source = _inspect.getsource(module)
    return (
        f"{source}\n"
        "import json as _j\n"
        f"_snap = collect_snapshot({{'output_path': {str(output_path)!r}}})\n"
        "print(_j.dumps(_snap))\n"
    )


def _recv_all(sock: socket.socket, buffer_size: int = _RECV_BUFFER) -> bytes:
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
