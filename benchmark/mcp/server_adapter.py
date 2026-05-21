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

# Error types that indicate a transient socket failure safe to retry.
_TRANSIENT_ERROR_TYPES = {"EmptySocketResponse", "SocketTimeout", "SocketError"}

# Tools whose calls are idempotent and can be safely retried once on transient errors.
# bma_set_transform / bma_set_material are always idempotent (overwrite existing state).
# bma_create_object is idempotent only when if_exists=update (handled in _call_create_object_with_if_exists).
_IDEMPOTENT_TOOLS = {"bma_set_transform", "bma_set_material", "bma_assign_material"}


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
        """Call a tool on the Blender MCP socket and return the parsed response.

        Idempotent tools (set_transform, set_material, assign_material) and
        bma_create_object with if_exists=update are retried once on transient
        socket failures (EmptySocketResponse, SocketTimeout, SocketError).
        """
        try:
            profile = McpProfile(self._config.profile)
        except ValueError:
            profile = McpProfile.FULL

        if not is_tool_allowed(tool_name, profile):
            raise ToolDisabledError(
                f"Tool '{tool_name}' is not allowed in profile '{self._config.profile}'"
            )

        # Handle if_exists for bma_create_object before dispatching to socket
        if tool_name == "bma_create_object" and isinstance(params, dict) and "if_exists" in params:
            return self._call_create_object_with_if_exists(params, profile)

        result = self._socket_call_once(tool_name, params)

        # Retry once for idempotent tools on transient failures.
        if (
            isinstance(result, dict)
            and not result.get("ok")
            and isinstance(result.get("error"), dict)
            and result["error"].get("type") in _TRANSIENT_ERROR_TYPES
            and tool_name in _IDEMPOTENT_TOOLS
        ):
            result = self._socket_call_once(tool_name, params)

        return result

    def _socket_call_once(self, tool_name: str, params: dict[str, Any] | None) -> Any:
        """Execute a single socket round-trip and return a parsed _tool_envelope dict."""
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
        except TimeoutError:
            return _tool_envelope(
                tool_name,
                ok=False,
                result=None,
                error_type="SocketTimeout",
                error_message=f"Tool call timed out after {self._config.request_timeout_sec} seconds",
            )
        except OSError as exc:
            return _tool_envelope(
                tool_name,
                ok=False,
                result=None,
                error_type="SocketError",
                error_message=f"Socket I/O error calling '{tool_name}': {exc}",
            )
        finally:
            sock.close()

        if not raw:
            return _tool_envelope(
                tool_name,
                ok=False,
                result={"raw_len": 0, "raw_preview": "", "failure_stage": "tool_response_parse"},
                error_type="EmptySocketResponse",
                error_message=f"Empty response from Blender socket for tool '{tool_name}'",
            )

        try:
            response = json.loads(raw)
        except json.JSONDecodeError as exc:
            raw_bytes = raw if isinstance(raw, (bytes, bytearray)) else raw.encode("utf-8", errors="replace")
            raw_preview = raw_bytes.decode("utf-8", errors="replace")[:200]
            return _tool_envelope(
                tool_name,
                ok=False,
                result={"raw_len": len(raw_bytes), "raw_preview": raw_preview, "failure_stage": "tool_response_parse"},
                error_type="InvalidJsonResponse",
                error_message=f"Invalid JSON from '{tool_name}': {exc}",
            )

        if isinstance(response, dict) and response.get("status") == "error":
            error_payload = response.get("error_details") or response.get("error") or response.get("message") or response
            message = error_payload.get("message") if isinstance(error_payload, dict) else error_payload
            error_type = error_payload.get("type") if isinstance(error_payload, dict) else "ToolError"
            return _tool_envelope(
                tool_name,
                ok=False,
                result=error_payload if isinstance(error_payload, dict) else None,
                error_type=str(error_type or "ToolError"),
                error_message=str(message),
            )

        result = response.get("result", response) if isinstance(response, dict) else response

        # If the addon returned an ok=False dict inside a status=success wrapper
        # (e.g. export_scene catches its own exceptions), surface it as an error envelope.
        if isinstance(result, dict) and result.get("ok") is False and "error" in result:
            _err = result.get("error") or {}
            _msg = _err.get("message") if isinstance(_err, dict) else str(_err)
            _etype = _err.get("type") if isinstance(_err, dict) else "ToolError"
            return _tool_envelope(
                tool_name,
                ok=False,
                result=_err if isinstance(_err, dict) else None,
                error_type=str(_etype or "ToolError"),
                error_message=str(_msg or "Tool failed"),
            )

        return _tool_envelope(tool_name, ok=True, result=result)

    def _call_create_object_with_if_exists(
        self, params: dict[str, Any], profile: "McpProfile"
    ) -> dict[str, Any]:
        """Handle bma_create_object with if_exists semantics."""
        if_exists = params.get("if_exists", "update")
        object_name = params.get("name")
        create_params = {k: v for k, v in params.items() if k != "if_exists"}

        # Check if object already exists via get_object_info
        existing = None
        if object_name:
            try:
                info_payload = json.dumps(
                    {"type": "get_object_info", "params": {"name": object_name}}
                ).encode()
                sock = socket.create_connection(
                    (_connect_host(self._config.blender_host), self._config.blender_port),
                    timeout=self._config.startup_timeout_sec,
                )
                try:
                    sock.settimeout(self._config.request_timeout_sec)
                    sock.sendall(info_payload)
                    raw = _recv_all(sock)
                    info = json.loads(raw)
                    result = info.get("result", info) if isinstance(info, dict) else None
                    if isinstance(result, dict) and result.get("name") == object_name:
                        existing = result
                except Exception:
                    existing = None
                finally:
                    sock.close()
            except Exception:
                existing = None

        if existing is not None:
            if if_exists == "skip":
                return {
                    "ok": True,
                    "tool": "bma_create_object",
                    "result": {
                        "object_name": object_name,
                        "created": False,
                        "updated": False,
                        "skipped": True,
                    },
                    "error": None,
                }
            if if_exists == "error":
                return _tool_envelope(
                    "bma_create_object",
                    ok=False,
                    result=None,
                    error_type="ObjectAlreadyExists",
                    error_message=f"Object '{object_name}' already exists",
                )
            # if_exists == "update": fall through to normal create (Blender will auto-update or rename)

        raw_result = self._socket_call_once("bma_create_object", create_params)

        # Retry once on transient failures when if_exists=update (idempotent).
        if (
            if_exists == "update"
            and isinstance(raw_result, dict)
            and not raw_result.get("ok")
            and isinstance(raw_result.get("error"), dict)
            and raw_result["error"].get("type") in _TRANSIENT_ERROR_TYPES
        ):
            raw_result = self._socket_call_once("bma_create_object", create_params)

        if isinstance(raw_result, dict) and raw_result.get("ok"):
            inner = raw_result.get("result") or {}
            if isinstance(inner, dict):
                inner.setdefault("created", existing is None)
                inner.setdefault("updated", existing is not None)
                inner.setdefault("skipped", False)
                raw_result = {**raw_result, "result": inner}
        return raw_result

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
        sock = None
        try:
            sock = socket.create_connection(
                (_connect_host(self._config.blender_host), self._config.blender_port),
                timeout=self._config.startup_timeout_sec,
            )
            sock.settimeout(self._config.request_timeout_sec)
            sock.sendall(payload)
            raw = _recv_all(sock)
            response = _loads_socket_response(raw, "execute_code")
            if isinstance(response, dict) and response.get("status") == "error":
                message = response.get("error") or response.get("message") or response
                return {"warning": f"execute_code failed: {message}"}
            return response.get("result", response) if isinstance(response, dict) else {}
        except Exception as exc:
            return {"warning": f"execute_code failed: {exc}"}
        finally:
            if sock is not None:
                sock.close()

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
    "bma_get_scene_snapshot": "get_scene_info",
    "bma_get_object_info": "get_object_info",
    "bma_create_object": "create_object",
    "bma_set_transform": "set_transform",
    "bma_set_material": "set_material",
    "bma_assign_material": "set_material",
    "bma_create_light": "create_light",
    "bma_create_camera": "create_camera",
    "bma_create_camera_look_at": "create_camera_look_at",
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


def _loads_socket_response(raw: bytes | bytearray | str, command_type: str) -> Any:
    if not raw:
        raise McpExecutionError(f"No response from Blender socket for '{command_type}'")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else raw
        preview = preview[:200]
        raise McpExecutionError(
            f"Invalid JSON response from '{command_type}': {exc}; response={preview!r}"
        ) from exc


def _tool_envelope(
    tool_name: str,
    *,
    ok: bool,
    result: Any,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    if isinstance(result, dict) and {"ok", "tool", "result", "error"}.issubset(result.keys()):
        return result
    return {
        "ok": ok,
        "tool": tool_name,
        "result": result if ok else None,
        "error": None if ok else {
            "type": error_type or "ToolError",
            "message": error_message or "Tool failed",
            **(result if isinstance(result, dict) else {}),
        },
    }
