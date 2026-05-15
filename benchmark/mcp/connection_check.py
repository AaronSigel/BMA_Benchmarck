from __future__ import annotations

import socket
from dataclasses import dataclass, field

from benchmark.mcp.config import McpServerConfig
from benchmark.mcp.errors import BlenderSocketUnavailableError


@dataclass
class ConnectionCheckResult:
    blender_socket_available: bool
    host: str
    port: int
    profile: str
    telemetry_disabled: bool
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.blender_socket_available and not self.issues


def check_blender_socket(host: str, port: int, timeout_sec: float = 5.0) -> None:
    """Attempt a TCP connection to the Blender add-on socket.

    Raises BlenderSocketUnavailableError if unreachable.
    """
    try:
        sock = socket.create_connection((host, port), timeout=timeout_sec)
        sock.close()
    except OSError as exc:
        raise BlenderSocketUnavailableError(
            f"Cannot reach Blender socket at {host}:{port}: {exc}"
        ) from exc


def check_mcp_server_config(config: McpServerConfig) -> ConnectionCheckResult:
    """Validate config and probe the Blender socket.

    Returns ConnectionCheckResult with blender_socket_available, host, port,
    profile, telemetry_disabled, and a list of any issues found.
    """
    issues: list[str] = []

    from benchmark.mcp.profiles import McpProfile
    try:
        McpProfile(config.profile)
    except ValueError:
        issues.append(
            f"Unknown profile '{config.profile}'. "
            f"Valid profiles: {[p.value for p in McpProfile]}"
        )

    if config.server_distribution in ("fork", "local") and not config.package_source:
        issues.append(
            f"server_distribution='{config.server_distribution}' requires package_source"
        )

    socket_available = False
    try:
        check_blender_socket(
            config.blender_host,
            config.blender_port,
            timeout_sec=float(config.startup_timeout_sec),
        )
        socket_available = True
    except BlenderSocketUnavailableError as exc:
        issues.append(str(exc))

    return ConnectionCheckResult(
        blender_socket_available=socket_available,
        host=config.blender_host,
        port=config.blender_port,
        profile=config.profile,
        telemetry_disabled=config.disable_telemetry,
        issues=issues,
    )


def is_blender_socket_available(config: McpServerConfig) -> bool:
    """Return True if the Blender socket is reachable, False otherwise."""
    try:
        check_blender_socket(
            config.blender_host,
            config.blender_port,
            timeout_sec=float(config.startup_timeout_sec),
        )
        return True
    except BlenderSocketUnavailableError:
        return False
