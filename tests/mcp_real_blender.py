"""Утилиты для интеграционных MCP-тестов с живым Blender."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmark.mcp.config import McpServerConfig, config_from_env, load_mcp_config
from benchmark.mcp.connection_check import check_blender_socket
from benchmark.mcp.errors import BlenderSocketUnavailableError


def require_blender_mcp_socket(
    cfg: McpServerConfig | None = None,
    *,
    config_path: str | Path | None = None,
    timeout_sec: float | None = None,
) -> McpServerConfig:
    """Короткий TCP connect к сокету аддона; иначе pytest.skip (без долгого smoke).

    Таймаут по умолчанию 2 с; переопределение: BMA_MCP_TEST_SOCKET_TIMEOUT_SEC.
    """
    if timeout_sec is None:
        raw = os.environ.get("BMA_MCP_TEST_SOCKET_TIMEOUT_SEC", "").strip()
        timeout_sec = float(raw) if raw else 2.0

    if cfg is not None:
        resolved = cfg
    elif config_path is not None:
        resolved = load_mcp_config(Path(config_path))
    else:
        resolved = config_from_env()

    try:
        check_blender_socket(
            resolved.blender_host,
            resolved.blender_port,
            timeout_sec=timeout_sec,
        )
    except BlenderSocketUnavailableError as exc:
        pytest.skip(
            f"{exc} (fast probe {timeout_sec:g}s). "
            "Поднимите Blender с blender-mcp или задайте BMA_SOCKET_HOST / BMA_SOCKET_PORT."
        )
    return resolved
