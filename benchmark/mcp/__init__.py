"""MCP layer for BMA_Bench: config, profiles, tool registry, server adapter, backends."""

from benchmark.mcp.config import (
    McpServerConfig,
    build_mcp_env,
    config_from_env,
    dump_mcp_config,
    load_mcp_config,
    load_named_config,
)
from benchmark.mcp.errors import (
    BlenderSocketUnavailableError,
    McpConfigError,
    McpExecutionError,
    McpHealthcheckError,
    McpLayerError,
    McpProfileError,
    McpServerStartError,
    McpSmokeError,
    ToolDisabledError,
    UnknownToolError,
)
from benchmark.mcp.profiles import McpProfile, get_allowed_tools, is_tool_allowed
from benchmark.mcp.tool_contract import ToolContract
from benchmark.mcp.tool_registry import McpToolRegistry
from benchmark.mcp.server_adapter import ExternalBlenderMcpServerAdapter
from benchmark.mcp.connection_check import check_blender_socket, is_blender_socket_available
from benchmark.mcp.models import McpSmokeResult

__all__ = [
    "BlenderSocketUnavailableError",
    "check_blender_socket",
    "config_from_env",
    "ExternalBlenderMcpServerAdapter",
    "get_allowed_tools",
    "is_blender_socket_available",
    "is_tool_allowed",
    "build_mcp_env",
    "dump_mcp_config",
    "load_mcp_config",
    "load_named_config",
    "McpConfigError",
    "McpExecutionError",
    "McpHealthcheckError",
    "McpLayerError",
    "McpProfile",
    "McpProfileError",
    "McpServerConfig",
    "McpServerStartError",
    "McpSmokeError",
    "McpSmokeResult",
    "McpToolRegistry",
    "ToolContract",
    "ToolDisabledError",
    "UnknownToolError",
]
