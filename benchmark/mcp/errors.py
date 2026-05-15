class McpLayerError(Exception):
    """Base exception for all MCP-layer errors."""


class McpConfigError(McpLayerError):
    """Raised when an MCP server configuration is invalid or cannot be loaded."""


class McpProfileError(McpLayerError):
    """Raised when an unknown or incompatible BMA profile is requested."""


class ToolDisabledError(McpLayerError):
    """Raised when a tool is blocked by the active BMA profile."""


class UnknownToolError(McpLayerError):
    """Raised when a requested tool is not registered in the tool registry."""


class McpServerStartError(McpLayerError):
    """Raised when the headless Blender MCP server fails to start."""


class BlenderSocketUnavailableError(McpLayerError):
    """Raised when a connection to the Blender add-on socket cannot be established."""


class McpSmokeError(McpLayerError):
    """Raised when a smoke-check tool call fails."""


# --- Execution-level errors not triggered by profile/socket issues ---

class McpExecutionError(McpLayerError):
    """Raised when an MCP tool call fails during execution (I/O, bad response, etc.)."""


class McpHealthcheckError(McpLayerError):
    """Raised when a healthcheck against the Blender socket times out."""
