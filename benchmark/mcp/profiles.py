from __future__ import annotations

from enum import Enum


class McpProfile(str, Enum):
    MINIMAL = "minimal"
    NO_PYTHON = "no_python"
    PYTHON_ENABLED = "python_enabled"
    INSPECTION_ENABLED = "inspection_enabled"
    FULL = "full"


# Fork-only bma_* benchmark-safe structured tools.
_BMA_SAFE_TOOLS: frozenset[str] = frozenset({
    "bma_get_scene_info",
    "bma_create_object",
    "bma_set_transform",
    "bma_set_material",
    "bma_create_light",
    "bma_create_camera",
    "bma_export_scene",
})

# External asset integration tools (require network + keys).
_EXTERNAL_ASSET_TOOLS: frozenset[str] = frozenset({
    "get_polyhaven_status",
    "get_polyhaven_categories",
    "search_polyhaven_assets",
    "download_polyhaven_asset",
    "get_sketchfab_status",
    "search_sketchfab_models",
    "download_sketchfab_model",
    "get_hyper3d_status",
    "generate_hyper3d_model_via_text",
    "generate_hyper3d_model_via_images",
    "poll_rodin_job_status",
    "import_generated_asset",
    "get_hunyuan3d_status",
    "generate_hunyuan3d_model",
    "poll_hunyuan_job_status",
    "import_generated_asset_hunyuan",
})

# Upstream inspection tools.
_INSPECTION_TOOLS: frozenset[str] = frozenset({
    "get_bma_profile_info",
    "get_scene_info",
    "get_object_info",
    "get_viewport_screenshot",
})

# Python execution tool.
_PYTHON_TOOLS: frozenset[str] = frozenset({"execute_blender_code"})

# Remaining non-asset, non-python upstream tools.
_OTHER_UPSTREAM_TOOLS: frozenset[str] = frozenset({"set_texture"})

# Complete set of all known tools (upstream + fork bma_*).
_ALL_TOOLS: frozenset[str] = (
    _INSPECTION_TOOLS
    | _PYTHON_TOOLS
    | _BMA_SAFE_TOOLS
    | _EXTERNAL_ASSET_TOOLS
    | _OTHER_UPSTREAM_TOOLS
)

# Core tools safe in all profiles (no Python, no external assets).
_CORE_TOOLS: frozenset[str] = _INSPECTION_TOOLS | _BMA_SAFE_TOOLS | _OTHER_UPSTREAM_TOOLS

# None means "all tools allowed" (no gating).
_ALLOWED_TOOLS: dict[McpProfile, frozenset[str] | None] = {
    # minimal — safe read-only + structured bma_* tools; no Python, no external.
    McpProfile.MINIMAL: frozenset({
        "get_bma_profile_info",
        "get_scene_info",
        "get_object_info",
    }) | _BMA_SAFE_TOOLS,

    # inspection_enabled — focused read-only scene inspection; no Python, no external.
    McpProfile.INSPECTION_ENABLED: frozenset({
        "get_bma_profile_info",
        "get_scene_info",
        "get_object_info",
        "get_viewport_screenshot",
    }),

    # no_python — all core + non-asset tools; no Python, no external.
    McpProfile.NO_PYTHON: _CORE_TOOLS,

    # python_enabled — core tools + execute_blender_code; no external assets.
    McpProfile.PYTHON_ENABLED: _CORE_TOOLS | _PYTHON_TOOLS,

    # full — unrestricted; all upstream + fork tools.
    McpProfile.FULL: None,
}


def get_allowed_tools(profile: McpProfile) -> frozenset[str] | None:
    """Return the allowed tool names for the profile, or None for unrestricted."""
    return _ALLOWED_TOOLS[profile]


def is_tool_allowed(tool_name: str, profile: McpProfile) -> bool:
    allowed = get_allowed_tools(profile)
    return allowed is None or tool_name in allowed


def profile_from_env(env_value: str | None) -> McpProfile:
    """Parse BMA_MCP_PROFILE env value; fall back to FULL on unknown values."""
    if env_value is None:
        return McpProfile.FULL
    try:
        return McpProfile(env_value.strip().lower())
    except ValueError:
        return McpProfile.FULL
