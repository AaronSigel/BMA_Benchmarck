"""Headless bootstrap for the blender-mcp add-on (vendored/local fork).

Used when blender-mcp-bma is vendored inside BMA_Bench (server_distribution="local")
rather than installed as a system package.

Usage:
    blender --background --factory-startup \\
        --python benchmark/mcp/headless/start_blender_mcp_headless.py \\
        -- \\
        --addon vendor/blender-mcp-bma/addon.py \\
        --host 127.0.0.1 \\
        --port 9876 \\
        --disable-external-assets

Arguments after '--' override the corresponding environment variables.

Environment variables (all optional):
    BMA_ADDON_PATH              Explicit path to addon.py (overrides auto-discovery)
    BMA_SOCKET_HOST             Bind host (default: localhost)
    BMA_SOCKET_PORT             Bind port (default: 9876)
    BMA_MCP_PROFILE             Tool-gating profile (default: minimal)
    BMA_ENABLE_EXTERNAL_ASSETS  'true' to enable asset integrations (default: false)
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Guard: this script only runs inside a Blender Python interpreter.
# ---------------------------------------------------------------------------
try:
    import bpy  # type: ignore[import]
except ImportError:
    print("[BMA headless] ERROR: must be executed inside Blender (--background --python).")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Parse CLI args passed after '--' separator (override env vars)
# ---------------------------------------------------------------------------
def _parse_script_args() -> argparse.Namespace:
    """Parse args from sys.argv that follow the '--' Blender separator."""
    parser = argparse.ArgumentParser(prog="start_blender_mcp_headless", add_help=False)
    parser.add_argument("--addon", default=None, help="Path to addon.py")
    parser.add_argument("--host", default=None, help="Socket bind host")
    parser.add_argument("--port", type=int, default=None, help="Socket bind port")
    parser.add_argument("--disable-external-assets", action="store_true", default=False)
    # Blender passes everything after '--' into sys.argv; slice past the separator.
    try:
        sep_idx = sys.argv.index("--")
        script_argv = sys.argv[sep_idx + 1:]
    except ValueError:
        script_argv = []
    args, _ = parser.parse_known_args(script_argv)
    return args


_cli = _parse_script_args()

# ---------------------------------------------------------------------------
# Configuration: CLI args > env vars > defaults
# ---------------------------------------------------------------------------
_HOST = _cli.host or os.environ.get("BMA_SOCKET_HOST", "localhost")
_PORT = _cli.port or int(os.environ.get("BMA_SOCKET_PORT", "9876"))
_PROFILE = os.environ.get("BMA_MCP_PROFILE", "minimal")
_EXTERNAL_ASSETS: bool
if _cli.disable_external_assets:
    _EXTERNAL_ASSETS = False
else:
    _EXTERNAL_ASSETS = os.environ.get("BMA_ENABLE_EXTERNAL_ASSETS", "false").lower() in (
        "1", "true", "yes"
    )

# ---------------------------------------------------------------------------
# Locate addon.py
# ---------------------------------------------------------------------------
# Priority:
#   1. --addon CLI arg
#   2. BMA_ADDON_PATH env var
#   3. Vendored fork next to the BMA_Bench root  (blender-mcp-bma/addon.py)
#   4. vendor/blender-mcp-bma/addon.py  (alternative vendor directory)

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent.parent  # BMA_Bench/

_CANDIDATE_PATHS: list[Path] = []
if _cli.addon:
    _CANDIDATE_PATHS.append(Path(_cli.addon).resolve())
if os.environ.get("BMA_ADDON_PATH"):
    _CANDIDATE_PATHS.append(Path(os.environ["BMA_ADDON_PATH"]).resolve())
_CANDIDATE_PATHS += [
    _PROJECT_ROOT / "blender-mcp-bma" / "addon.py",
    _PROJECT_ROOT / "vendor" / "blender-mcp-bma" / "addon.py",
]


def _find_addon() -> Path:
    for p in _CANDIDATE_PATHS:
        if p.is_file():
            return p
    checked = "\n  ".join(str(p) for p in _CANDIDATE_PATHS)
    raise FileNotFoundError(
        f"[BMA headless] Cannot find addon.py. Checked:\n  {checked}\n"
        "Set BMA_ADDON_PATH to the explicit path."
    )


_addon_path = _find_addon()
print(f"[BMA headless] Loading addon from {_addon_path}")

# ---------------------------------------------------------------------------
# Import addon.py programmatically (it lives outside any installed package)
# ---------------------------------------------------------------------------
if str(_addon_path.parent) not in sys.path:
    sys.path.insert(0, str(_addon_path.parent))

_spec = importlib.util.spec_from_file_location("blender_mcp_addon", _addon_path)
_addon = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["blender_mcp_addon"] = _addon
_spec.loader.exec_module(_addon)  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
# Register the add-on
# ---------------------------------------------------------------------------
_addon.register()

# ---------------------------------------------------------------------------
# Configure scene properties (set after register() creates them)
# ---------------------------------------------------------------------------
scene = bpy.context.scene
scene.blendermcp_port = _PORT
scene.blendermcp_use_polyhaven = _EXTERNAL_ASSETS
scene.blendermcp_use_hyper3d = _EXTERNAL_ASSETS
scene.blendermcp_use_sketchfab = _EXTERNAL_ASSETS
if hasattr(scene, "blendermcp_use_hunyuan3d"):
    scene.blendermcp_use_hunyuan3d = _EXTERNAL_ASSETS

# ---------------------------------------------------------------------------
# Start the Blender-side socket server
# ---------------------------------------------------------------------------
bpy.types.blendermcp_server = _addon.BlenderMCPServer(host=_HOST, port=_PORT)
bpy.types.blendermcp_server.start()
scene.blendermcp_server_running = True

print(
    f"[BMA headless] Socket server started  host={_HOST}  port={_PORT}  "
    f"profile={_PROFILE}  external_assets={_EXTERNAL_ASSETS}"
)

# ---------------------------------------------------------------------------
# Signal handlers — clean shutdown on SIGTERM / SIGINT
# ---------------------------------------------------------------------------
_shutdown_requested = False


def _request_shutdown(signum, frame):  # noqa: ARG001
    global _shutdown_requested
    print(f"[BMA headless] Signal {signum} received, requesting shutdown…")
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _request_shutdown)
signal.signal(signal.SIGINT, _request_shutdown)

# ---------------------------------------------------------------------------
# Keep-alive modal operator: polls every 0.5 s for shutdown signal
# ---------------------------------------------------------------------------

class BMA_OT_HeadlessKeepAlive(bpy.types.Operator):
    bl_idname = "bma.headless_keep_alive"
    bl_label = "BMA Headless Keep-Alive"

    _timer = None

    def modal(self, context, event):
        if event.type == "TIMER":
            server_dead = (
                hasattr(bpy.types, "blendermcp_server")
                and not bpy.types.blendermcp_server.running
            )
            if _shutdown_requested or server_dead:
                self.cancel(context)
                bpy.ops.wm.quit_blender()
                return {"CANCELLED"}
        return {"PASS_THROUGH"}

    def execute(self, context):
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if hasattr(bpy.types, "blendermcp_server"):
            bpy.types.blendermcp_server.stop()
            del bpy.types.blendermcp_server
        scene.blendermcp_server_running = False
        print("[BMA headless] Shutdown complete.")


bpy.utils.register_class(BMA_OT_HeadlessKeepAlive)


def _start_keepalive():
    bpy.ops.bma.headless_keep_alive("INVOKE_DEFAULT")


bpy.app.timers.register(_start_keepalive, first_interval=0.1)
