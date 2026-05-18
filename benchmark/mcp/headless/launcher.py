from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path

from benchmark.mcp.config import McpServerConfig, build_mcp_env
from benchmark.mcp.errors import McpServerStartError

# Default relative path to the vendored fork's addon.py, resolved from the
# project root (BMA_Bench/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_HEADLESS_DIR = Path(__file__).parent
_BLOCKING_ADDON_SCRIPT = _PROJECT_ROOT / "blender-mcp-bma" / "headless" / "start_headless_blocking.py"
_ADDON_SCRIPT = (
    _BLOCKING_ADDON_SCRIPT
    if _BLOCKING_ADDON_SCRIPT.is_file()
    else _HEADLESS_DIR / "start_blender_mcp_headless.py"
)
_DEFAULT_ADDON_PATH = _PROJECT_ROOT / "blender-mcp-bma" / "addon.py"


class HeadlessBlenderMcpLauncher:
    """Launches Blender in background mode with the blender-mcp add-on socket active.

    Builds and runs:
        blender --background --factory-startup \\
            --python start_blender_mcp_headless.py \\
            -- \\
            --addon <addon_path> \\
            --host <host> \\
            --port <port> \\
            [--disable-external-assets]
    """

    def __init__(
        self,
        config: McpServerConfig,
        addon_path: Path | str | None = None,
    ) -> None:
        self._config = config
        self._addon_path = Path(addon_path) if addon_path else _resolve_addon_path(config)
        self._process: subprocess.Popen | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def build_command(self) -> list[str]:
        """Return the full argv list for launching headless Blender."""
        blender_bin = _find_blender()
        if blender_bin is None:
            raise McpServerStartError(
                "Blender executable not found. "
                "Set BMA_BLENDER_BIN or add 'blender' to PATH."
            )

        cmd: list[str] = [
            blender_bin,
            "--background",
            "--factory-startup",
            "--python", str(_ADDON_SCRIPT),
            "--",                               # separator: Blender args / script args
            "--addon", str(self._addon_path),
            "--host", self._config.blender_host,
            "--port", str(self._config.blender_port),
        ]

        # Only pass --disable-external-assets when telemetry/asset tools should be off.
        # The flag is always added in benchmark mode (controlled by BMA_MCP_PROFILE).
        from benchmark.mcp.profiles import McpProfile, _EXTERNAL_ASSET_TOOLS
        try:
            profile = McpProfile(self._config.profile)
        except ValueError:
            profile = McpProfile.FULL

        external_enabled = (
            profile == McpProfile.FULL
            and self._config.server_distribution in ("fork", "local")
        )
        if not external_enabled:
            cmd.append("--disable-external-assets")

        return cmd

    def build_env(self) -> dict[str, str]:
        """Return environment for the Blender subprocess."""
        env = {**os.environ, **build_mcp_env(self._config)}
        env["BMA_HEADLESS"] = "1"
        env["BMA_ADDON_PATH"] = str(self._addon_path)
        from benchmark.mcp.profiles import McpProfile

        try:
            profile = McpProfile(self._config.profile)
        except ValueError:
            profile = McpProfile.FULL
        external_enabled = (
            profile == McpProfile.FULL
            and self._config.server_distribution in ("fork", "local")
        )
        env["BMA_ENABLE_EXTERNAL_ASSETS"] = "true" if external_enabled else "false"
        return env

    def start(self) -> None:
        """Launch the headless Blender process."""
        if self.is_running:
            return

        cmd = self.build_command()
        env = self.build_env()

        try:
            self._process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise McpServerStartError(
                f"Blender binary not found: {cmd[0]!r}"
            ) from exc
        except OSError as exc:
            raise McpServerStartError(
                f"Failed to launch headless Blender: {exc}"
            ) from exc

    def stop(self, timeout: float = 10.0) -> None:
        """Gracefully stop the headless Blender process."""
        if self._process is None:
            return
        if self._process.poll() is None:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self._process.terminate()
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    self._process.kill()
                self._process.wait()
        self._process = None

    def __enter__(self) -> "HeadlessBlenderMcpLauncher":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_blender() -> str | None:
    env_val = os.environ.get("BMA_BLENDER_BIN")
    if env_val:
        return env_val
    return shutil.which("blender")


def _resolve_addon_path(config: McpServerConfig) -> Path:
    """Derive the addon.py path from config.package_source or the default vendor location."""
    if config.package_source and not config.package_source.startswith("git+"):
        candidate = Path(config.package_source) / "addon.py"
        if candidate.is_file():
            return candidate.resolve()
    if _DEFAULT_ADDON_PATH.is_file():
        return _DEFAULT_ADDON_PATH
    # Return the default path even if not yet present (e.g. CI setup step pending).
    return _DEFAULT_ADDON_PATH
