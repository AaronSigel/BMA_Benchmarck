# MCP Headless Mode

This document explains how BMA_Bench launches Blender in `--background` mode for automated benchmark runs without a graphical interface.

---

## Why the UI add-on workflow does not work headless

The standard blender-mcp workflow assumes a full Blender session with a UI:

1. The user installs the add-on via Blender's Preferences → Add-ons panel.
2. The add-on registers a **modal operator** (`BlenderMCPModalOperator`) that runs on every timer tick inside Blender's event loop.
3. The event loop processes UI events, including the timer ticks that drive the operator.

In `--background` mode (`blender --background`) there is **no window and no UI event loop**. Modal operators require a context with a window to invoke — calling `bpy.ops.*` in background mode raises:

```
RuntimeError: Operator bpy.ops.wm.some_modal_op.poll() failed, context is incorrect
```

Additionally, the standard workflow relies on the user manually enabling the add-on through the UI, which is impossible in a headless CI pipeline.

---

## Launching Blender with `--background`

`HeadlessBlenderMcpLauncher` (in `benchmark/mcp/headless/launcher.py`) builds and runs the full command automatically. The equivalent manual invocation is:

```bash
blender \
    --background \
    --factory-startup \
    --python benchmark/mcp/headless/start_blender_mcp_headless.py \
    -- \
    --addon blender-mcp-bma/addon.py \
    --host localhost \
    --port 9876 \
    --disable-external-assets
```

Flag reference:

| Flag | Purpose |
|---|---|
| `--background` | No window, no GUI. Required for headless mode. |
| `--factory-startup` | Ignore the user's Blender preferences and start with defaults. Prevents accidental add-on conflicts. |
| `--python <script>` | Python script to run immediately after Blender initialises. This is the headless bootstrap. |
| `--` | Separator: everything after this is passed to the Python script as `sys.argv`, not to Blender. |
| `--addon <path>` | Path to `addon.py` from the BMA fork. |
| `--host` / `--port` | Socket address the blender-mcp server will listen on. |
| `--disable-external-assets` | Prevents the add-on from trying to contact Poly Haven, Sketchfab, etc. |

### Using the launcher (Python API)

```python
from pathlib import Path
from benchmark.mcp.config import McpServerConfig
from benchmark.mcp.headless.launcher import HeadlessBlenderMcpLauncher

config = McpServerConfig(
    server_distribution="local",
    package_source="./blender-mcp-bma",
    blender_host="localhost",
    blender_port=9876,
    profile="minimal",
    disable_telemetry=True,
)

# Context-manager form: start on __enter__, stop on __exit__.
with HeadlessBlenderMcpLauncher(config) as launcher:
    # Blender is running here, socket is (or will be) available.
    ...
```

### Using the CLI

```bash
# Start and block until SIGINT/SIGTERM (--wait mode):
bma-mcp --config configs/mcp/minimal.yaml start-headless-blender --wait

# Start without blocking (fire-and-forget, for scripted pipelines):
bma-mcp --config configs/mcp/minimal.yaml start-headless-blender
```

### Blender binary discovery

The launcher looks for `blender` in this order:

1. `BMA_BLENDER_BIN` environment variable (explicit path)
2. `shutil.which("blender")` (PATH lookup)

If neither finds a binary, `McpServerStartError` is raised before any subprocess is spawned.

---

## How the headless bootstrap works

`benchmark/mcp/headless/start_blender_mcp_headless.py` is the Python script Blender executes via `--python`. Its steps:

1. **Locate `addon.py`** — checks (in order):
   - `--addon <path>` CLI argument (passed after `--`)
   - `BMA_ADDON_PATH` environment variable
   - `blender-mcp-bma/addon.py` relative to the project root
   - `vendor/blender-mcp-bma/addon.py`

2. **Import the add-on** — uses `importlib.util.spec_from_file_location` to load `addon.py` from an arbitrary filesystem path without it being on `sys.path`.

3. **Register the add-on** — calls `addon.register()`, which starts the blender-mcp socket server thread.

4. **Configure the server** — reads `BMA_SOCKET_HOST`, `BMA_SOCKET_PORT`, `BMA_MCP_PROFILE`, and `BMA_ENABLE_EXTERNAL_ASSETS` from the environment to override defaults.

5. **Install the headless keep-alive** — calls `install_headless_keepalive()` from `blender-mcp-bma/headless/headless_socket_mode.py` (see below).

6. **Install signal handlers** — SIGTERM and SIGINT set a `_shutdown_requested` flag, which the keep-alive timer checks on every tick.

### The keep-alive timer

Instead of a modal operator, the fork uses `bpy.app.timers.register()` with `persistent=True`:

```python
def _tick() -> float | None:
    if shutdown_requested() or not server.running:
        server.stop()
        sys.exit(0)
    return interval  # float → reschedule; None → cancel

bpy.app.timers.register(_tick, first_interval=0.5, persistent=True)
```

The timer fires on every tick (default 0.5 s). Returning a `float` reschedules it; returning `None` cancels it, allowing Blender to exit normally. `persistent=True` keeps the timer alive across scene loads and file resets.

Incoming tool commands from the socket are executed in the Blender main thread via the existing `bpy.app.timers.register(execute_wrapper)` mechanism already present in the add-on, so no changes to the command-execution path were needed.

---

## Checking the socket

After launching, poll until the socket is ready before sending any commands:

### Python API

```python
from benchmark.mcp.headless.healthcheck import wait_for_blender_socket

wait_for_blender_socket(config, retries=20, interval_sec=0.5)
# Raises BlenderSocketUnavailableError if socket is not up after ~10 s.
```

For a one-shot probe:

```python
from benchmark.mcp.headless.healthcheck import get_scene_info_via_socket

info = get_scene_info_via_socket("localhost", 9876, timeout_sec=5.0)
```

### CLI check

```bash
bma-mcp --config configs/mcp/minimal.yaml check
```

Returns JSON with `status: OK` when the socket is reachable, `status: FAIL` with a `blender_socket_available: false` field otherwise.

### Low-level socket probe

```python
from benchmark.mcp.connection_check import check_blender_socket, BlenderSocketUnavailableError

try:
    check_blender_socket("localhost", 9876, timeout_sec=2.0)
    print("Blender socket is up")
except BlenderSocketUnavailableError as exc:
    print(f"Not ready: {exc}")
```

---

## Why `get_viewport_screenshot` is not required in headless

`get_viewport_screenshot` captures the contents of an active 3D viewport. In `--background` mode:

- There is no window, no viewport, and no GPU context.
- The tool would either return an error or produce a blank/black image.

The `inspection_enabled` profile includes `get_viewport_screenshot` for interactive sessions where a viewport is present. In headless benchmark runs, the `minimal` or `no_python` profile is used instead, neither of which lists `get_viewport_screenshot` in their allowed tool sets.

If a benchmark task genuinely needs a render, use `bma_export_scene` (which writes a `.blend` file for offline rendering) or the Blender Automation layer (Stage 2/3 of BMA_Bench), which handles render submission as a separate pipeline step.

---

## Stopping the process

### Via the launcher (recommended)

```python
launcher.stop(timeout=10.0)
```

Sequence:
1. Sends **SIGTERM** to the entire process group (`os.killpg`) so child processes of Blender also receive the signal.
2. Waits up to `timeout` seconds for the process to exit.
3. If it does not exit in time, sends **SIGKILL** to the process group.
4. Calls `process.wait()` to reap the zombie.
5. Sets `self._process = None`.

### Via the CLI (`--wait` mode)

When `start-headless-blender --wait` is used, the CLI blocks until it receives SIGINT (Ctrl-C) or SIGTERM, then calls `launcher.stop()` automatically.

### Manually (kill by PID)

```bash
# Find the PID:
pgrep -f "blender.*background"

# Graceful shutdown:
kill -TERM <pid>

# Immediate kill (only if TERM is ignored):
kill -KILL <pid>
```

The headless bootstrap installs SIGTERM and SIGINT handlers that set a shutdown flag, so a well-behaved `kill -TERM` is enough for a clean exit in normal operation.
