# blender-mcp Fork Patch Notes

This document describes every change BMA_Bench makes to the upstream `blender-mcp` server, why each change exists, and how to maintain the fork over time.

- **Upstream:** [ahujasid/blender-mcp](https://github.com/ahujasid/blender-mcp)
- **Fork location:** `blender-mcp-bma/` (gitignored in the main repo — it is its own git repository)
- **Fork branch:** `bma-benchmark-profile-support`
- **Patch marker:** every BMA-specific addition in source files is tagged with a `# BMA_PATCH` comment

---

## Changes made to the fork

### New files

| File | Purpose |
|---|---|
| `src/blender_mcp/bma_env.py` | Reads `BMA_*` environment variables with safe defaults. Re-reads on every call so tests can monkey-patch `os.environ`. |
| `src/blender_mcp/tool_profiles.py` | Defines the five benchmark profiles, their allowed tool sets, and the public API (`is_tool_enabled`, `is_python_allowed`, `is_external_asset_allowed`, `get_enabled_tools`, `get_disabled_tools`). |
| `headless/headless_socket_mode.py` | Provides `install_headless_keepalive()` — a `bpy.app.timers`-based keep-alive for `--background` mode (modal operators require a window). |
| `headless/start_blender_mcp_headless.py` | Blender `--python` bootstrap: discovers and imports `addon.py`, configures the socket server from env vars, installs the keep-alive timer, and sets up SIGTERM/SIGINT handlers. |
| `tests/test_tool_profiles.py` | Unit tests for tool-gating logic (no Blender required). |
| `BMA_PATCHES.md` | Human-readable changelog in Russian (original planning notes). |

### Modified files

| File | Change summary |
|---|---|
| `src/blender_mcp/server.py` | Added `_bma_gated()` decorator for all tool functions; added `get_bma_profile_info` tool; added seven `bma_*` benchmark-safe structured tools. |
| `src/blender_mcp/telemetry.py` | Telemetry defaults to **off** (`DISABLE_TELEMETRY=true`); opt-in requires `BMA_ENABLE_TELEMETRY=true`. |
| `addon.py` | Headless support: detects `BMA_HEADLESS=1` and skips UI-dependent registration paths. |

### Files not modified

`main.py`, `pyproject.toml`, `uv.lock`, `LICENSE`, `README.md`, `TERMS_AND_CONDITIONS.md`, `src/blender_mcp/__init__.py`, `src/blender_mcp/telemetry_decorator.py`.

---

## Environment variables added

All variables are read fresh on every access (via `bma_env.py`) so environment patches in tests take effect immediately.

| Variable | Default | Description |
|---|---|---|
| `BMA_MCP_PROFILE` | `minimal` | Active tool-gating profile. Must be one of `minimal`, `inspection_enabled`, `no_python`, `python_enabled`, `full`. |
| `BMA_DISABLED_TOOLS` | _(empty)_ | Comma-separated list of tools to disable on top of the profile's restrictions. Cannot re-enable tools that the profile blocks absolutely. |
| `BMA_ENABLE_EXTERNAL_ASSETS` | `false` | Set to `true` to allow external asset tools in profiles that normally permit them (e.g. `full`). Has no effect in safe profiles. |
| `BMA_ALLOW_PYTHON_EXECUTION` | `false` | Set to `true` to allow `execute_blender_code` in profiles that normally permit Python. Has no effect in safe profiles (`minimal`, `no_python`, `inspection_enabled`). |
| `BMA_HEADLESS` | `0` | Set to `1` to activate headless mode (no UI, `bpy.app.timers` keep-alive). |
| `BMA_ADDON_PATH` | _(auto)_ | Explicit path to `addon.py` when the default discovery fails. |
| `BMA_SOCKET_HOST` | `localhost` | Host the Blender socket server listens on. |
| `BMA_SOCKET_PORT` | `9876` | Port the Blender socket server listens on. |
| `DISABLE_TELEMETRY` | `true` (fork default) | Direct telemetry kill-switch inherited from upstream. Always set to `true` by `build_mcp_env()` in the main project. |
| `BMA_ENABLE_TELEMETRY` | _(unset)_ | **Only** way to turn telemetry back on in benchmark mode. Set `BMA_ENABLE_TELEMETRY=true` explicitly. |

---

## How profile enforcement works

### Enforcement layers

1. **Absolute restrictions (cannot be overridden):**
   - `execute_blender_code` is **always blocked** for `minimal`, `no_python`, and `inspection_enabled` profiles — regardless of `BMA_ALLOW_PYTHON_EXECUTION` or `BMA_DISABLED_TOOLS`.
   - All external asset tools (Poly Haven, Sketchfab, Hyper3D, Hunyuan3D) are **always blocked** for the same three profiles.

2. **Profile allowed set:** each profile defines an explicit `frozenset` of permitted tools (or `None` for `full`, meaning unrestricted).

3. **`BMA_DISABLED_TOOLS` override:** additional tools can be disabled at runtime for any profile. This layer is applied after the profile check, not before.

### The `_bma_gated` decorator

Every tool function in `server.py` is wrapped with `@_bma_gated("tool_name")`:

```python
def _bma_gated(tool_name: str):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            profile = get_profile()                       # reads BMA_MCP_PROFILE
            if not is_tool_enabled(tool_name, profile):  # checks all layers
                raise RuntimeError(
                    f"Tool '{tool_name}' is disabled by benchmark profile '{profile.name}'."
                )
            return func(*args, **kwargs)
        return _wrapper
    return _decorator
```

The decorator reads the active profile on **every call** (not at import time), so the profile can be changed between calls in tests by patching `os.environ`.

### Enforcement constants

Two `frozenset` constants in `tool_profiles.py` define which profiles are safe:

```python
# Profiles where execute_blender_code is unconditionally blocked.
PYTHON_SAFE_PROFILES = frozenset({"minimal", "no_python", "inspection_enabled"})

# Profiles where external asset tools are unconditionally blocked.
EXTERNAL_ASSET_SAFE_PROFILES = frozenset({"minimal", "no_python", "inspection_enabled"})
```

---

## Which tools are disabled by which profile

Legend: **✓** = allowed, **—** = blocked

| Tool | `minimal` | `inspection_enabled` | `no_python` | `python_enabled` | `full` |
|---|---|---|---|---|---|
| `get_bma_profile_info` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `get_scene_info` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `get_object_info` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `bma_get_scene_info` | ✓ | — | ✓ | ✓ | ✓ |
| `bma_create_object` | ✓ | — | ✓ | ✓ | ✓ |
| `bma_set_transform` | ✓ | — | ✓ | ✓ | ✓ |
| `bma_set_material` | ✓ | — | ✓ | ✓ | ✓ |
| `bma_create_light` | ✓ | — | ✓ | ✓ | ✓ |
| `bma_create_camera` | ✓ | — | ✓ | ✓ | ✓ |
| `bma_create_camera_look_at` | ✓ | — | ✓ | ✓ | ✓ |

### `create_camera_look_at` socket contract

The addon socket handler for `create_camera_look_at` must **always** return JSON (never an empty socket response):

- Success: `{"status": "success", "result": {"ok": true, "tool": "bma_create_camera_look_at", "camera_name": "...", "location": [...], "target": "...", "set_active": true}}`
- Missing target object: `{"status": "error", "error": {"type": "ObjectNotFound", "message": "...", "camera_name": "...", "target": "..."}}`
- Other failures: structured `CameraLookAtFailed` error payload with `camera_name` and `target`.
| `bma_export_scene` | ✓ | — | ✓ | ✓ | ✓ |
| `get_viewport_screenshot` | — | ✓ | ✓ | ✓ | ✓ |
| `execute_blender_code` | **—** | **—** | **—** | ✓ | ✓ |
| `get_polyhaven_status` | — | — | — | — | ✓ |
| `search_polyhaven_assets` | — | — | — | — | ✓ |
| `download_polyhaven_asset` | — | — | — | — | ✓ |
| `set_texture` | — | — | — | — | ✓ |
| `get_sketchfab_status` | — | — | — | — | ✓ |
| `search_sketchfab_models` | — | — | — | — | ✓ |
| `download_sketchfab_model` | — | — | — | — | ✓ |
| `get_hyper3d_status` | — | — | — | — | ✓ |
| `generate_hyper3d_model_via_text` | — | — | — | — | ✓ |
| `generate_hyper3d_model_via_images` | — | — | — | — | ✓ |
| `poll_rodin_job_status` | — | — | — | — | ✓ |
| `import_generated_asset` | — | — | — | — | ✓ |
| `get_hunyuan3d_status` | — | — | — | — | ✓ |
| `generate_hunyuan3d_model` | — | — | — | — | ✓ |
| `poll_hunyuan_job_status` | — | — | — | — | ✓ |
| `import_generated_asset_hunyuan` | — | — | — | — | ✓ |

The bold **—** rows for `execute_blender_code` are absolute restrictions that no environment variable override can lift.

---

## Updating from upstream

```bash
# 1. Add the upstream remote (once)
cd blender-mcp-bma
git remote add upstream https://github.com/ahujasid/blender-mcp.git

# 2. Fetch new upstream commits
git fetch upstream

# 3. Check what changed upstream
git log upstream/main ^HEAD --oneline

# 4. Rebase onto upstream (preferred over merge for clean history)
git checkout bma-benchmark-profile-support
git rebase upstream/main

# 5. Resolve conflicts
#    - Files tagged # BMA_PATCH contain benchmark-specific code — keep those blocks.
#    - New upstream tools added to server.py need @_bma_gated("new_tool_name")
#      and an entry in _ALL_TOOLS in tool_profiles.py.
#    - New upstream tools should default to blocked in safe profiles
#      unless they are provably safe (no Python, no network).

# 6. Run the fork tests
python -m pytest tests/ -v

# 7. Run the main project tests
cd ..
pytest -m "not mcp"
```

After a successful rebase, update the `upstream_commit` reference in any documentation that tracks it.

---

## Running the fork

### Via `uvx` (from a git URL)

```bash
# Install and run directly from the fork's GitHub remote:
uvx --from git+https://github.com/yourorg/blender-mcp-bma.git blender-mcp
```

### Via `uvx` (from a local path)

```bash
# Run from the local checkout — no install step needed:
uvx --from ./blender-mcp-bma blender-mcp
```

### Via `pip` / editable install

```bash
pip install -e ./blender-mcp-bma
blender-mcp
```

### Via the BMA CLI (recommended)

The `bma-mcp` CLI wraps all of the above and passes the correct environment variables automatically:

```bash
# Fork distribution from local path:
bma-mcp \
    --config configs/mcp/minimal.yaml \
    --profile minimal \
    start-server --distribution fork --package-source ./blender-mcp-bma

# Fork distribution from a git URL:
bma-mcp \
    --profile minimal \
    start-server \
    --distribution fork \
    --package-source git+https://github.com/yourorg/blender-mcp-bma.git
```

### Environment required when running manually

If you start the fork's MCP server without the BMA CLI, set these variables:

```bash
export BMA_MCP_PROFILE=minimal
export DISABLE_TELEMETRY=true
export BMA_HEADLESS=0          # 1 for --background mode

blender-mcp                    # or: uvx --from ./blender-mcp-bma blender-mcp
```
