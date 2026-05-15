# MCP Integration

This document covers Stage 5 of BMA_Bench: integrating the Blender MCP server into the benchmark pipeline.

---

## Why blender-mcp

[blender-mcp](https://github.com/ahujasid/blender-mcp) exposes Blender's Python API over the Model Context Protocol (MCP). It lets an LLM agent control a live Blender session through structured tool calls — querying scene state, creating objects, applying materials, running arbitrary Python, and interacting with external asset services (Poly Haven, Sketchfab, Hyper3D, Hunyuan3D).

BMA_Bench uses blender-mcp as the primary interface between the agent under evaluation and the Blender environment. All tool calls made by the agent during a benchmark run go through the MCP server, producing a reproducible, observable record of what the agent did.

---

## Why a fork is needed

The upstream blender-mcp server has no tool-gating mechanism: every registered tool is always available, and telemetry is opt-out at the user level. Benchmark runs need stricter control:

- **Profile-based tool gating** — restrict which tools are accessible depending on the benchmark scenario (read-only probing, no Python execution, full access, etc.).
- **Guaranteed telemetry off** — prevent any usage data from being sent to external services during automated runs.
- **Structured `bma_*` tools** — benchmark-safe tools that operate without arbitrary Python (`bma_create_object`, `bma_set_transform`, etc.), making it possible to evaluate agents that are restricted from running `execute_blender_code`.
- **Headless mode** — `blender --background` keep-alive via `bpy.app.timers` instead of modal operators (which require a UI window).
- **`get_bma_profile_info` tool** — lets the agent (and the benchmark runner) inspect the active profile at runtime.

The fork lives at `blender-mcp-bma/` (branch `bma-benchmark-profile-support`) and is ignored by the main project's `.gitignore` because it is its own git repository.

---

## Supported profiles

The active profile is set via the `BMA_MCP_PROFILE` environment variable (default: `minimal`). Each profile is an absolute restriction — environment variables cannot lift the core safety constraints.

| Profile | Python (`execute_blender_code`) | External assets | `bma_*` tools | Notes |
|---|---|---|---|---|
| `minimal` | No | No | Yes | Safe read + structured mutations |
| `inspection_enabled` | No | No | No | Adds `get_viewport_screenshot` |
| `no_python` | No | No | Yes | All core tools except Python |
| `python_enabled` | **Yes** | No | Yes | Unrestricted Python, no external assets |
| `full` | **Yes** | **Yes** | Yes | Identical to upstream behaviour |

Config files for all five profiles are in `configs/mcp/`.

Additional tools can be disabled at runtime with `BMA_DISABLED_TOOLS=tool1,tool2` (comma-separated).

---

## How telemetry is disabled

The fork honours the `DISABLE_TELEMETRY=true` environment variable and defaults to telemetry off. All five bundled config files (`configs/mcp/*.yaml`) set:

```yaml
disable_telemetry: true
env:
  DISABLE_TELEMETRY: "true"
```

The `smoke` command checks `telemetry_disabled` and fails if it is not `true`.

---

## Running the upstream server

The upstream server requires no installation when using `uvx`:

```bash
# Start Blender first (with the blender-mcp add-on enabled in Blender's preferences)
# Then start the MCP server:
bma-mcp --config configs/mcp/minimal.yaml start-server

# Or use uvx directly:
uvx blender-mcp
```

The upstream server does **not** support tool profiles or `bma_*` tools. Use it only with the `full` or `minimal` upstream configs when the fork is unavailable.

### Check connectivity

```bash
bma-mcp --config configs/mcp/minimal.yaml check
```

Returns a JSON report with `status: OK` if the Blender socket is reachable.

---

## Running the fork

The fork must be checked out at `blender-mcp-bma/` (or a vendored copy at `vendor/blender-mcp-bma/`).

### Headless mode (recommended for CI)

```bash
# Start Blender in background with the BMA add-on:
bma-mcp --config configs/mcp/minimal.yaml start-headless-blender --wait

# Or manually:
blender --background --factory-startup \
    --python benchmark/mcp/headless/start_blender_mcp_headless.py \
    -- --addon blender-mcp-bma/addon.py \
       --host localhost --port 9876 \
       --disable-external-assets
```

### Fork MCP server (upstream transport over fork add-on)

```bash
bma-mcp \
    --config configs/mcp/minimal.yaml \
    --profile minimal \
    start-server --distribution fork --package-source ./blender-mcp-bma
```

### Environment variables (fork)

| Variable | Default | Effect |
|---|---|---|
| `BMA_MCP_PROFILE` | `minimal` | Active tool-gating profile |
| `BMA_DISABLED_TOOLS` | _(empty)_ | Comma-separated extra disabled tools |
| `BMA_ENABLE_EXTERNAL_ASSETS` | `false` | Override to allow asset tools |
| `BMA_ALLOW_PYTHON_EXECUTION` | `false` | Override to allow `execute_blender_code` |
| `DISABLE_TELEMETRY` | `true` | Must be `true` for benchmark runs |

---

## Running the smoke check

The smoke command verifies the full MCP stack without involving an LLM agent:

```bash
# Basic smoke (fails if Blender is not running):
bma-mcp --config configs/mcp/minimal.yaml smoke --output /tmp/smoke.json

# With a specific profile:
bma-mcp --profile inspection_enabled smoke --output /tmp/smoke.json

# Using the Python API:
from benchmark.mcp.execution_backend import McpExecutionBackend
from benchmark.runner.models import ExecutionMode, RunConfig
from pathlib import Path

config = RunConfig(
    run_id="smoke_001",
    task_id="connectivity_check",
    execution_mode=ExecutionMode.MCP_SMOKE,
    artifacts_dir=Path("artifacts"),
    output_dir=Path("artifacts/out"),
    mcp_config_path=Path("configs/mcp/minimal.yaml"),
    mcp_profile="minimal",
)
result = McpExecutionBackend().execute(config)
print(result.ok)
```

The smoke command checks (in order):

1. Config loaded
2. Telemetry disabled
3. Profile valid
4. Blender socket reachable
5. MCP server reachable (same socket)
6. `get_scene_info` tool callable
7. `get_bma_profile_info` callable (fork/local distributions only)

Results are written as JSON to the path given by `--output`.

### Running tests

```bash
# All tests (no Blender required):
pytest -m "not mcp"

# Real MCP integration tests (requires Blender running):
pytest -m mcp
```

---

## What is not included in Stage 5

The following are explicitly out of scope for Stage 5:

- **LLM agent runtime** — Stage 5 only establishes the MCP transport layer. Wiring an actual language model to the MCP tool stream is a later stage.
- **Tool-call metrics** — latency, token counts, and per-tool success rates are not collected by `McpExecutionBackend`.
- **Multi-turn conversation** — the smoke backend makes one-shot tool calls; multi-turn agent dialogue is not implemented here.
- **Asset tool validation** — Poly Haven, Sketchfab, Hyper3D, and Hunyuan3D tools are catalogued and gated but not invoked during smoke checks (they require external network access).
- **Blender rendering pipeline** — render submission and image comparison live in the Blender Automation layer (Stage 2/3), not in the MCP integration.
- **Automated Blender installation** — `HeadlessBlenderMcpLauncher` assumes `blender` is already on `PATH` or discoverable via `BLENDER_PATH`.
- **Fork publication** — `blender-mcp-bma` is a local fork intended for benchmark use; it is not published to PyPI.
