# Blender Agent Benchmark

This project is a benchmark stand for evaluating AI agents that work with Blender through MCP.

The repository contains benchmark task definitions, Blender artifact automation, scene validation, an experiment runner, and an MCP integration layer for connecting agents to a live Blender session via the Model Context Protocol. LLM execution and agent runtime remain out of scope for the implemented pipeline.

Implemented stages:

- Stage 1: benchmark task format, task registry, schema, and task CLI.
- Stage 2: Blender automation artifacts such as `scene_snapshot.json`, renders, exports, and `.blend` files.
- Stage 3: scene validation and `validation_result.json`.
- Stage 4: experiment runner, batch runner, run artifacts, and summary metrics.
- Stage 5: MCP integration layer — blender-mcp fork, tool-gating profiles, headless mode, and MCP smoke checks.

## Documentation

The benchmark task YAML format is documented in [docs/task_format.md](docs/task_format.md).

The current task inventory is listed in [TASKS.md](TASKS.md).

Blender automation usage is documented in [docs/blender_automation.md](docs/blender_automation.md).

Scene validation usage and result format are documented in
[docs/scene_validation.md](docs/scene_validation.md).

Experiment runner usage, configs, artifact layout, and summary exports are
documented in [docs/experiment_runner.md](docs/experiment_runner.md).

MCP integration overview (fork, profiles, telemetry, smoke) is documented in
[docs/mcp_integration.md](docs/mcp_integration.md).

Headless Blender launch and keep-alive design are documented in
[docs/mcp_headless.md](docs/mcp_headless.md).

Fork patch notes, env variables, and profile enforcement are documented in
[docs/blender_mcp_fork_patch.md](docs/blender_mcp_fork_patch.md).

## Common Commands

Install the project in a virtual environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

Install optional dependencies for local Blender automation work:

```bash
.venv/bin/python -m pip install -e '.[blender]'
```

Install development dependencies:

```bash
.venv/bin/python -m pip install -e '.[dev]'
```

Run tests:

```bash
.venv/bin/pytest
```

List tasks:

```bash
.venv/bin/python -m benchmark.tasks.cli list --tasks-dir tasks
```

Show one task:

```bash
.venv/bin/python -m benchmark.tasks.cli show geometry_001_basic_primitives --tasks-dir tasks
```

Validate all tasks:

```bash
.venv/bin/python -m benchmark.tasks.cli validate --tasks-dir tasks
```

Run scene validation:

```bash
.venv/bin/python -m benchmark.validation.cli validate \
  --task tasks/geometry/geometry_001_basic_primitives.yaml \
  --snapshot artifacts/blender_smoke/scene_snapshot.json \
  --output artifacts/validation/geometry_001_validation_result.json
```

Run an experiment:

```bash
.venv/bin/python -m benchmark.runner.cli experiment \
  --config configs/example_experiment.yaml
```

The example experiment uses an existing `SceneSnapshot` fixture and does not
require Blender, MCP, or LLM access.

## Stage 2: Blender Automation

Stage 2 adds the dependency and configuration surface for running Blender in headless mode from the benchmark package. The goal of this stage is to launch Blender, execute Blender Python scripts inside that process, and save formal artifacts such as scene snapshots under `artifacts/`.

Full usage notes are documented in [docs/blender_automation.md](docs/blender_automation.md).

The optional `blender` extra installs Pillow for image-related artifact handling and NumPy for Blender's glTF exporter:

```bash
.venv/bin/python -m pip install -e '.[blender]'
```

The project does not install `bpy` as a pip dependency. Code that needs Blender's Python API must run inside a Blender process, and regular pytest tests must keep passing on machines where the Blender executable is not installed. Blender-specific tests should use the `blender` marker, and broader end-to-end checks should use the `integration` marker.

Stage 2 does not include MCP, LLM execution, or agent architectures.

## Stage 3: Scene Validation

Stage 3 validates a `SceneSnapshot` against a `BenchmarkTask`. The validator
checks objects, transforms, materials, lights, cameras, and export artifacts, and
writes a structured `validation_result.json`.

Full usage notes are documented in [docs/scene_validation.md](docs/scene_validation.md).

Stage 3 does not call MCP, LLMs, or Blender.

## Stage 4: Experiment Runner

Stage 4 adds `RunConfig`, `ExperimentConfig`, execution backends, `ExperimentRunner`,
`BatchRunner`, metrics aggregation, `summary.csv`, `summary.json`, and runner CLI
commands.

The runner supports:

- `external_snapshot`: validate an existing `SceneSnapshot` JSON file.
- `replay`: copy existing artifacts into a run directory and validate them.
- `blender_smoke`: run the existing Blender smoke automation when Blender is available.

Run artifacts are written under `artifacts/runs/<run_id>/`, and batch-level
artifacts include `experiment_result.json`, `run_results.json`, `summary.json`,
`summary.csv`, and `metrics.csv`.

Full usage notes are documented in [docs/experiment_runner.md](docs/experiment_runner.md).

Stage 4 still does not include MCP integration, LLM calls, agent runtime, or
tool-call metrics.

## Stage 5: MCP Integration Layer

Stage 5 connects the benchmark pipeline to a live Blender session via the Model Context Protocol. It does not implement an LLM agent or tool-call metrics — it establishes the transport layer that a future agent stage will use.

### blender-mcp fork

Stage 5 does not write an MCP server from scratch. It uses a fork of the open-source [blender-mcp](https://github.com/ahujasid/blender-mcp) server, maintained at `blender-mcp-bma/` on branch `bma-benchmark-profile-support`. The fork adds benchmark-specific features while keeping all upstream tool implementations intact.

Every BMA-specific addition in the fork is tagged with a `# BMA_PATCH` comment. See [docs/blender_mcp_fork_patch.md](docs/blender_mcp_fork_patch.md) for the full list of changes, env variables, and the upstream rebase procedure.

### Tool-gating profiles

The fork enforces five named profiles via the `BMA_MCP_PROFILE` environment variable. Each profile controls which MCP tools are accessible:

| Profile | Python execution | External assets | Notes |
|---|---|---|---|
| `minimal` | No | No | Safe read + structured `bma_*` mutations |
| `inspection_enabled` | No | No | Adds `get_viewport_screenshot` |
| `no_python` | No | No | All core tools, no Python |
| `python_enabled` | Yes | No | Adds `execute_blender_code` |
| `full` | Yes | Yes | Identical to upstream behaviour |

`execute_blender_code` and all external asset tools are **unconditionally blocked** in `minimal`, `no_python`, and `inspection_enabled` — no environment variable can lift these restrictions.

Config files for all five profiles are in `configs/mcp/`. See [docs/mcp_integration.md](docs/mcp_integration.md) for the full profile reference.

### Telemetry

Telemetry is **off by default** in the fork. The `DISABLE_TELEMETRY=true` variable is set automatically by `build_mcp_env()`. The MCP smoke command fails if telemetry is not disabled. To opt back in (not recommended for automated runs), set `BMA_ENABLE_TELEMETRY=true`.

### Headless mode

Benchmark runs do not use a graphical Blender session. Stage 5 provides `HeadlessBlenderMcpLauncher`, which runs:

```bash
blender --background --factory-startup \
    --python benchmark/mcp/headless/start_blender_mcp_headless.py \
    -- --addon blender-mcp-bma/addon.py --host localhost --port 9876
```

Modal operators cannot run in `--background` mode (no window). The fork replaces them with a `bpy.app.timers`-based keep-alive (`persistent=True`) that keeps Blender alive and processes socket commands in the main thread.

```bash
# CLI (blocks until Ctrl-C or SIGTERM):
bma-mcp --config configs/mcp/minimal.yaml start-headless-blender --wait
```

See [docs/mcp_headless.md](docs/mcp_headless.md) for details on the bootstrap sequence, socket healthcheck, and process shutdown.

### MCP smoke check (no LLM required)

`McpExecutionBackend` integrates with the Stage 4 `ExecutionBackend` ABC and runs a lightweight MCP smoke check without any LLM or agent runtime:

```bash
bma-mcp --config configs/mcp/minimal.yaml smoke --output /tmp/smoke.json
```

The smoke command verifies: config loaded → telemetry disabled → profile valid → Blender socket reachable → `get_scene_info` callable → `get_bma_profile_info` callable (fork only). Results are written as `mcp_smoke_result.json` in the run output directory.

```bash
# Check connectivity only:
bma-mcp --config configs/mcp/minimal.yaml check

# List tools allowed by a profile:
bma-mcp --profile minimal list-tools

# Run tests (no Blender required):
pytest -m "not mcp"

# Run real MCP integration tests (requires Blender running):
pytest -m mcp
```
