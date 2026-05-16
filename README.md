# Blender Agent Benchmark

This project is a benchmark stand for evaluating AI agents that work with Blender through MCP.

The repository contains benchmark task definitions, Blender artifact automation, scene validation, an experiment runner, an MCP integration layer, and an agent runtime for running benchmark tasks through external LLM APIs and remote agents.

Implemented stages:

- Stage 1: benchmark task format, task registry, schema, and task CLI.
- Stage 2: Blender automation artifacts such as `scene_snapshot.json`, renders, exports, and `.blend` files.
- Stage 3: scene validation and `validation_result.json`.
- Stage 4: experiment runner, batch runner, run artifacts, and summary metrics.
- Stage 5: MCP integration layer — blender-mcp fork, tool-gating profiles, headless mode, and MCP smoke checks.
- Stage 6: Agent Runtime — LLM clients, agent strategies, trace recording, and `AgentExecutionBackend`.
- Stage 7: Trace Metrics and Benchmark Reporting — analysis package, tool-call metrics, agent metrics, validation metrics, error taxonomy, comparative reports, and Markdown/HTML/CSV/JSON exports.

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

Agent Runtime architecture, provider formats, strategies, trace format, and CLI
usage are documented in [docs/agent_runtime.md](docs/agent_runtime.md).

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

## Stage 6: Agent Runtime

Stage 6 adds a complete agent execution layer that connects `BenchmarkTask` definitions to external LLM APIs and remote agents, records structured traces, and integrates with the Stage 4 `ExperimentRunner` via `AgentExecutionBackend`.

### Local models and Ollama

Stage 6 uses **API-based models only**. Local inference (Ollama, llama.cpp, GGUF/ONNX) is not supported. `provider: ollama` is rejected at config load time with a `ValidationError`. Local model support can be added in a separate stage as an independent `LlmClient`.

### Supported providers

| Provider | Description |
|---|---|
| `openrouter` | OpenAI-compatible endpoint at `openrouter.ai` |
| `openai_compatible` | Any OpenAI-compatible API (OpenAI, Azure, vLLM, …); `base_url` required |
| `anthropic` | Anthropic Messages API; uses `x-api-key` header and `input_schema` tool format |
| `remote_agent` | Delegates the full task to an external agent via HTTP or subprocess |
| `mock` | Deterministic in-process client for tests; no API key required |

API keys are read from environment variables specified in `api_key_env` and are never stored in traces or config files.

### Agent strategies

| Strategy | Description |
|---|---|
| `direct_tool_calling` | Single LLM call → parse `tool_calls` → execute → repeat until no more calls |
| `react` | Iterative Reason + Act loop; tool errors are fed back as observations |
| `plan_and_execute` | LLM generates a JSON plan; steps are executed sequentially |
| `remote_agent` | No local LLM; full task sent to an external agent |

### Artifact layout

Each agent run writes artifacts to a stable directory:

```
artifacts/agent_runs/<run_id>/
├── agent_trace.json      ← full step-by-step execution trace
├── agent_config.yaml
├── task.yaml
├── tool_results.json
├── scene_snapshot.json
├── run_result.json
└── logs/
```

### Testing

Real API tests are separated from the default test suite using pytest markers:

| Marker | Requires |
|---|---|
| `llm` | Live LLM API key |
| `remote_agent` | Running hosted agent |
| `agent_integration` | MCP + Blender + external API |

```bash
# Default run — no external APIs needed:
pytest

# Run with a real LLM:
ANTHROPIC_API_KEY=sk-ant-... pytest -m llm
```

### Agent CLI

```bash
# Run an agent on a single task:
python -m benchmark.agent.cli run \
  --task tasks/geometry/geometry_001_basic_primitives.yaml \
  --agent-config configs/agents/react_anthropic.yaml \
  --output-dir artifacts/

# Print a trace summary:
python -m benchmark.agent.cli trace-summary \
  --trace artifacts/agent_runs/<run_id>/agent_trace.json

# List available strategies and providers:
python -m benchmark.agent.cli list-strategies
python -m benchmark.agent.cli list-providers
```

Agent configs are in [`configs/agents/`](configs/agents/). Full documentation is in [docs/agent_runtime.md](docs/agent_runtime.md).

## Stage 7: Trace Metrics and Benchmark Reporting

Stage 7 adds the `benchmark.analysis` package, which converts raw run artifacts
(`agent_trace.json`, `validation_result.json`, `run_result.json`) into structured
metrics, comparison tables, and human-readable reports.

Full documentation is in [docs/benchmark_reporting.md](docs/benchmark_reporting.md).

### Analysis package

The `benchmark.analysis` package is organized into the following modules:

| Module | Responsibility |
|---|---|
| `tool_metrics` | Per-tool and aggregate tool-call statistics |
| `agent_metrics` | Step-level agent behaviour (LLM calls, retries, tokens) |
| `validation_metrics` | Validator scores, issue counts, per-validator breakdowns |
| `error_taxonomy` | Classify and aggregate errors from traces and validation results |
| `comparison` | Group runs by dimension, rank results, build experiment summaries |
| `run_analysis` | Combine all metrics into a single `RunAnalysisResult` |
| `report_builder` | Build Markdown and HTML reports from `ExperimentAnalysisResult` |
| `export` | Write JSON, CSV, Markdown, and HTML files |
| `cli` | Command-line interface for analysis and report generation |

### Tool-call metrics

Extracted from `AgentTrace` by `compute_tool_summary()`. Key metrics include
`tool_call_count`, `unique_tool_count`, `invalid_tool_call_count`,
`disabled_tool_call_count`, `tool_error_count`, `inspection_tool_count`,
`mutation_tool_count`, `python_tool_call_count`, `tool_repetition_count`, and
`average_tool_duration_sec`. Tool categories follow the `ToolCategory` enum from
`benchmark.mcp.tool_contract`.

### Agent metrics

Extracted by `compute_agent_summary()`. Key metrics include `llm_call_count`,
`planning_step_count`, `observation_count`, `final_step_present`, `retry_count`,
`step_limit_reached`, `self_correction_attempts`, `tool_error_recovery_count`,
`error_count`, `prompt_tokens`, `completion_tokens`, and `total_tokens`.

### Validation metrics

Extracted by `compute_validation_summary()`. Covers `scene_total_score`,
`scene_overall_status`, passed/failed/skipped validator counts, error and
warning counts, and per-validator scores (`object_score`, `transform_score`,
`material_score`, `light_score`, `camera_score`, `export_score`).

### Error taxonomy

Errors are classified into 16 categories (`ErrorCategory`) covering tool errors
(disabled, unknown, invalid arguments, runtime), LLM errors (parse, timeout),
agent errors (step limit), scene validation mismatches (object, transform,
material, light, camera, export), and connectivity errors.

```python
from benchmark.analysis.error_taxonomy import extract_errors, aggregate_errors

errors = extract_errors(trace)             # list[ErrorRecord]
counts = aggregate_errors(bundle)          # dict[str, int] — category → count
```

### Comparative reports

Runs can be grouped and compared along eight dimensions: `strategy`, `model`,
`mcp_profile`, `run`, `agent_id`, `task_category`, `difficulty`, and
`remote_provider`. Each group exposes `run_count`, `success_rate`, `avg_score`,
`avg_tool_calls`, and `avg_duration_sec`. Runs and groups can be ranked by
score, success rate, or time efficiency (`score / duration`).

```python
from benchmark.analysis.comparison import analyze_experiment, compare_runs
from benchmark.analysis.models import ComparisonDimension

analysis = analyze_experiment(Path("artifacts/experiments/exp_001"))
report   = compare_runs(analysis.runs, ComparisonDimension.STRATEGY)
```

### Markdown/HTML/CSV/JSON exports

```python
from benchmark.analysis.export import (
    write_experiment_analysis_json,
    write_run_metrics_csv,
    write_group_comparison_csv,
    write_error_taxonomy_csv,
)
from benchmark.analysis.report_builder import build_markdown_report, build_html_report
```

Report generation is controlled by `ReportConfig`, which selects output formats
(`json`, `csv`, `markdown`, `html`) and toggles report sections
(`include_runs`, `include_group_comparison`, `include_error_taxonomy`,
`include_artifact_links`).

### CLI examples

```bash
# Analyse a single run directory:
python -m benchmark.analysis.cli analyze-run \
    --run-dir artifacts/runs/run_001

# Analyse all runs under an experiment directory:
python -m benchmark.analysis.cli analyze-experiment \
    --experiment-dir artifacts/experiments/exp_001

# Build reports from a YAML config (JSON + CSV + Markdown + HTML):
python -m benchmark.analysis.cli build-report \
    --config configs/reports/default_report.yaml \
    --input  artifacts/experiments/exp_001 \
    --output reports/exp_001

# Compare runs grouped by a dimension (printed to stdout):
python -m benchmark.analysis.cli compare \
    --input    artifacts/experiments/exp_001 \
    --group-by strategy
```

Analysis and report generation can also be triggered automatically after an
experiment run:

```bash
# Run experiment then analyse:
python -m benchmark.runner.cli experiment \
    --config configs/example_experiment.yaml \
    --analyze

# Run experiment then build Markdown and HTML reports:
python -m benchmark.runner.cli experiment \
    --config configs/example_experiment.yaml \
    --report
```
