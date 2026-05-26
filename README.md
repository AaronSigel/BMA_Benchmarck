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
- Stage 8: Experimental Matrix and E2E Benchmark Runs — matrix configs, ExperimentConfig generation, readiness checks, run manifests, smoke/baseline/API/remote-agent matrices, run-and-report workflow, and E2E CLI.

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

## Report-Ready MVP

### Назначение стенда

BMA Benchmark оценивает AI-агентов, которые выполняют Blender-задачи через MCP. Стенд фиксирует не только итоговую сцену, но и процесс tool-use: доступные инструменты, ошибки вызовов, runtime-состояние агента, validation issues и provider-reported стоимость.

### Архитектура benchmark pipeline

Pipeline состоит из матрицы эксперимента, генерации run-конфигов, запуска агента через MCP/Blender runtime, сохранения raw artifacts, scene validation, анализа запусков и генерации отчётного пакета. Основной результат report-ready запуска находится в `report_bundle/`.

### Формат задач

Задачи описаны YAML-файлами в `tasks/` и сгруппированы по категориям: geometry, materials, lighting, camera и export. Каждая задача задаёт ожидаемые объекты, трансформы, материалы, свет, камеру или export-артефакты, которые затем проверяются валидаторами.

### Стратегии агента

Поддерживаются `direct_tool_calling`, `plan_and_execute` и `react`. Для report-ready MVP основным рабочим режимом обычно считается наиболее устойчивый режим по фактическому `reported_success_rate`; `react` сохраняется как диагностическая стратегия для изучения ограничений многошагового agent loop.

### MCP-профили

Матрица сравнивает профили `minimal`, `no_python`, `inspection_enabled`, `python_enabled` и `full`. Они отражают разные поверхности доступных MCP-инструментов и позволяют оценить влияние tool gating на устойчивость выполнения.

### Метрики и статусы

Главный отчётный статус находится в колонке `pass_type`:

| pass_type | Meaning |
|---|---|
| `clean_pass` | Сцена прошла validation без замечаний. |
| `soft_pass` | Запуск прошёл по score, но содержит validation issues. |
| `failed_validation` | Финальная сцена получена, но validation failed. |
| `runtime_error` | Агент, tool runtime или инфраструктура не дошли до корректной финальной сцены. |

`reported_success_rate = (clean_pass + soft_pass) / total_runs`, `strict_success_rate = clean_pass / total_runs`, `failure_rate = (failed_validation + runtime_error) / total_runs`.

### Основной report-ready запуск

```bash
python -m bma_benchmark run-matrix \
  --config configs/matrices/diagnostic_repeat_gemini_v5.yaml
```

Матрица `diagnostic_repeat_gemini_v5` рассчитана на 720 запусков: 18 задач, 4 стратегии, 5 MCP-профилей, 1 модель и 2 повторности. Для неё включён `report_ready_mvp`, поэтому отчёты и `report_bundle/` создаются автоматически.

### Стабилизированный workflow

Перед большим запуском можно сохранить отдельный preflight-артефакт:

```bash
python -m bma_benchmark preflight \
  --config configs/matrices/diagnostic_repeat_gemini_v5.yaml
```

Для возобновления прерванной матрицы используйте `--resume`. Уже завершённые run-директории с валидным `artifact_manifest.json` будут пропущены, неполные или повреждённые — перезапущены.

```bash
python -m bma_benchmark run-matrix \
  --config configs/matrices/diagnostic_repeat_gemini_v5.yaml \
  --resume
```

Анализ и отчёт можно пересобрать из raw artifacts без повторного запуска LLM:

```bash
python -m bma_benchmark analyze --input artifacts/experiments/<run>
python -m bma_benchmark build-report --input artifacts/experiments/<run>
python -m bma_benchmark validate-report-bundle artifacts/experiments/<run>/report_bundle
```

`build-report` также добавляет в `report_bundle/` доказательные артефакты валидации и примеров сцен:

```text
report_bundle/validator_audit/
report_bundle/scene_examples/
```

Их можно пересобрать отдельно:

```bash
python -m bma_benchmark audit-validators --tasks-dir tasks --out artifacts/validator_audit

python -m bma_benchmark build-scene-gallery \
  --input artifacts/experiments/<run> \
  --out artifacts/experiments/<run>/report_bundle/scene_examples
```

`validate-report-bundle` возвращает exit code `0` только для валидного пакета. Результат проверки сохраняется в:

```text
report_bundle/bundle_validation_result.json
report_bundle/bundle_validation_result.md
```

Дополнительные эксплуатационные команды:

```bash
python -m bma_benchmark list-strategies
python -m bma_benchmark compare-bundles run_a/report_bundle run_b/report_bundle
```

Для локальной проверки report pipeline без OpenRouter, Blender и MCP socket есть offline-матрица:

```bash
python -m bma_benchmark run-matrix \
  --config configs/matrices/mock_report_ready.yaml
```

Каждый run содержит обязательные `run_result.json`, `artifact_manifest.json` и `metrics.json`. Для `agent_mcp` и `remote_agent` запусков обязателен `agent_trace.json`; если сбой произошёл до agent loop, создаётся stub trace со structured error. Отсутствующие optional artifacts объясняются marker-файлами: `scene_snapshot_not_available.json`, `validation_result_not_available.json`, `exports_not_available.json`.

Runtime ошибки нормализуются в единый contract:

```json
{
  "error_type": "SnapshotUnavailable",
  "message": "pre-run scene snapshot could not be collected",
  "source": "blender",
  "recoverable": true,
  "failure_stage": "pre_run_snapshot",
  "raw_error": "pre-run scene snapshot could not be collected"
}
```

`summary.csv` включает `error_type`, `error_source` и `failure_stage`. `UnknownError` не используется для известных failure patterns; редкий fallback называется `UnclassifiedError`.

При `--resume` дополнительно создаются:

```text
resume_report.json
resume_report.md
```

`report_bundle/run_artifact_manifests.json` содержит агрегированную сводку по run manifests: число runs, количество complete manifests, суммарные missing required artifacts и краткую запись по каждому run. Общий `manifest.json` содержит версии протокола и hashes конфигурации, task set, tool contract и report config.

### Как читать report_bundle

`summary.csv` является главным источником данных для таблиц. `experiment_analysis.json` и `summary.json` содержат машинно-читаемые агрегаты. `report.md` и `report.html` содержат таблицы, key findings и diagnostics. `report_text_ru.md` содержит готовый русский текст анализа. `README_REPORT.md` объясняет назначение файлов и интерпретацию статусов. PNG-графики находятся в `figures/`.

### Ограничения MVP

Report-ready MVP использует одну модель в основной матрице. ReAct сохраняется как диагностическая стратегия. Export и Lighting остаются сложными категориями. Результаты зависят от доступности OpenRouter, MCP socket, Blender runtime и provider-reported cost. Стоимость берётся только из OpenRouter provider-reported данных, без внутренней подстановки цены.

### Дальнейшее развитие

Следующие шаги: добавить повторности для статистической устойчивости, расширить набор моделей, усилить export/import-back диагностику, стабилизировать ReAct loop, добавить confidence intervals и подготовить cross-run сравнение нескольких report bundles.

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

## Stage 8: Experimental Matrix and E2E Benchmark Runs

Stage 8 closes the practical part of the benchmark stand. It adds a declarative
experimental matrix system and a full end-to-end pipeline from matrix config to
analysis report.

Full documentation is in [docs/experimental_matrix.md](docs/experimental_matrix.md).

### Matrix configs

Experiment matrices live in `configs/matrices/`. Each YAML file describes a
Cartesian product of benchmark axes:

```
model / remote-agent
    × agent strategy
    × MCP profile
    × task category
    × difficulty
    × repetitions
```

| File | Purpose |
|---|---|
| `smoke_matrix.yaml` | Fast local check — no API, no Blender |
| `baseline_matrix.yaml` | Main series: Direct vs ReAct vs Plan-and-Execute |
| `api_models_matrix.yaml` | Opt-in: compare API providers (OpenRouter, Anthropic, …) |
| `remote_agents_matrix.yaml` | Opt-in: external server-side agents |

### ExperimentConfig generation

`generate_experiment_config(matrix)` expands the matrix into a flat list of
`RunConfig` objects. Each run gets a stable ID of the form:

```
<matrix_id>__<task_id>__<agent_id>__<mcp_profile>__r<repetition>
```

```bash
python -m benchmark.experiments.cli generate \
  --matrix configs/matrices/smoke_matrix.yaml \
  --output experiment.yaml
```

### Readiness checks

Before any batch run the system validates the environment:

```bash
python -m benchmark.experiments.cli readiness \
  --matrix configs/matrices/baseline_matrix.yaml \
  --output readiness.json
```

Checks include: task IDs found, agent and MCP configs present, API key env vars
set (warning if missing, error if `strict_readiness: true`), Blender executable
available (for `agent_mcp` mode), and MCP socket reachable. Missing required
items abort the run before any API call is made.

### Run manifests

At the start of every batch run `manifest.json` is written to `output_root/`.
The manifest records git commit, Python version, platform, config hash (SHA-256
of the sanitised matrix dump), environment requirements, and readiness outcome.
Secret keys are stripped automatically. The config hash lets you verify that two
runs used identical configurations.

### Smoke matrix

The smoke matrix runs without external services, API keys, or Blender:

```bash
python -m benchmark.experiments.cli run-and-report \
  --matrix configs/matrices/smoke_matrix.yaml
```

Uses `execution_modes: [external_snapshot]` and `provider: mock`. Suitable for
CI and local development.

### Baseline, API, and remote-agent matrices

**Baseline** — requires Blender + MCP + provider API keys:

```bash
python -m benchmark.experiments.cli run-and-report \
  --matrix configs/matrices/baseline_matrix.yaml
```

**API models** (opt-in) — paid API quota consumed, run outside CI:

```bash
python -m benchmark.experiments.cli run-and-report \
  --matrix configs/matrices/api_models_matrix.yaml
```

**Remote agents** (opt-in) — requires a configured external agent runtime:

```bash
python -m benchmark.experiments.cli run-and-report \
  --matrix configs/matrices/remote_agents_matrix.yaml
```

### run-and-report workflow

`run-and-report` executes the full pipeline in one command:

1. Load and validate matrix
2. Readiness check — abort on errors
3. Generate `ExperimentConfig`
4. Write `manifest.json`
5. Run all experiments via `BatchRunner`
6. Write `experiment_result.json`
7. Analyse results → `experiment_analysis.json`, `metrics.csv`, `summary.*`
8. Build `report.md` and `report.html`

### Test markers

| Marker | Requires |
|---|---|
| `api_e2e` | Live API keys (api_models_matrix) |
| `remote_agent_e2e` | Running remote agent (remote_agents_matrix) |
| `llm` | Live LLM API key |
| `mcp` | Running Blender + MCP server |
| `blender` | Blender executable |

```bash
# Default — no external services:
pytest

# Run E2E with real APIs:
OPENROUTER_API_KEY=... pytest -m api_e2e
```

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

### Pilot comparison v3

Run the existing five-category pilot:

```bash
python -m benchmark.experiments.cli run-and-report \
  --matrix configs/matrices/pilot_5category_openrouter_v2.yaml \
  --clean-output
```

Run the comparison matrix:

```bash
python -m benchmark.experiments.cli run-and-report \
  --matrix configs/matrices/pilot_comparison_openrouter_v3.yaml \
  --clean-output
```

`pilot_comparison_openrouter_v3` runs `model x strategy x MCP profile x task x
repetition`: `3 x 3 x 2 x 5 x 3 = 270` runs. It uses only OpenRouter model ids
and does not add Ollama, local models, or GigaChat. The matrix model id
overrides the base model in each agent config.

`summary.json` is the runner aggregate. `experiment_analysis.json` is the
analysis artifact for reports and papers. Per run it includes model, strategy,
MCP profile, total score, validation status, validator counts,
`validation_coverage`, tool-call metrics, token usage when available, estimated
cost, and issues.

`validation_coverage = validators_run / validators_total`. A high score with
low coverage means the result passed only a partial validation surface.

`export_score` checks that the expected file exists and is non-empty.
`export_import_score` imports an exported GLB into a clean temporary Blender
scene and checks that object structure, materials, transforms, duplicate names,
and coarse scene bounds still match expectations.

`no_python` is the primary MCP-tool-use regime: agents can use structured
benchmark-safe `bma_*` tools, but cannot solve tasks by executing arbitrary
Blender Python.

## Report-ready MVP run

The main report-ready benchmark entrypoint is:

```bash
python -m bma_benchmark run-matrix \
  --config configs/matrices/diagnostic_repeat_gemini_v5.yaml
```

This matrix runs 18 Blender tasks across four strategies
(`direct`, `plan_and_execute`, `react`, `plan_execute_react_repair`), five MCP profiles
(`minimal`, `no_python`, `inspection_enabled`, `python_enabled`, `full`), one
OpenRouter model (`google/gemini-2.5-flash-lite`), and two repetitions: 720 runs
total. The command creates a timestamped output directory:

```text
artifacts/diagnostic_repeat_gemini_v5_<timestamp>/report_bundle/
```

The benchmark pipeline is:

```text
matrix config -> run configs -> agent/MCP execution -> validation
-> per-run artifacts -> analysis summary -> report bundle
```

`summary.csv` in `report_bundle` is the primary source for tables and
post-processing. `report.md` and `report.html` contain ready-to-copy tables and
key findings. `report_text_ru.md` contains Russian scientific/technical prose
for insertion into a report. `figures/*.png` contains static charts suitable
for document insertion.

### Metrics and statuses

The main reporting status is `pass_type`:

| pass_type | Meaning |
| --- | --- |
| `clean_pass` | Scene passed validation without validation issues. |
| `soft_pass` | Scene passed by score/status but has validation issues. |
| `failed_validation` | Agent completed and a scene is available, but validation failed. |
| `runtime_error` | The run did not reach a correct final scene because of an agent, tool, or runtime error. |

Derived rates use the same formulas in CSV, JSON, Markdown, and HTML:

```text
reported_success_rate = (clean_pass + soft_pass) / total_runs
strict_success_rate = clean_pass / total_runs
failure_rate = (failed_validation + runtime_error) / total_runs
```

Technical fields such as `run_status`, `scene_status`, and `agent_status`
remain in `summary.csv`, but report tables use `pass_type`.

### Strategies and MCP profiles

`direct_tool_calling` issues tool calls directly, `plan_and_execute` separates
planning from execution, and `react` uses an iterative reasoning/action loop.
ReAct is evaluated in validator-guided repair mode; when reported success is high
(>85%), it is treated as a working diagnostic/repair contour rather than a broken strategy.

MCP profiles define the available tool surface. `minimal` is the smallest
surface, `no_python` disables arbitrary Python while keeping structured BMA
tools, `inspection_enabled` adds scene inspection, `python_enabled` enables
Python-oriented tools, and `full` exposes the broadest profile.

### Experimental scope

The primary experimental contour is limited to API-based LLM backends (OpenRouter).
Claude Code / Codex CLI are treated as an experimental remote-agent extension and
are not part of the main benchmark matrix. `generation_profile` is fixed per matrix;
decoding-parameter sweeps (`top_p`, `top_k`, `temperature`) are outside the main
experiment. Diagnostic matrices use `repetitions: 2` to balance cost, duration,
and run-to-run stability assessment.

### OpenRouter cost

Cost is reported only from provider-reported OpenRouter usage fields. Internal
token-to-price estimation formulas are not used for report totals.

### MVP limitations

The MVP focuses on producing a reproducible report package, not on maximizing
success rate. It intentionally preserves real failures from agents, strategies,
tools, validators, and MCP profiles as classified benchmark results. Infra failures
(Blender socket/runtime) are reported separately from model and validation failures.
Multi-model comparison beyond configured matrices, render similarity, visual feedback loops,
and human-in-the-loop workflows are outside this MVP scope.
