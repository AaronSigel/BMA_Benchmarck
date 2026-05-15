# Blender Agent Benchmark

This project is a benchmark stand for evaluating AI agents that work with Blender through MCP.

The repository currently contains benchmark task definitions, Blender artifact automation, scene validation, and an experiment runner for local-file benchmark runs. MCP server integration and LLM execution remain intentionally out of scope for the implemented pipeline.

Implemented stages:

- Stage 1: benchmark task format, task registry, schema, and task CLI.
- Stage 2: Blender automation artifacts such as `scene_snapshot.json`, renders, exports, and `.blend` files.
- Stage 3: scene validation and `validation_result.json`.
- Stage 4: experiment runner, batch runner, run artifacts, and summary metrics.

## Documentation

The benchmark task YAML format is documented in [docs/task_format.md](docs/task_format.md).

The current task inventory is listed in [TASKS.md](TASKS.md).

Blender automation usage is documented in [docs/blender_automation.md](docs/blender_automation.md).

Scene validation usage and result format are documented in
[docs/scene_validation.md](docs/scene_validation.md).

Experiment runner usage, configs, artifact layout, and summary exports are
documented in [docs/experiment_runner.md](docs/experiment_runner.md).

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
