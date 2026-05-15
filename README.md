# Blender Agent Benchmark

This project is a benchmark stand for evaluating AI agents that work with Blender through MCP.

Current stage: formalizing benchmark tasks and preparing the Blender automation layer. The repository contains Pydantic models, a JSON Schema, YAML task files, a loader, a registry, a validator, CLI helpers, and pytest coverage for the task registry format.

MCP server integration and LLM execution are intentionally out of scope for the current stage.

## Task Format

The benchmark task YAML format is documented in [docs/task_format.md](docs/task_format.md).

The current task inventory is listed in [TASKS.md](TASKS.md).

Scene validation usage and result format are documented in
[docs/scene_validation.md](docs/scene_validation.md).

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

## Stage 2: Blender Automation

Stage 2 adds the dependency and configuration surface for running Blender in headless mode from the benchmark package. The goal of this stage is to launch Blender, execute Blender Python scripts inside that process, and save formal artifacts such as scene snapshots under `artifacts/`.

Full usage notes are documented in [docs/blender_automation.md](docs/blender_automation.md).

The optional `blender` extra installs Pillow for image-related artifact handling and NumPy for Blender's glTF exporter:

```bash
.venv/bin/python -m pip install -e '.[blender]'
```

The project does not install `bpy` as a pip dependency. Code that needs Blender's Python API must run inside a Blender process, and regular pytest tests must keep passing on machines where the Blender executable is not installed. Blender-specific tests should use the `blender` marker, and broader end-to-end checks should use the `integration` marker.

Stage 2 does not include MCP, LLM execution, or agent architectures.
