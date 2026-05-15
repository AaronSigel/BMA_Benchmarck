# Blender Agent Benchmark

This project is a benchmark stand for evaluating AI agents that work with Blender through MCP.

Current stage: formalizing benchmark tasks only. The repository contains Pydantic models, a JSON Schema, YAML task files, a loader, a registry, a validator, CLI helpers, and pytest coverage for the task registry format.

Blender, MCP server integration, and LLM execution are intentionally out of scope for this stage.

## Task Format

The benchmark task YAML format is documented in [docs/task_format.md](docs/task_format.md).

## Common Commands

Install the project in a virtual environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
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
