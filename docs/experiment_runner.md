# Experiment Runner

Stage 4 turns the separate task, Blender artifact, and validation layers into a
minimal benchmark pipeline. It runs one or more benchmark configs, validates a
`SceneSnapshot`, writes run artifacts, and exports summary metrics.

Pipeline:

```text
BenchmarkTask YAML
  -> SceneSnapshot JSON
  -> SceneValidator
  -> validation_result.json
  -> run_result.json
  -> experiment_result.json / summary.csv / summary.json / metrics.csv
```

## Stage Links

Stage 1 defines benchmark tasks in YAML and loads them as `BenchmarkTask`.
Stage 2 can produce Blender artifacts, including `scene_snapshot.json`.
Stage 3 validates a snapshot against a task and returns `SceneValidationResult`.
Stage 4 orchestrates those existing pieces into repeatable benchmark runs.

The runner can also use an already prepared snapshot. This keeps unit tests and
local replay workflows independent of Blender.

## RunConfig

One run is described by `benchmark.runner.models.RunConfig`:

```yaml
run_id: geometry_001_replay
task_id: geometry_001_basic_primitives
execution_mode: external_snapshot
task_path: tasks/geometry/geometry_001_basic_primitives.yaml
snapshot_path: artifacts/blender_smoke/scene_snapshot.json
artifacts_dir: artifacts/blender_smoke
output_dir: artifacts/runs/geometry_001_replay
metadata:
  description: Optional run metadata.
```

Fields:

- `run_id`: stable id for this run.
- `task_id`: benchmark task id.
- `execution_mode`: `external_snapshot`, `replay`, or `blender_smoke`.
- `task_path`: optional task YAML path. If omitted, `ExperimentRunner` needs a `TaskRegistry`.
- `snapshot_path`: existing snapshot path for `external_snapshot`.
- `artifacts_dir`: input artifacts directory for validation or replay.
- `output_dir`: run output directory, or a root directory where `<run_id>/` is created.
- `metadata`: free-form JSON-compatible metadata.

## ExperimentConfig

An experiment is a list of run configs:

```yaml
experiment_id: local_validation_baseline
runs:
  - run_id: geometry_001_replay
    task_id: geometry_001_basic_primitives
    execution_mode: external_snapshot
    task_path: tasks/geometry/geometry_001_basic_primitives.yaml
    snapshot_path: artifacts/blender_smoke/scene_snapshot.json
    artifacts_dir: artifacts/blender_smoke
    output_dir: artifacts/runs/geometry_001_replay
metadata:
  description: Baseline validation run using an existing scene snapshot.
```

See [configs/example_experiment.yaml](../configs/example_experiment.yaml).

## Execution Modes

`external_snapshot` uses `snapshot_path`, verifies that the file exists, and
validates it as `SceneSnapshot`. It does not require Blender.

`replay` copies `scene_snapshot.json` and the existing artifact directory into
the run output directory, then validates the copied snapshot. It is intended for
repeatable validation without launching Blender.

`blender_smoke` calls the existing Blender smoke automation through
`BlenderLauncher`. If the Blender executable is not available, the backend
returns `ok=false` with a clear error instead of importing `bpy`.

## CLI

Run one config:

```bash
python -m benchmark.runner.cli run \
  --config configs/run_geometry_001.yaml
```

Run an experiment:

```bash
python -m benchmark.runner.cli experiment \
  --config configs/example_experiment.yaml
```

The checked-in example uses `external_snapshot` and an existing lightweight
`SceneSnapshot` fixture, so it does not require Blender, MCP, or LLM access.

Summarize an existing experiment result:

```bash
python -m benchmark.runner.cli summarize \
  --results artifacts/runs/experiment_result.json
```

## Run Artifacts

`RunArtifactLayout` writes run-level files under:

```text
artifacts/runs/<run_id>/
├── scene_snapshot.json
├── validation_result.json
├── run_result.json
├── metrics.json
└── logs/
```

`run_result.json` is a serialized `RunResult` with:

- `run_id`, `task_id`, `execution_mode`
- `status`: `pending`, `running`, `passed`, `failed`, or `error`
- `validation_result_path`
- `scene_snapshot_path`
- `artifacts_dir`
- `total_score`
- `overall_status`
- `started_at`, `finished_at`, `duration_sec`
- `error`
- `summary`

## Experiment Artifacts

Batch runs write experiment-level files in the common output root:

```text
artifacts/runs/
├── experiment_result.json
├── run_results.json
├── summary.json
├── summary.csv
└── metrics.csv
```

`experiment_result.json` is a serialized `ExperimentResult`:

- `experiment_id`
- `runs`: list of `RunResult`
- `summary`: aggregate counts and score statistics

`summary.csv` has one row per run:

```text
run_id,task_id,status,execution_mode,total_score,overall_status,duration_sec,validation_result_path,scene_snapshot_path,error
```

`summary.json` stores `MetricsSummary`, including total run counts, passed,
failed and error counts, average score, min score, max score, and extracted run
metrics.

`metrics.csv` exports individual `RunMetric` rows for comparing runs.

## Out Of Scope

Stage 4 intentionally does not include:

- MCP client or Blender MCP server integration.
- LLM calls.
- Agent runtime or agent architecture.
- ReAct / Plan-and-Execute.
- Tool-call metrics.
- Token or cost tracking.
- Visual similarity metrics.
