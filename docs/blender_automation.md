# Blender Automation

## Purpose

Stage 2 adds a Python automation layer for running Blender in headless mode and collecting formal scene artifacts. It bridges the benchmark package and Blender's Python API without introducing MCP, LLM execution, or agent strategies.

The main flow is:

```text
benchmark package
  -> BlenderLauncher
  -> Blender headless process
  -> Blender Python script
  -> scene artifact
```

## Scope

Stage 2 includes:

- Headless Blender launch through `subprocess.run`.
- Fixture scene creation for smoke tests.
- `scene_snapshot.json` collection.
- Saving `.blend` files.
- PNG rendering.
- Scene export to `.blend`, `.glb`, `.gltf`, and `.fbx` when the Blender operator is available.
- Artifact path helpers for repeatable output layout.

Stage 2 does not include:

- MCP.
- LLM execution.
- Agent runtime.
- Full scene validator.
- Metrics engine.

## Dependencies

`bpy` is not a pip dependency for this project. Blender provides `bpy` inside its own Python process, so project modules must not import `bpy` at normal Python import time.

Install regular test dependencies:

```bash
.venv/bin/python -m pip install -e '.[test]'
```

Install optional Blender automation dependencies:

```bash
.venv/bin/python -m pip install -e '.[blender]'
```

This extra includes `numpy`, which Blender's bundled glTF exporter may need when exporting `.glb` or `.gltf` files. Blender scripts add the active virtual environment's `site-packages` to Blender Python's `sys.path` when `VIRTUAL_ENV` is set.

## Blender Executable

The launcher looks for Blender in this order:

1. `BMA_BLENDER_BIN`
2. `shutil.which("blender")`

To set the executable explicitly:

```bash
export BMA_BLENDER_BIN=/path/to/blender
```

Check discovery without launching a scene:

```bash
.venv/bin/python -m benchmark.blender.cli check
```

## CLI Commands

All commands use `argparse` and write artifacts under the provided output directory.

```bash
.venv/bin/python -m benchmark.blender.cli fixture --output-dir artifacts/blender_smoke
```

Creates a fixture scene and saves `result.blend`.

```bash
.venv/bin/python -m benchmark.blender.cli snapshot --output-dir artifacts/blender_smoke
```

Collects `scene_snapshot.json` from the active Blender scene.

```bash
.venv/bin/python -m benchmark.blender.cli render --output-dir artifacts/blender_smoke
```

Writes `render.png`.

```bash
.venv/bin/python -m benchmark.blender.cli export --format glb --output-dir artifacts/blender_smoke
```

Writes `exports/result.glb`.

```bash
.venv/bin/python -m benchmark.blender.cli smoke --output-dir artifacts/blender_smoke
```

Runs one Blender process and performs:

1. `reset_scene`
2. `create_fixture_scene`
3. `collect_snapshot`
4. `save_scene`
5. `render_scene`
6. `export_scene` as GLB

Expected smoke artifacts:

```text
artifacts/blender_smoke/
├── scene_snapshot.json
├── result.blend
├── render.png
├── exports/
│   └── result.glb
├── smoke_input.json
└── smoke_output.json
```

## Artifact Layout

`ArtifactLayout` defines the future runner layout:

```text
artifacts/runs/<run_id>/
├── input.json
├── <command>.output.json
├── scene_snapshot.json
├── result.blend
├── render.png
├── exports/
│   └── result.<format>
└── logs/
    ├── <command>.stdout.log
    └── <command>.stderr.log
```

The layout helper only computes paths unless `ensure()` is called. `ensure()` creates the run directory, `exports/`, and `logs/`.

## Scene Snapshot

`scene_snapshot.json` is the stable interchange format between Blender automation and future validation code. It is compatible with `benchmark.blender.models.SceneSnapshot`.

The snapshot contains:

- `scene_name`
- `objects`
- `materials`
- `lights`
- `cameras`
- `collections`
- `render_settings`
- `frame_current`
- `blender_version`
- `created_at`

Object entries include transform, dimensions, material slot names, parent name, collection names, and mesh vertex/polygon counts when available.

Material entries include base color, roughness, metallic, and `use_nodes`.

Light and camera entries include transforms plus Blender-specific data such as light energy, light color, camera lens, sensor width, and active camera state.

## Tests

Run all regular tests:

```bash
.venv/bin/python -m pytest
```

Blender integration tests are marked with `blender` and `integration`. They skip automatically if Blender is not found.

```bash
.venv/bin/python -m pytest -m blender
```

## Troubleshooting

### Blender executable not found

Set `BMA_BLENDER_BIN`:

```bash
export BMA_BLENDER_BIN=/path/to/blender
```

Then run:

```bash
.venv/bin/python -m benchmark.blender.cli check
```

### `No module named bpy`

This is expected if a Blender script is run with normal Python. `bpy` exists only inside Blender's Python process. Use the CLI or `BlenderLauncher` so scripts run through Blender.

### `No module named numpy` inside Blender

Install or refresh the Blender extra in the active virtual environment:

```bash
.venv/bin/python -m pip install -e '.[blender]'
```

Then run the CLI from that activated environment so `VIRTUAL_ENV` is set:

```bash
python -m benchmark.blender.cli smoke --output-dir artifacts/blender_smoke
```

### `No module named pydantic` inside Blender

Blender's Python environment is separate from the project virtual environment. Scripts executed inside Blender should avoid depending on project pip packages unless they are explicitly installed into Blender's Python. Stage 2 Blender scripts keep runtime dependencies minimal for this reason.

### Empty or missing render output

Check that the scene has an active camera. The fixture scene creates `FixtureCamera`, and `render_scene` returns a clear error if no camera is available.

### Export operator is unavailable

Some Blender builds may not provide every export operator. `export_scene` reports a clear error if `gltf` or `fbx` export is unavailable.
