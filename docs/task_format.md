# Benchmark Task Format

Benchmark tasks are YAML files that describe a scene-building objective for a future Blender MCP agent benchmark. At this stage they are formal specifications only: they are loaded, validated, indexed, and tested, but they do not execute Blender, MCP, or an LLM.

Each task is represented by the `BenchmarkTask` Pydantic model in `benchmark/tasks/models.py` and should also validate against `benchmark/schemas/task.schema.json`.

## Top-Level Fields

`schema_version`:
Format version for the task document. Current value is `"1.0"`.

`id`:
Unique task identifier. It must be a non-empty string and should start with the task category prefix, for example `geometry_001`.

`title`:
Human-readable task title.

`category`:
Task category. Allowed values are `geometry`, `materials`, `lighting`, `camera`, and `export`.

`difficulty`:
Task difficulty. Allowed values are `easy`, `medium`, and `hard`.

`prompt`:
Non-empty natural-language instruction that will later be given to an agent.

`tags`:
List of strings for search and grouping. Tags should not be empty strings.

`allowed_tools`:
List of abstract tool names the future agent is allowed to use for the task. These are benchmark-level tool names, not direct Blender Python calls.

`expected_scene`:
Structured description of the scene elements expected after task completion.

`success_criteria`:
List of weighted metrics used to evaluate the task. The model requires each weight to be in the range `0..1` and the total weight to be no greater than `1.0`; current validation expects task weights to sum close to `1.0`.

`metadata`:
Optional object with `author`, `version`, and `description`.

## Expected Scene

`expected_scene` contains five lists:

`objects`:
Expected mesh or scene objects.

`materials`:
Expected materials and material parameters.

`lights`:
Expected light sources.

`cameras`:
Expected camera setup.

`exports`:
Expected exported files.

Category-specific validation expects:

- `geometry`: `objects` is not empty.
- `materials`: `materials` is not empty and objects should reference assigned materials.
- `lighting`: `lights` is not empty.
- `camera`: `cameras` is not empty.
- `export`: `exports` is not empty.

## Scene Elements

### ExpectedObject

Fields:

- `name: str | None`
- `type: str`
- `primitive: str | None`
- `location: Vector3 | None`
- `rotation: Vector3 | None`
- `scale: Vector3 | None`
- `dimensions: Vector3 | None`
- `material: str | None`
- `tolerance: float = 0.05`

`tolerance` must be positive.

Allowed `primitive` values are `cube`, `sphere`, `cylinder`, `cone`, and `plane`.

### ExpectedMaterial

Fields:

- `name: str`
- `base_color: ColorRGBA | None`
- `roughness: float | None`
- `metallic: float | None`
- `tolerance: float = 0.05`

`base_color` uses RGBA channels in the range `0..1`.

`roughness` and `metallic` must be between `0` and `1` when provided.

### ExpectedLight

Fields:

- `name: str | None`
- `type: str`
- `location: Vector3 | None`
- `rotation: Vector3 | None`
- `energy: float | None`
- `tolerance: float = 0.05`

Allowed `type` values are `AREA`, `SUN`, `POINT`, and `SPOT`.

### ExpectedCamera

Fields:

- `name: str | None`
- `location: Vector3 | None`
- `rotation: Vector3 | None`
- `focal_length: float | None`
- `target: str | None`
- `tolerance: float = 0.05`

Camera tasks should provide `location` and at least one of `rotation` or `target`.

### ExpectedExport

Fields:

- `format: str`
- `filename: str | None`
- `must_exist: bool = true`

Allowed `format` values are `blend`, `glb`, and `fbx`. Current export tasks use `blend` and `glb` with explicit filenames.

### Vector3

Fields:

- `x: float`
- `y: float`
- `z: float`

### ColorRGBA

Fields:

- `r: float`
- `g: float`
- `b: float`
- `a: float = 1.0`

All channels must be between `0` and `1`.

## Success Criteria

Each success criterion contains:

- `metric: str`
- `weight: float`
- `required: bool = true`

Weights must be from `0` to `1`. The total task weight must not exceed `1.0`; for the curated task set, use weights that sum to `1.0`.

Common metrics by category:

- Geometry: `object_existence`, `geometry_accuracy`, `object_placement`
- Materials: `object_existence`, `material_accuracy`, `parameter_correctness`
- Lighting: `light_existence`, `lighting_correctness`, `parameter_correctness`
- Camera: `camera_existence`, `camera_correctness`, `target_visibility`
- Export: `object_existence`, `export_validity`

## ID Naming

Use the category prefix followed by a numeric sequence and a short slug:

- `geometry_001_basic_primitives`
- `materials_001_basic_colors`
- `lighting_001_area_light`
- `camera_001_front_view`
- `export_001_blend_file`

Short forms like `geometry_001`, `materials_001`, `lighting_001`, `camera_001`, and `export_001` are valid prefixes, but descriptive suffixes make the task set easier to browse.

## Full Geometry Example

```yaml
schema_version: "1.0"
id: geometry_001_basic_primitives
title: Basic primitives
category: geometry
difficulty: easy
prompt: >
  Create three basic mesh primitives: a cube at the origin, a UV sphere to the
  right, and a cylinder to the left. Use simple default materials and keep the
  scene focused on object creation and placement.
tags:
  - geometry
  - primitives
  - baseline
allowed_tools:
  - create_object
  - set_transform
  - assign_material
  - inspect_scene
expected_scene:
  objects:
    - name: Cube
      type: MESH
      primitive: cube
      location: { x: 0.0, y: 0.0, z: 0.0 }
      dimensions: { x: 2.0, y: 2.0, z: 2.0 }
      tolerance: 0.1
    - name: Sphere
      type: MESH
      primitive: sphere
      location: { x: 3.0, y: 0.0, z: 0.0 }
      tolerance: 0.1
    - name: Cylinder
      type: MESH
      primitive: cylinder
      location: { x: -3.0, y: 0.0, z: 0.0 }
      tolerance: 0.1
  materials: []
  lights: []
  cameras: []
  exports: []
success_criteria:
  - metric: object_existence
    weight: 0.4
    required: true
  - metric: geometry_accuracy
    weight: 0.3
    required: true
  - metric: object_placement
    weight: 0.3
    required: true
metadata:
  author: benchmark
  version: "1.0"
  description: Baseline geometry task for primitive creation.
```

## Adding a New Task

1. Choose the category directory under `tasks/`: `geometry`, `materials`, `lighting`, `camera`, or `export`.
2. Pick a unique `id` that starts with the category prefix, for example `geometry_006_new_shape`.
3. Create a `.yaml` file whose filename matches the task id.
4. Fill all required top-level fields: `schema_version`, `id`, `title`, `category`, `difficulty`, `prompt`, `tags`, `allowed_tools`, `expected_scene`, and `success_criteria`.
5. Fill the category-specific expected scene list.
6. Use success criteria weights that sum to `1.0`.
7. Run validation:

```bash
python -m benchmark.tasks.cli validate --tasks-dir tasks
```

8. Run tests:

```bash
pytest
```
