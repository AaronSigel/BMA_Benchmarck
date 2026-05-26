# Scene Validation

Stage 3 adds an offline validation layer for benchmark runs. Its job is to
answer one question: how closely does a captured Blender scene match the
formal `expected_scene` in a `BenchmarkTask`?

The validation layer works with Pydantic models and JSON files only. It does
not launch Blender and does not import Blender runtime modules.

## Scope

Scene validation compares:

```text
BenchmarkTask.expected_scene
        +
SceneSnapshot
        ↓
SceneValidator
        ↓
validation_result.json
```

Included checks:

- object validation: expected objects exist and have the expected type and primitive hint;
- transform validation: expected object `location`, `rotation`, `scale`, and `dimensions`;
- material validation: material existence, `base_color`, `roughness`, `metallic`, and object material slots;
- light validation: light existence, type, location, rotation, and energy;
- camera validation: camera existence, location, rotation, focal length against `lens`, and active camera;
- export validation: expected artifact files such as `result.blend` and `exports/result.glb`;
- JSON output as `validation_result.json`.

Not included:

- MCP integration;
- LLM execution;
- agent runtime or orchestration;
- render or visual similarity;
- full report generation.

## Validate A Task And Snapshot

Run validation with:

```bash
python -m benchmark.validation.cli validate \
  --task tasks/geometry/geometry_001_basic_primitives.yaml \
  --snapshot artifacts/blender_smoke/scene_snapshot.json \
  --artifacts-dir artifacts/blender_smoke \
  --output artifacts/blender_smoke/validation_result.json
```

`--artifacts-dir` is used by export validation. It should point at the run
directory containing files such as `result.blend` and `exports/result.glb`.

The command prints a compact summary:

```text
task_id: geometry_001_basic_primitives
overall_status: passed
total_score: 1.000
issues: 0
Saved validation result: artifacts/blender_smoke/validation_result.json
```

Read an existing result with:

```bash
python -m benchmark.validation.cli summary \
  --result artifacts/blender_smoke/validation_result.json
```

If an input file is missing or invalid, the CLI prints an `ERROR:` message and
returns a non-zero exit code. A failed scene validation still writes a result
file and returns success for the CLI command, because the validation itself ran
successfully.

## Result Format

`validation_result.json` is a serialized `SceneValidationResult`:

```json
{
  "task_id": "geometry_001_basic_primitives",
  "overall_status": "passed",
  "total_score": 1.0,
  "validators": [
    {
      "name": "object_validator",
      "status": "passed",
      "score": 1.0,
      "max_score": 1.0,
      "issues": [],
      "metrics": [
        {
          "name": "object_existence_score",
          "score": 1.0,
          "weight": 0.6,
          "passed": true,
          "issues": []
        }
      ]
    }
  ],
  "issues": [],
  "check_table": [
    {
      "validator_name": "object_validator",
      "check_name": "object exists",
      "entity_ref": "Cube",
      "field": "object",
      "expected": "Cube",
      "actual": "Cube",
      "passed": true,
      "score": 1.0
    }
  ],
  "summary": {
    "validators_total": 6,
    "validators_run": 2,
    "validators_skipped": 4,
    "validators_passed": 2,
    "validators_failed": 0,
    "issues_total": 0,
    "error_count": 0,
    "weights": {
      "object_validator": 0.7,
      "transform_validator": 0.3
    }
  }
}
```

`check_table` is optional for backward compatibility and is omitted in older
validation artifacts. New validation runs populate it with compact,
explainable rows for object, transform, material, light, camera, export and
GLB import-back checks.

Validator inventory artifacts can be generated without Blender:

```bash
python -m bma_benchmark audit-validators --tasks-dir tasks --out artifacts/validator_audit
```

Statuses are:

- `passed`;
- `failed`;
- `warning`;
- `skipped`.

Issue severities are:

- `info`;
- `warning`;
- `error`.

Every issue includes a machine-readable `code`, a human-readable `message`,
and optional `expected_path`, `actual_path`, `expected_value`, and
`actual_value`.

## Scoring

All scores are normalized to `0.0..1.0`.

Each private validator returns a `ValidatorResult`:

- `ObjectValidator`: existence, type, and primitive scores;
- `TransformValidator`: average over only the transform fields specified in the task;
- `MaterialValidator`: material existence, material parameters, and object assignment;
- `LightValidator`: existence, type, transform, and energy;
- `CameraValidator`: existence, transform, focal length, and active camera;
- `ExportValidator`: expected export files exist and are non-empty.

`SceneValidator` runs all validators and computes `total_score` as a weighted
average of non-skipped validators. Skipped validators do not lower the total
score.

Weights come from `BenchmarkTask.success_criteria` when a criterion maps to a
validator. For example:

- `object_existence` contributes to `object_validator`;
- `object_placement` contributes to `transform_validator`;
- `material_accuracy` and `parameter_correctness` contribute to `material_validator`;
- `light_existence` and `lighting_correctness` contribute to `light_validator`;
- `camera_existence` and `camera_correctness` contribute to `camera_validator`;
- `export_validity` contributes to `export_validator`.

If no success criterion maps to a validator, a default weight of `1.0` is used.

Overall status is derived as follows:

- `passed`: `total_score >= 0.85` and no error from required criteria;
- `warning`: `total_score >= 0.6`;
- `failed`: `total_score < 0.6` or an error in required criteria.

## Matching

Scene matching is intentionally heuristic. Names are normalized by lowercasing,
removing spaces, underscores, hyphens, and Blender suffixes such as `.001`.

Object matching tries:

1. expected name;
2. expected `primitive` against snapshot `primitive_hint`;
3. expected `type`.

Materials, lights, and cameras use similar name-first matching. Lights can
fall back to type. Cameras can fall back to the active camera, then the first
camera candidate.

## Known Limitations

- Matching is heuristic and can choose the wrong candidate in ambiguous scenes.
- Render similarity is not implemented.
- Complex mesh shape validation is simplified to object type, primitive hints,
  transforms, and coarse mesh metadata where available.
- Camera target visibility is represented through direction-to-target checks,
  not through image analysis.
- Material validation checks basic material parameters and assignment slots,
  not full shader graphs.

## Camera Target Validation

Expected cameras may define `target`, `require_active`, and
`direction_tolerance_deg`. When `target` is present, `CameraValidator` checks
the actual camera forward direction against the target point. Euler rotation
mismatch is not reported as long as the camera is looking at the target within
tolerance.

## Type-aware Object Counts

Scene snapshots and validation summaries expose `mesh_object_count`,
`light_count`, `camera_count`, and `all_object_count`. Contamination warnings
compare expected mesh objects, lights, and cameras separately, so expected
lights and cameras no longer appear as unexpected mesh objects.

## GLB Import-back Validation

`ExportValidator` checks file existence and size. `GlbImportBackValidator`
imports expected GLB exports into a clean temporary Blender scene, takes a
snapshot, and checks mesh count, expected names, material presence and base
colors, transforms, duplicate base names, and suspicious object counts. Its
score is reported separately as `export_import_score`.
