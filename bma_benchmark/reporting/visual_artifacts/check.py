from __future__ import annotations

import csv
import json
from pathlib import Path

from bma_benchmark.reporting.scene_examples.discovery import discover_runs


def collect_visual_artifact_rows(experiment_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for ref in discover_runs(experiment_dir):
        run_dir = ref.run_dir
        has_final_blend = _exists(run_dir / "final_scene.blend") or _exists(ref.blend_path)
        has_viewport = _exists(run_dir / "viewport.png") or _exists(ref.viewport_path)
        has_final_render = _exists(run_dir / "final_render.png") or _exists(ref.render_path)
        has_glb = _exists(ref.glb_path)
        has_validation = _exists(ref.validation_result_path)
        has_snapshot = _exists(ref.snapshot_path)
        visual_ready = has_final_blend and (has_viewport or has_final_render)
        missing_reason = None
        if not visual_ready:
            if not has_final_blend:
                missing_reason = "missing final_scene.blend"
            elif not (has_viewport or has_final_render):
                missing_reason = "missing viewport/final_render image"

        rows.append(
            {
                "run_id": ref.run_id,
                "task_id": ref.task_id or "",
                "pass_type": ref.pass_type or "",
                "has_final_blend": has_final_blend,
                "has_viewport": has_viewport,
                "has_final_render": has_final_render,
                "has_glb": has_glb,
                "has_validation_result": has_validation,
                "has_scene_snapshot": has_snapshot,
                "visual_ready": visual_ready,
                "missing_reason": missing_reason or "",
            }
        )
    return rows


def write_visual_artifacts_check(experiment_dir: Path) -> tuple[Path, Path]:
    experiment_dir = Path(experiment_dir)
    rows = collect_visual_artifact_rows(experiment_dir)
    json_path = experiment_dir / "visual_artifacts_check.json"
    csv_path = experiment_dir / "visual_artifacts_check.csv"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = list(rows[0].keys()) if rows else [
        "run_id",
        "task_id",
        "pass_type",
        "has_final_blend",
        "has_viewport",
        "has_final_render",
        "has_glb",
        "has_validation_result",
        "has_scene_snapshot",
        "visual_ready",
        "missing_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path, json_path


def _exists(path: Path | None) -> bool:
    return path is not None and Path(path).is_file() and Path(path).stat().st_size > 0
