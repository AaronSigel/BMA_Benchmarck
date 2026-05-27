from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from bma_benchmark.reporting.scene_examples.image_finder import find_scene_images
from bma_benchmark.reporting.scene_examples.models import RunArtifactRef


def discover_runs(experiment_dir: Path) -> list[RunArtifactRef]:
    root = Path(experiment_dir)
    summary_rows = _summary_rows(root / "summary.csv")
    candidates = _candidate_dirs(root)
    refs = [_build_ref(path, summary_rows.get(path.name, {})) for path in sorted(candidates)]
    by_id: dict[str, RunArtifactRef] = {}
    for ref in refs:
        by_id[ref.run_id] = ref
    return list(by_id.values())


def _candidate_dirs(root: Path) -> set[Path]:
    found: set[Path] = set()
    for name in ("run_result.json", "artifact_manifest.json", "metrics.json"):
        for path in root.rglob(name):
            if path.parent != root:
                found.add(path.parent)
    run_results = _read_json(root / "run_results.json")
    if isinstance(run_results, list):
        for item in run_results:
            if isinstance(item, dict):
                run_dir = item.get("run_dir") or item.get("artifacts_dir")
                if run_dir:
                    p = Path(run_dir)
                    if p.is_absolute():
                        found.add(p)
                    else:
                        direct = root / p.name
                        nested = root / p
                        if direct.exists():
                            found.add(direct)
                        elif nested.exists():
                            found.add(nested)
                        else:
                            found.add(nested)
    return found


def _build_ref(run_dir: Path, summary: dict[str, str]) -> RunArtifactRef:
    run_result = _read_json(run_dir / "run_result.json") or {}
    manifest = _read_json(run_dir / "artifact_manifest.json") or {}
    metrics = _read_json(run_dir / "metrics.json") or {}
    validation = _read_json(run_dir / "validation_result.json") or {}
    render_path, viewport_path, missing_reason = find_scene_images(run_dir)
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else {}
    files = manifest.get("files") if isinstance(manifest, dict) else []

    task_id = _first(summary.get("task_id"), run_result.get("task_id"), manifest.get("task_id"))
    pass_type = _first(summary.get("pass_type"), manifest.get("status"), _pass_type_from_run(run_result, validation))

    return RunArtifactRef(
        run_id=str(_first(summary.get("run_id"), run_result.get("run_id"), manifest.get("run_id"), run_dir.name)),
        run_dir=run_dir,
        task_id=str(task_id) if task_id else None,
        category=str(_first(summary.get("category"), _category_from_task(task_id))) if task_id else None,
        model=_first(summary.get("model"), manifest.get("model"), _nested(run_result, "summary", "model")),
        strategy=_first(summary.get("strategy"), manifest.get("strategy"), _nested(run_result, "summary", "strategy")),
        mcp_profile=_first(summary.get("mcp_profile"), manifest.get("mcp_profile"), _nested(run_result, "summary", "mcp_profile")),
        pass_type=str(pass_type) if pass_type else None,
        scene_score=_float(_first(summary.get("score"), summary.get("total_score"), summary.get("scene_score"), run_result.get("total_score"), validation.get("total_score"))),
        strict_success=_bool(_first(summary.get("strict_success"), run_result.get("status") == "passed" if run_result else None)),
        snapshot_path=_existing_path(run_dir, artifacts, "scene_snapshot", "scene_snapshot.json"),
        validation_result_path=_existing_path(run_dir, artifacts, "validation_result", "validation_result.json"),
        render_path=render_path,
        viewport_path=viewport_path,
        blend_path=_find_file(run_dir, files, ".blend") or _first_existing([
            run_dir / "final_scene.blend",
            run_dir / "result.blend",
        ]),
        glb_path=_find_file(run_dir, files, ".glb") or _first_existing([run_dir / "exports/result.glb", run_dir / "result.glb"]),
        artifact_manifest=manifest if isinstance(manifest, dict) else {},
        run_result=run_result if isinstance(run_result, dict) else {},
        metrics=metrics if isinstance(metrics, dict) else {},
        validation_result=validation if isinstance(validation, dict) else {},
        render_missing_reason=missing_reason,
    )


def _summary_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except Exception:  # noqa: BLE001
        return {}
    return {row.get("run_id", ""): row for row in rows if row.get("run_id")}


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _existing_path(run_dir: Path, artifacts: Any, key: str, fallback: str) -> Path | None:
    if isinstance(artifacts, dict) and isinstance(artifacts.get(key), dict):
        rel = artifacts[key].get("path")
        if rel and (run_dir / str(rel)).is_file():
            return run_dir / str(rel)
    path = run_dir / fallback
    return path if path.is_file() else None


def _find_file(run_dir: Path, files: Any, suffix: str) -> Path | None:
    if isinstance(files, list):
        for name in files:
            path = run_dir / str(name)
            if path.suffix.lower() == suffix and path.is_file():
                return path
    matches = sorted(run_dir.rglob(f"*{suffix}"))
    return matches[0] if matches else None


def _first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def _first(*values: Any) -> Any:
    return next((value for value in values if value not in (None, "")), None)


def _nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no"}:
            return False
    return None


def _category_from_task(task_id: Any) -> str | None:
    if not isinstance(task_id, str) or "_" not in task_id:
        return None
    return task_id.split("_", 1)[0]


def _pass_type_from_run(run_result: dict[str, Any], validation: dict[str, Any]) -> str | None:
    if run_result.get("pass_type"):
        return str(run_result["pass_type"])
    from benchmark.analysis.pass_type_rules import apply_export_pass_type_guard

    status = str(run_result.get("status") or "").lower()
    val_status = str(validation.get("overall_status") or run_result.get("overall_status") or "").lower()
    task_id = str(run_result.get("task_id") or validation.get("task_id") or "")
    issues = [
        issue if isinstance(issue, dict) else {"code": getattr(issue, "code", "")}
        for issue in validation.get("issues") or []
    ]
    scores = validation.get("summary", {}).get("scores", {}) if isinstance(validation.get("summary"), dict) else {}
    if status == "passed" and val_status == "passed":
        pass_type = "clean_pass"
    elif status == "passed" and val_status == "warning":
        pass_type = "soft_pass"
    elif val_status == "failed":
        pass_type = "failed_validation"
    else:
        return status or None
    return apply_export_pass_type_guard(
        pass_type,
        task_id,
        issues,
        object_score=scores.get("scene_score"),
        export_score=scores.get("export_score"),
        import_back_score=scores.get("import_back_score"),
    )
