from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.analysis.trace_reader import discover_run_artifacts
from benchmark.experiments.e2e_runner import build_reports, run_analysis
from benchmark.experiments.models import ExperimentMatrix


def resolve_experiment_root(path: Path) -> Path:
    """Принимает experiment root или report_bundle и возвращает корень эксперимента."""
    resolved = path.resolve()
    if resolved.name == "report_bundle":
        return resolved.parent
    return resolved


def _agent_id_from_run_id(run_id: str) -> str | None:
    parts = run_id.split("__")
    if len(parts) < 3:
        return None
    return parts[2]


def _link_run_dir(source: Path, target: Path, *, use_symlinks: bool) -> None:
    if target.exists() or target.is_symlink():
        raise ValueError(f"target run directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if use_symlinks:
        os.symlink(source.resolve(), target, target_is_directory=True)
    else:
        shutil.copytree(source, target)


def _dedupe_run_dirs(run_dirs: list[Path]) -> list[Path]:
    """Оставляет один каталог на run_id (discover может вернуть дубликаты путей)."""
    by_name: dict[str, Path] = {}
    for run_dir in run_dirs:
        resolved = run_dir.resolve()
        existing = by_name.get(run_dir.name)
        if existing is None:
            by_name[run_dir.name] = resolved
    return sorted(by_name.values(), key=lambda path: path.name)


def merge_experiment_runs(
    *,
    base: Path | str,
    replacement: Path | str,
    output: Path | str,
    replace_agent: str,
    replacement_reason: str | None = None,
    expected_replacement_runs: int = 900,
    expected_base_non_replaced_runs: int = 2700,
    expected_total_runs: int = 3600,
    use_symlinks: bool = True,
    rebuild_reports: bool = True,
) -> dict[str, Any]:
    """Собирает merged experiment root из base (без replace_agent) и replacement runs."""
    base_root = resolve_experiment_root(Path(base))
    replacement_root = resolve_experiment_root(Path(replacement))
    output_root = Path(output).resolve()

    if output_root.exists():
        raise ValueError(f"output directory already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    base_runs = _dedupe_run_dirs(discover_run_artifacts(base_root))
    replacement_runs = _dedupe_run_dirs(discover_run_artifacts(replacement_root))

    base_selected = [
        run_dir
        for run_dir in base_runs
        if _agent_id_from_run_id(run_dir.name) != replace_agent
    ]
    replacement_selected = list(replacement_runs)

    if len(replacement_selected) != expected_replacement_runs:
        raise ValueError(
            f"replacement run count mismatch: expected {expected_replacement_runs}, got {len(replacement_selected)}"
        )
    if len(base_selected) != expected_base_non_replaced_runs:
        raise ValueError(
            "base non-replaced run count mismatch: "
            f"expected {expected_base_non_replaced_runs}, got {len(base_selected)}"
        )

    merged_run_ids: set[str] = set()
    for run_dir in base_selected:
        if run_dir.name in merged_run_ids:
            raise ValueError(f"duplicate run_id in base selection: {run_dir.name}")
        merged_run_ids.add(run_dir.name)
        _link_run_dir(run_dir, output_root / run_dir.name, use_symlinks=use_symlinks)

    for run_dir in replacement_selected:
        if run_dir.name in merged_run_ids:
            raise ValueError(f"duplicate run_id during merge: {run_dir.name}")
        merged_run_ids.add(run_dir.name)
        _link_run_dir(run_dir, output_root / run_dir.name, use_symlinks=use_symlinks)

    if len(merged_run_ids) != expected_total_runs:
        raise ValueError(
            f"merged run count mismatch: expected {expected_total_runs}, got {len(merged_run_ids)}"
        )

    for filename in ("preflight_report.json", "preflight_report.md"):
        source = base_root / filename
        if source.exists():
            shutil.copy2(source, output_root / filename)

    base_manifest = _read_json_if_exists(base_root / "manifest.json") or {}
    replacement_manifest = _read_json_if_exists(replacement_root / "manifest.json") or {}
    merge_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    provenance = {
        "merged": True,
        "base_run_id": base_root.name,
        "replacement_run_id": replacement_root.name,
        "replaced_agent_ids": [replace_agent],
        "replacement_reason": replacement_reason
        or (
            "Merged benchmark dataset: non-direct runs from base full matrix plus direct axis "
            "from parser-normalized direct-only rerun."
        ),
        "merge_timestamp": merge_timestamp,
        "expected_runs": expected_total_runs,
        "base_manifest_matrix_id": base_manifest.get("matrix_id"),
        "replacement_manifest_matrix_id": replacement_manifest.get("matrix_id"),
    }

    merged_manifest = dict(base_manifest)
    merged_metadata = dict(base_manifest.get("metadata") or {})
    merged_metadata.update(provenance)
    merged_metadata["final_benchmark"] = True
    merged_metadata["final_benchmark_merged"] = True
    merged_manifest["metadata"] = merged_metadata
    merged_manifest["generated_at"] = merge_timestamp
    merged_manifest["matrix_id"] = str(base_manifest.get("matrix_id") or output_root.name)
    merged_manifest["agent_ids"] = sorted(
        {
            agent
            for agent in (base_manifest.get("agent_ids") or [])
            if agent != replace_agent
        }
        | set(replacement_manifest.get("agent_ids") or [replace_agent])
    )
    (output_root / "manifest.json").write_text(
        json.dumps(merged_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    result: dict[str, Any] = {
        "output_root": str(output_root),
        "total_runs": len(merged_run_ids),
        "base_runs_kept": len(base_selected),
        "replacement_runs_added": len(replacement_selected),
        "provenance": provenance,
    }

    if rebuild_reports:
        analysis = run_analysis(output_root)
        report_config_path = base_manifest.get("report_config_path") or "configs/reports/default_report.yaml"
        matrix = ExperimentMatrix(
            matrix_id=str(merged_manifest.get("matrix_id") or output_root.name),
            output_root=output_root,
            report_config_path=Path(report_config_path),
            metadata=merged_metadata,
        )
        bundle_path = build_reports(analysis, matrix)
        result["report_bundle"] = str(bundle_path)

    return result


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None
