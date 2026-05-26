from __future__ import annotations

from collections import defaultdict

from bma_benchmark.reporting.scene_examples.models import (
    RunArtifactRef,
    SceneExample,
    SceneExampleBundle,
    SceneExampleSelectionConfig,
)
from bma_benchmark.reporting.scene_examples.selection import _example


TARGET_ISSUE_PREFIXES = (
    "object_missing",
    "location_mismatch",
    "material_",
    "light_",
    "camera_",
    "export_",
)


def select_evidence_examples(
    runs: list[RunArtifactRef],
    config: SceneExampleSelectionConfig,
) -> SceneExampleBundle:
    """Выбор примеров по правилам TZ для report evidence pack."""
    warnings: list[str] = []
    by_status: dict[str, list[RunArtifactRef]] = defaultdict(list)
    for run in runs:
        if run.pass_type in config.pass_types:
            by_status[str(run.pass_type)].append(run)

    selected: list[SceneExample] = []
    clean = _select_clean_pass(by_status.get("clean_pass", []), config)
    failed = _select_failed_validation(by_status.get("failed_validation", []), config)
    soft = _select_soft_pass(by_status.get("soft_pass", []), config)

    if len(clean) < config.examples_per_status:
        warnings.append(f"only {len(clean)} clean_pass examples (target {config.examples_per_status})")
    if len(failed) < config.examples_per_status:
        warnings.append(f"only {len(failed)} failed_validation examples (target {config.examples_per_status})")
    if not soft:
        warnings.append("no soft_pass examples found")

    for run in clean:
        selected.append(_example(run, _clean_reason(run, config)))
    for run in failed:
        selected.append(_example(run, _failed_reason(run)))
    for run in soft:
        selected.append(_example(run, _soft_reason(run, config)))

    return SceneExampleBundle(examples=selected, config=config, warnings=warnings)


def _select_clean_pass(runs: list[RunArtifactRef], config: SceneExampleSelectionConfig) -> list[RunArtifactRef]:
    eligible = [
        run for run in runs
        if run.scene_score is None or run.scene_score >= config.min_clean_pass_score
    ]
    if not eligible:
        eligible = list(runs)
        if runs:
            pass  # fallback to all clean_pass runs
    ranked = sorted(eligible, key=lambda run: _clean_sort_key(run, config.priority_tasks))
    return _pick_diverse_tasks(ranked, config.examples_per_status, config.priority_tasks)


def _select_failed_validation(
    runs: list[RunArtifactRef],
    config: SceneExampleSelectionConfig,
) -> list[RunArtifactRef]:
    preferred = [r for r in runs if r.strategy in config.failed_prefer_strategies]
    pool = preferred or list(runs)
    ranked = sorted(pool, key=lambda run: _failed_sort_key(run, config.priority_tasks))
    if not config.diversify_issue_codes:
        return ranked[: config.examples_per_status]
    return _pick_diverse_issues(ranked, config.examples_per_status)


def _select_soft_pass(runs: list[RunArtifactRef], config: SceneExampleSelectionConfig) -> list[RunArtifactRef]:
    if not runs:
        return []
    ranked = sorted(runs, key=lambda run: _soft_sort_key(run, config))
    return ranked[: min(2, len(ranked))]


def _pick_diverse_tasks(
    ranked: list[RunArtifactRef],
    limit: int,
    priority_tasks: list[str],
) -> list[RunArtifactRef]:
    picked: list[RunArtifactRef] = []
    seen_tasks: set[str] = set()
    for task_id in priority_tasks:
        for run in ranked:
            if run.task_id == task_id and run.run_id not in {r.run_id for r in picked}:
                picked.append(run)
                seen_tasks.add(task_id)
                break
        if len(picked) >= limit:
            return picked[:limit]
    for run in ranked:
        if run.run_id not in {r.run_id for r in picked}:
            picked.append(run)
        if len(picked) >= limit:
            break
    return picked[:limit]


def _pick_diverse_issues(ranked: list[RunArtifactRef], limit: int) -> list[RunArtifactRef]:
    picked: list[RunArtifactRef] = []
    seen_codes: set[str] = set()
    for run in ranked:
        codes = _issue_codes(run)
        new_codes = [c for c in codes if c not in seen_codes]
        if new_codes or not picked:
            picked.append(run)
            seen_codes.update(codes)
        if len(picked) >= limit:
            break
    if len(picked) < limit:
        for run in ranked:
            if run.run_id not in {r.run_id for r in picked}:
                picked.append(run)
            if len(picked) >= limit:
                break
    return picked[:limit]


def _issue_codes(run: RunArtifactRef) -> list[str]:
    validation = run.validation_result or {}
    codes: list[str] = []
    for issue in validation.get("issues") or []:
        if isinstance(issue, dict) and issue.get("code"):
            codes.append(str(issue["code"]))
    for row in validation.get("check_table") or []:
        if isinstance(row, dict) and row.get("issue_code"):
            codes.append(str(row["issue_code"]))
    return codes


def _clean_sort_key(run: RunArtifactRef, priority_tasks: list[str]) -> tuple:
    try:
        priority = priority_tasks.index(run.task_id or "")
    except ValueError:
        priority = len(priority_tasks)
    image_present = 0 if (run.render_path or run.viewport_path) else 1
    validation_present = 0 if run.validation_result_path else 1
    score = -(run.scene_score if run.scene_score is not None else -1.0)
    return (priority, image_present, validation_present, score, run.run_id)


def _failed_sort_key(run: RunArtifactRef, priority_tasks: list[str]) -> tuple:
    base = _clean_sort_key(run, priority_tasks)
    issue_richness = -len(_issue_codes(run))
    return base[:1] + (issue_richness,) + base[1:]


def _soft_sort_key(run: RunArtifactRef, config: SceneExampleSelectionConfig) -> tuple:
    task_pref = 0 if run.task_id in config.soft_pass_prefer_tasks else 1
    strat_pref = 0 if run.strategy in config.soft_pass_prefer_strategies else 1
    has_glb = 0 if run.glb_path else 1
    has_validation = 0 if run.validation_result_path else 1
    score = -(run.scene_score if run.scene_score is not None else -1.0)
    return (task_pref, strat_pref, has_glb, has_validation, score, run.run_id)


def _clean_reason(run: RunArtifactRef, config: SceneExampleSelectionConfig) -> str:
    parts = [f"clean pass {run.task_id or 'unknown'} example"]
    if run.scene_score is not None and run.scene_score >= config.min_clean_pass_score:
        parts.append("with visible object positioning" if "geometry" in (run.task_id or "") else "with high scene score")
    if run.render_path or run.viewport_path:
        parts.append("with render/viewport image")
    return " ".join(parts)


def _failed_reason(run: RunArtifactRef) -> str:
    codes = _issue_codes(run)
    if codes:
        return f"failed validation with {codes[0]}"
    return f"failed validation {run.task_id or 'unknown'}"


def _soft_reason(run: RunArtifactRef, config: SceneExampleSelectionConfig) -> str:
    if run.task_id in config.soft_pass_prefer_tasks:
        return "soft pass export example showing valid GLB but diagnostic termination"
    return f"soft pass {run.task_id or 'unknown'} with diagnostic agent termination"
