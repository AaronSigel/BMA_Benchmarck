from __future__ import annotations

from collections import defaultdict

from bma_benchmark.reporting.scene_examples.models import (
    RunArtifactRef,
    SceneExample,
    SceneExampleBundle,
    SceneExampleSelectionConfig,
)


def select_scene_examples(
    runs: list[RunArtifactRef],
    config: SceneExampleSelectionConfig,
) -> SceneExampleBundle:
    selected: list[SceneExample] = []
    warnings: list[str] = []
    by_status: dict[str, list[RunArtifactRef]] = defaultdict(list)
    for run in runs:
        if run.pass_type in config.pass_types:
            by_status[str(run.pass_type)].append(run)

    for pass_type in config.pass_types:
        ranked = sorted(
            by_status.get(pass_type, []),
            key=lambda run: _sort_key(run, config.priority_tasks),
        )
        if not ranked:
            warnings.append(f"no examples found for {pass_type}")
        for run in ranked[: config.examples_per_status]:
            selected.append(_example(run, f"selected for {pass_type}"))

    return SceneExampleBundle(examples=selected, config=config, warnings=warnings)


def _sort_key(run: RunArtifactRef, priority_tasks: list[str]) -> tuple:
    try:
        priority = priority_tasks.index(run.task_id or "")
    except ValueError:
        priority = len(priority_tasks)
    image_present = 0 if (run.render_path or run.viewport_path) else 1
    validation_present = 0 if run.validation_result_path else 1
    score = -(run.scene_score if run.scene_score is not None else -1.0)
    return (priority, image_present, validation_present, score, run.run_id)


def _example(run: RunArtifactRef, reason: str) -> SceneExample:
    validation = run.validation_result or {}
    return SceneExample(
        run_id=run.run_id,
        task_id=run.task_id or "unknown",
        category=run.category,
        model=run.model,
        strategy=run.strategy,
        mcp_profile=run.mcp_profile,
        pass_type=run.pass_type or "unknown",
        scene_score=run.scene_score,
        strict_success=run.strict_success,
        run_dir=run.run_dir,
        snapshot_path=run.snapshot_path,
        validation_result_path=run.validation_result_path,
        render_path=run.render_path,
        viewport_path=run.viewport_path,
        blend_path=run.blend_path,
        glb_path=run.glb_path,
        top_issues=_top_issues(validation),
        check_table_excerpt=_check_excerpt(validation),
        selection_reason=reason,
        render_missing_reason=run.render_missing_reason if not (run.render_path or run.viewport_path) else None,
    )


def _top_issues(validation: dict) -> list[str]:
    issues = validation.get("issues")
    if not isinstance(issues, list):
        return []
    result = []
    for issue in issues[:3]:
        if isinstance(issue, dict):
            result.append(str(issue.get("code") or issue.get("message") or "issue"))
    return result


def _check_excerpt(validation: dict) -> list[dict]:
    checks = validation.get("check_table")
    if not isinstance(checks, list):
        return []
    excerpt = []
    for row in checks:
        if isinstance(row, dict):
            excerpt.append({
                "validator_name": row.get("validator_name"),
                "check_name": row.get("check_name"),
                "entity_ref": row.get("entity_ref"),
                "field": row.get("field"),
                "expected": row.get("expected"),
                "actual": row.get("actual"),
                "tolerance": row.get("tolerance"),
                "passed": row.get("passed"),
                "score": row.get("score"),
                "issue_code": row.get("issue_code"),
            })
        if len(excerpt) >= 5:
            break
    return excerpt
