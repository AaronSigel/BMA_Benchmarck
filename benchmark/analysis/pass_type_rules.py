"""Правила классификации pass_type, включая export-задачи."""

from __future__ import annotations

from typing import Any

EXPORT_BLOCKING_ISSUE_CODES = frozenset({
    "object_missing",
    "primitive_mismatch",
    "export_missing",
    "export_empty_file",
    "export_import_failed",
    "export_import_missing",
    "export_import_object_missing",
})


def export_task_blocks_soft_pass(
    task_id: str | None,
    issues: list[dict[str, Any]] | None,
    *,
    object_score: float | None = None,
    export_score: float | None = None,
    import_back_score: float | None = None,
) -> bool:
    """Для export_* soft_pass запрещён при критических расхождениях."""
    if not task_id or not str(task_id).startswith("export_"):
        return False
    if object_score is not None and object_score < 0.5:
        return True
    if export_score is not None and export_score < 1.0:
        return True
    if import_back_score is not None and import_back_score < 1.0:
        return True
    for issue in issues or []:
        if not isinstance(issue, dict):
            continue
        code = str(issue.get("code") or issue.get("issue_code") or "").strip()
        if code in EXPORT_BLOCKING_ISSUE_CODES:
            return True
    return False


def apply_export_pass_type_guard(
    pass_type: str,
    task_id: str | None,
    issues: list[dict[str, Any]] | None,
    *,
    object_score: float | None = None,
    export_score: float | None = None,
    import_back_score: float | None = None,
) -> str:
    if pass_type in {"clean_pass", "soft_pass"} and export_task_blocks_soft_pass(
        task_id,
        issues,
        object_score=object_score,
        export_score=export_score,
        import_back_score=import_back_score,
    ):
        return "failed_validation"
    return pass_type
