from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.tasks.loader import load_tasks_from_dir
from benchmark.validation.models import ValidationStatus
from benchmark.validation.scene_validator import SceneValidator

from bma_benchmark.reporting.evidence_pack.sanity_fixtures import (
    SANITY_CASES,
    build_sanity_snapshot,
    find_task_path,
)


@dataclass
class SanityCaseResult:
    case_id: str
    validator: str
    positive_or_negative: str
    expected_outcome: str
    actual_status: str
    scene_score: float
    passed_as_expected: bool
    issue_codes: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class SanitySuiteResult:
    cases: list[SanityCaseResult] = field(default_factory=list)
    all_passed_as_expected: bool = True
    failed_cases: list[SanityCaseResult] = field(default_factory=list)


def run_validator_sanity_suite(out_dir: Path, *, tasks_root: Path = Path("tasks")) -> SanitySuiteResult:
    """Готовит и прогоняет sanity-набор валидаторов."""
    out_dir.mkdir(parents=True, exist_ok=True)
    validator = SceneValidator()
    result = SanitySuiteResult()

    for spec in SANITY_CASES:
        case_dir = out_dir / spec.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        task_path = find_task_path(spec.task_id, tasks_root)
        if task_path is None:
            result.cases.append(
                SanityCaseResult(
                    case_id=spec.case_id,
                    validator=spec.validator,
                    positive_or_negative=spec.positive_or_negative,
                    expected_outcome=spec.expected_outcome,
                    actual_status="error",
                    scene_score=0.0,
                    passed_as_expected=False,
                    notes=f"task not found: {spec.task_id}",
                )
            )
            continue

        shutil.copy2(task_path, case_dir / "task.yaml")
        snapshot = build_sanity_snapshot(spec)
        snapshot_path = case_dir / "scene_snapshot.json"
        snapshot_path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")

        artifacts_dir = case_dir
        if spec.copy_export_glb:
            src = Path("tests/fixtures/validation/export_artifacts/exports/result.glb")
            if src.is_file():
                exports = case_dir / "exports"
                exports.mkdir(exist_ok=True)
                shutil.copy2(src, exports / "result.glb")

        from benchmark.tasks.loader import load_task

        task = load_task(case_dir / "task.yaml")
        validation = validator.validate(task, snapshot, artifacts_dir=artifacts_dir)
        validation_path = case_dir / "validation_result.json"
        validation_path.write_text(validation.model_dump_json(indent=2), encoding="utf-8")

        actual = _outcome_label(validation.overall_status, validation.total_score, spec.expected_outcome)
        passed = _matches_expected(spec.expected_outcome, validation, spec.case_id)
        issue_codes = [issue.code for issue in validation.issues]
        case_result = SanityCaseResult(
            case_id=spec.case_id,
            validator=spec.validator,
            positive_or_negative=spec.positive_or_negative,
            expected_outcome=spec.expected_outcome,
            actual_status=actual,
            scene_score=validation.total_score,
            passed_as_expected=passed,
            issue_codes=issue_codes,
            notes=spec.notes,
        )
        result.cases.append(case_result)
        if not passed:
            result.all_passed_as_expected = False
            result.failed_cases.append(case_result)

    if result.failed_cases:
        _write_failed_sanity_cases(out_dir / "FAILED_SANITY_CASES.md", result.failed_cases)
    return result


def write_validator_sanity_results_csv(path: Path, suite: SanitySuiteResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "validator",
        "positive_or_negative",
        "expected_outcome",
        "actual_status",
        "scene_score",
        "passed_as_expected",
        "issue_codes",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for case in suite.cases:
            writer.writerow({
                "case_id": case.case_id,
                "validator": case.validator,
                "positive_or_negative": case.positive_or_negative,
                "expected_outcome": case.expected_outcome,
                "actual_status": case.actual_status,
                "scene_score": f"{case.scene_score:.4f}",
                "passed_as_expected": case.passed_as_expected,
                "issue_codes": ";".join(case.issue_codes),
                "notes": case.notes,
            })
    return path


def _outcome_label(status: ValidationStatus, score: float, expected: str) -> str:
    if expected == "pass":
        return "pass" if status == ValidationStatus.PASSED else status.value
    if expected in {"fail", "fail/partial"}:
        if status == ValidationStatus.FAILED:
            return "fail"
        if status == ValidationStatus.WARNING or score < 1.0:
            return "partial"
        return status.value
    return status.value


def _matches_expected(expected: str, validation, case_id: str) -> bool:
    status = validation.overall_status
    score = validation.total_score
    if expected == "export_pass":
        for validator in validation.validators:
            if validator.name == "export_validator":
                return validator.status == ValidationStatus.PASSED
        return False
    if expected == "pass":
        return status == ValidationStatus.PASSED and score >= 0.99
    if expected == "fail":
        return status == ValidationStatus.FAILED
    if expected == "fail/partial":
        return status in {ValidationStatus.FAILED, ValidationStatus.WARNING} or score < 1.0
    return False


def _write_failed_sanity_cases(path: Path, failed: list[SanityCaseResult]) -> None:
    lines = ["# Failed Sanity Cases", ""]
    for case in failed:
        lines.append(f"## {case.case_id}")
        lines.append(f"- validator: {case.validator}")
        lines.append(f"- expected: {case.expected_outcome}")
        lines.append(f"- actual: {case.actual_status} (score={case.scene_score:.4f})")
        lines.append(f"- issue_codes: {', '.join(case.issue_codes) or 'none'}")
        if case.notes:
            lines.append(f"- notes: {case.notes}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
