from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from bma_benchmark.validation_audit.models import ValidatorAuditReport

FIELDS = [
    "task_id",
    "category",
    "validator_name",
    "criterion_name",
    "checked_entity",
    "checked_field",
    "expected_path",
    "actual_path",
    "tolerance",
    "weight",
    "required",
    "scoring_rule",
    "pass_threshold",
    "warning_threshold",
    "issue_codes",
    "limitation",
]


def write_validator_audit(report: ValidatorAuditReport, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        write_inventory_csv(report, out_dir / "validator_inventory.csv"),
        write_inventory_json(report, out_dir / "validator_inventory.json"),
        write_audit_markdown(report, out_dir / "validator_audit.md"),
        write_limitations_markdown(report, out_dir / "validator_limitations.md"),
    ]
    return paths


def write_inventory_csv(report: ValidatorAuditReport, path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in report.rows:
            data = row.model_dump(mode="json")
            data["issue_codes"] = ";".join(data.get("issue_codes") or [])
            writer.writerow({field: data.get(field) for field in FIELDS})
    return path


def write_inventory_json(report: ValidatorAuditReport, path: Path) -> Path:
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_audit_markdown(report: ValidatorAuditReport, path: Path) -> Path:
    lines = [
        "# Validator Audit",
        "",
        f"tasks_dir: `{report.tasks_dir}`",
        f"task_count: {report.task_count}",
        f"check_rows: {len(report.rows)}",
        "",
        "| Task | Category | Validator | Criterion | Field | Weight | Tolerance | Required |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.rows:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row.task_id,
                    row.category,
                    row.validator_name,
                    row.criterion_name,
                    row.checked_field,
                    row.weight,
                    row.tolerance,
                    row.required,
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_limitations_markdown(report: ValidatorAuditReport, path: Path) -> Path:
    lines = ["# Validator Limitations", ""]
    for item in report.limitations:
        lines.append(f"- `{item.validator_name}`: {item.limitation}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _cell(value: Any) -> str:
    if value is None:
        return ""
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")
