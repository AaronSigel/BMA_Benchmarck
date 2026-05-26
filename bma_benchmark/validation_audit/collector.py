from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.tasks.loader import load_tasks_from_dir
from benchmark.tasks.models import BenchmarkTask, SuccessCriterion
from benchmark.validation.scene_validator import DEFAULT_VALIDATOR_WEIGHT, VALIDATOR_METRIC_ALIASES

from bma_benchmark.validation_audit.models import (
    ValidatorAuditReport,
    ValidatorAuditRow,
    ValidatorLimitation,
)
from bma_benchmark.validation_audit.registry import (
    PASS_THRESHOLD,
    VALIDATOR_FIELD_SPECS,
    VALIDATOR_LIMITATIONS,
    WARNING_THRESHOLD,
)


def collect_validator_audit(tasks_dir: Path) -> ValidatorAuditReport:
    tasks = load_tasks_from_dir(tasks_dir)
    rows: list[ValidatorAuditRow] = []
    for task in tasks:
        for spec in VALIDATOR_FIELD_SPECS:
            entities = list(getattr(task.expected_scene, spec.expected_section))
            if not entities:
                continue
            if spec.validator_name == "glb_import_back_validator":
                entities = [entity for entity in entities if getattr(entity, "format", "").lower() == "glb"]
                if not entities:
                    continue
            for index, entity in enumerate(entities):
                if spec.expected_attr is not None and spec.expected_attr != "format":
                    value = getattr(entity, spec.expected_attr, None)
                    if value is None:
                        continue
                rows.append(_row(task, spec, entity, index))

    limitations = [
        ValidatorLimitation(validator_name=name, limitation=text)
        for name, text in sorted(VALIDATOR_LIMITATIONS.items())
    ]
    return ValidatorAuditReport(
        tasks_dir=str(tasks_dir),
        task_count=len(tasks),
        rows=rows,
        limitations=limitations,
    )


def _row(task: BenchmarkTask, spec, entity: Any, index: int) -> ValidatorAuditRow:
    weight, required = _criterion_values(spec.validator_name, task.success_criteria)
    tolerance = _tolerance(spec, entity)
    return ValidatorAuditRow(
        task_id=task.id,
        category=task.category.value,
        validator_name=spec.validator_name,
        criterion_name=spec.criterion_name,
        checked_entity=spec.checked_entity,
        checked_field=spec.checked_field,
        expected_path=f"expected_scene.{spec.expected_section}[{index}]"
        + (f".{spec.expected_attr}" if spec.expected_attr and spec.expected_attr != "format" else ""),
        actual_path=spec.actual_path_template.format(index=index),
        tolerance=tolerance,
        weight=weight,
        required=required,
        scoring_rule=spec.scoring_rule,
        pass_threshold=PASS_THRESHOLD,
        warning_threshold=WARNING_THRESHOLD,
        issue_codes=list(spec.issue_codes),
        limitation=spec.limitation or VALIDATOR_LIMITATIONS.get(spec.validator_name),
    )


def _criterion_values(
    validator_name: str,
    success_criteria: list[SuccessCriterion],
) -> tuple[float, bool]:
    aliases = VALIDATOR_METRIC_ALIASES.get(validator_name, {validator_name})
    matched = [
        criterion
        for criterion in success_criteria
        if criterion.metric in aliases or criterion.metric == validator_name
    ]
    if not matched:
        return DEFAULT_VALIDATOR_WEIGHT, True
    return sum(criterion.weight for criterion in matched), any(c.required for c in matched)


def _tolerance(spec, entity: Any) -> float | None:
    if spec.checked_field == "target" and hasattr(entity, "direction_tolerance_deg"):
        return float(entity.direction_tolerance_deg)
    if spec.checked_field == "energy" and getattr(entity, "energy", None) is not None:
        return max(abs(float(entity.energy)) * float(getattr(entity, "tolerance", spec.default_tolerance or 0.0)), 1.0)
    if hasattr(entity, "tolerance") and spec.default_tolerance is not None:
        return float(entity.tolerance)
    return spec.default_tolerance
