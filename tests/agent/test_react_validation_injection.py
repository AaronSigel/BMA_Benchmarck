"""Tests for ReAct validation injection before repair steps."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from pathlib import Path

from benchmark.agent.strategies.react import (
    ReactStrategy,
    _build_step_context,
    _repair_activation_metadata,
    _validation_passed_or_warning,
)


def _make_val_result(status: str = "failed", issue_codes: list[str] | None = None):
    issue_codes = issue_codes if issue_codes is not None else ["object_missing"]
    val = MagicMock()
    val.overall_status = status
    val.total_score = 0.0 if status == "failed" else 1.0
    issues = []
    for index, code in enumerate(issue_codes):
        issue = MagicMock()
        issue.code = code
        issue.message = f"{code} issue"
        issue.severity = "error"
        issue.expected_path = f"expected_scene.lights[{index}]" if "light" in code else f"expected_scene.objects[{index}]"
        issues.append(issue)
    val.issues = issues
    return val


def test_react_gets_validation_result_before_step() -> None:
    val_result = _make_val_result()
    _, _, activation = _build_step_context(val_result, MagicMock(), None, [])
    assert activation["validation_result_available"] is True
    assert activation["issue_count"] == 1


def test_react_uses_mapper_when_issue_exists() -> None:
    import yaml

    val_result = _make_val_result(status="failed", issue_codes=["light_missing"])
    task_obj = yaml.safe_load(Path("tasks/lighting/lighting_001_area_light.yaml").read_text(encoding="utf-8"))
    from benchmark.tasks.models import BenchmarkTask

    task_model = BenchmarkTask.model_validate(task_obj)
    ctx, repair, activation = _build_step_context(val_result, task_model, None, [])
    assert activation["repair_mapped"] is True
    assert activation["repair_tool"] == "bma_create_light"
    assert ctx is not None
    assert ctx.get("top_issue", {}).get("code") == "light_missing"


def test_react_logs_no_snapshot_reason() -> None:
    activation = _repair_activation_metadata(
        None, MagicMock(), None, None, [], deterministic_executed=False, fallback_reason="no_validation_result"
    )
    assert activation["fallback_reason"] == "no_validation_result"
    assert activation["snapshot_available"] is False


def test_react_early_stops_after_passed_validation() -> None:
    val_result = _make_val_result(status="passed", issue_codes=[])
    assert _validation_passed_or_warning(val_result) is True


def test_react_strategy_has_initial_validation_result_attr() -> None:
    strategy = ReactStrategy()
    strategy.initial_validation_result = _make_val_result(status="failed")
    assert strategy.initial_validation_result is not None
