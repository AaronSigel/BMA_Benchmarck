"""Тесты таблицы validator expected/actual в evidence pack."""

from __future__ import annotations

from pathlib import Path

from bma_benchmark.reporting.evidence_pack.figures import _best_validator_example
from bma_benchmark.reporting.evidence_pack.figure_renderers import (
    _prioritize_failed_rows,
    _row_status,
)
from bma_benchmark.reporting.scene_examples.models import SceneExample
from benchmark.validation.check_labels import display_check_id, display_object_ref


def test_display_check_id_maps_export_rows() -> None:
    row = {
        "validator_name": "export_validator",
        "check_name": "file exists",
        "field": "path",
    }
    assert display_check_id(row) == "export.file_exists"
    assert display_object_ref(row) == "export"


def test_prioritize_failed_rows_puts_export_first() -> None:
    rows = [
        {
            "validator_name": "object_validator",
            "check_name": "object exists",
            "field": "object",
            "passed": False,
            "status": "fail",
        },
        {
            "validator_name": "export_validator",
            "check_name": "file exists",
            "field": "path",
            "passed": False,
            "status": "fail",
        },
        {
            "validator_name": "transform_validator",
            "check_name": "dimensions",
            "field": "dimensions",
            "passed": False,
            "status": "skip",
        },
    ]
    ordered = _prioritize_failed_rows(rows, limit=10)
    assert ordered[0]["validator_name"] == "export_validator"
    assert _row_status(ordered[2]) == "skip"


def test_best_validator_example_prefers_partial_pass_over_total_failure() -> None:
    all_fail = SceneExample(
        run_id="fail_run",
        task_id="export_002_glb_file",
        pass_type="failed_validation",
        scene_score=0.15,
        run_dir=Path("/tmp/fail"),
        selection_reason="test",
        check_table_excerpt=[
            {"passed": False, "status": "fail"},
            {"passed": False, "status": "fail"},
        ],
    )
    partial = SceneExample(
        run_id="partial_run",
        task_id="lighting_003_three_point_lighting",
        pass_type="soft_pass",
        scene_score=0.88,
        run_dir=Path("/tmp/partial"),
        selection_reason="test",
        check_table_excerpt=[
            {"passed": True, "status": "pass"},
            {"passed": True, "status": "pass"},
            {"passed": False, "status": "fail"},
        ],
    )
    chosen = _best_validator_example([all_fail, partial])
    assert chosen is not None
    assert chosen.run_id == "partial_run"
