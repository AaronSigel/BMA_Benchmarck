import csv
import json
from pathlib import Path

from benchmark.metrics.aggregate import aggregate_run_results
from benchmark.metrics.export import (
    METRICS_CSV_COLUMNS,
    SUMMARY_CSV_COLUMNS,
    write_metrics_csv,
    write_run_results_json,
    write_summary_csv,
    write_summary_json,
)
from benchmark.metrics.models import MetricsSummary, RunMetric
from benchmark.runner.models import ExecutionMode, RunResult, RunStatus


def make_run_result(
    run_id: str = "run_001",
    score: float | None = 0.75,
    status: RunStatus = RunStatus.PASSED,
) -> RunResult:
    return RunResult(
        run_id=run_id,
        task_id="geometry_001_basic_primitives",
        status=status,
        execution_mode=ExecutionMode.EXTERNAL_SNAPSHOT,
        validation_result_path=Path("artifacts/runs/run_001/validation_result.json"),
        scene_snapshot_path=Path("artifacts/blender_smoke/scene_snapshot.json"),
        artifacts_dir=Path("artifacts/blender_smoke"),
        total_score=score,
        overall_status="passed" if status is not RunStatus.ERROR else None,
        started_at="2026-05-15T10:00:00Z",
        finished_at="2026-05-15T10:00:01Z",
        duration_sec=1.0,
        error="boom" if status is RunStatus.ERROR else None,
        summary={},
    )


def test_write_summary_json_round_trips_through_pydantic(tmp_path: Path) -> None:
    summary = aggregate_run_results([make_run_result()])
    path = tmp_path / "summary.json"

    write_summary_json(summary, path)

    loaded = MetricsSummary.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded == summary


def test_write_run_results_json_round_trips_through_pydantic(tmp_path: Path) -> None:
    results = [make_run_result("run_001"), make_run_result("run_002", None, RunStatus.ERROR)]
    path = tmp_path / "run_results.json"

    write_run_results_json(results, path)

    loaded = [
        RunResult.model_validate(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]
    assert loaded == results


def test_write_summary_csv_uses_expected_columns(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"

    write_summary_csv([make_run_result()], path)

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    assert reader.fieldnames == SUMMARY_CSV_COLUMNS
    assert rows[0]["run_id"] == "run_001"
    assert rows[0]["status"] == "passed"
    assert rows[0]["total_score"] == "0.75"


def test_write_metrics_csv_uses_expected_columns(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    metric = RunMetric(
        run_id="run_001",
        task_id="geometry_001_basic_primitives",
        name="total_score",
        value=0.75,
        group="run",
        source="run_result",
    )

    write_metrics_csv([metric], path)

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    assert reader.fieldnames == METRICS_CSV_COLUMNS
    assert rows[0]["name"] == "total_score"
    assert rows[0]["value"] == "0.75"


def test_empty_exports_write_headers_and_valid_json(tmp_path: Path) -> None:
    summary = aggregate_run_results([])

    write_summary_json(summary, tmp_path / "summary.json")
    write_run_results_json([], tmp_path / "run_results.json")
    write_summary_csv([], tmp_path / "summary.csv")
    write_metrics_csv([], tmp_path / "metrics.csv")

    assert MetricsSummary.model_validate_json(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    ) == summary
    assert json.loads((tmp_path / "run_results.json").read_text(encoding="utf-8")) == []
    assert (tmp_path / "summary.csv").read_text(encoding="utf-8").splitlines()[0] == ",".join(
        SUMMARY_CSV_COLUMNS
    )
    assert (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines()[0] == ",".join(
        METRICS_CSV_COLUMNS
    )
