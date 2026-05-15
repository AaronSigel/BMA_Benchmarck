import csv
from pathlib import Path

from benchmark.metrics.models import MetricsSummary, RunMetric
from benchmark.runner.models import RunResult

SUMMARY_CSV_COLUMNS = [
    "run_id",
    "task_id",
    "status",
    "execution_mode",
    "total_score",
    "overall_status",
    "duration_sec",
    "validation_result_path",
    "scene_snapshot_path",
    "error",
]

METRICS_CSV_COLUMNS = [
    "run_id",
    "task_id",
    "name",
    "value",
    "group",
    "source",
]


def write_summary_json(summary: MetricsSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def write_run_results_json(results: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "[" + ",".join(result.model_dump_json(indent=2) for result in results) + "]"
    path.write_text(payload, encoding="utf-8")


def write_summary_csv(results: list[RunResult], path: Path) -> None:
    _write_csv(
        [_run_summary_row(result) for result in results],
        SUMMARY_CSV_COLUMNS,
        path,
    )


def write_metrics_csv(metrics: list[RunMetric], path: Path) -> None:
    _write_csv(
        [metric.model_dump(mode="json") for metric in metrics],
        METRICS_CSV_COLUMNS,
        path,
    )


def _run_summary_row(result: RunResult) -> dict[str, object]:
    return {
        "run_id": result.run_id,
        "task_id": result.task_id,
        "status": result.status.value,
        "execution_mode": result.execution_mode.value,
        "total_score": result.total_score,
        "overall_status": result.overall_status,
        "duration_sec": result.duration_sec,
        "validation_result_path": (
            str(result.validation_result_path) if result.validation_result_path else None
        ),
        "scene_snapshot_path": str(result.scene_snapshot_path) if result.scene_snapshot_path else None,
        "error": result.error,
    }


def _write_csv(rows: list[dict[str, object]], fieldnames: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
