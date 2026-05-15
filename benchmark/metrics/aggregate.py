from collections import defaultdict

from benchmark.metrics.models import MetricSummary, MetricsSummary, RunMetric, RunMetricRow
from benchmark.runner.models import RunResult, RunStatus


def aggregate_metric_rows(rows: list[RunMetricRow]) -> MetricSummary:
    scores = [row.total_score for row in rows if row.total_score is not None]
    average_score = sum(scores) / len(scores) if scores else 0.0
    return MetricSummary(
        total_runs=len(rows),
        average_score=average_score,
        passed_runs=sum(1 for row in rows if row.overall_status == "passed"),
        failed_runs=sum(1 for row in rows if row.overall_status == "failed"),
        error_runs=sum(1 for row in rows if row.overall_status == "error"),
    )


def aggregate_run_results(results: list[RunResult]) -> MetricsSummary:
    scores = [result.total_score for result in results if result.total_score is not None]
    return MetricsSummary(
        total_runs=len(results),
        passed_runs=sum(1 for result in results if result.status is RunStatus.PASSED),
        failed_runs=sum(1 for result in results if result.status is RunStatus.FAILED),
        error_runs=sum(1 for result in results if result.status is RunStatus.ERROR),
        average_score=(sum(scores) / len(scores) if scores else None),
        min_score=(min(scores) if scores else None),
        max_score=(max(scores) if scores else None),
        metrics=_summary_metrics(results),
    )


def group_by_task(results: list[RunResult]) -> dict[str, MetricsSummary]:
    grouped: dict[str, list[RunResult]] = defaultdict(list)
    for result in results:
        grouped[result.task_id].append(result)
    return {
        task_id: aggregate_run_results(task_results)
        for task_id, task_results in sorted(grouped.items())
    }


def group_by_execution_mode(results: list[RunResult]) -> dict[str, MetricsSummary]:
    grouped: dict[str, list[RunResult]] = defaultdict(list)
    for result in results:
        grouped[result.execution_mode.value].append(result)
    return {
        mode: aggregate_run_results(mode_results)
        for mode, mode_results in sorted(grouped.items())
    }


def rank_runs_by_score(results: list[RunResult]) -> list[RunResult]:
    return sorted(
        results,
        key=lambda result: (
            result.total_score is not None,
            result.total_score if result.total_score is not None else -1.0,
        ),
        reverse=True,
    )


def _summary_metrics(results: list[RunResult]) -> list[RunMetric]:
    metrics: list[RunMetric] = []
    for result in results:
        if result.total_score is not None:
            metrics.append(
                RunMetric(
                    run_id=result.run_id,
                    task_id=result.task_id,
                    name="total_score",
                    value=result.total_score,
                    group="run",
                    source="run_result",
                )
            )
        if result.overall_status is not None:
            metrics.append(
                RunMetric(
                    run_id=result.run_id,
                    task_id=result.task_id,
                    name="overall_status",
                    value=result.overall_status,
                    group="run",
                    source="run_result",
                )
            )
    return metrics
