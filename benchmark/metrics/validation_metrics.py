from benchmark.metrics.models import RunMetric
from benchmark.validation.models import SceneValidationResult, ValidationSeverity


def extract_validation_metrics(result: SceneValidationResult) -> dict[str, float | str | int]:
    return {
        "task_id": result.task_id,
        "overall_status": result.overall_status.value,
        "total_score": result.total_score,
        "issue_count": len(result.issues),
        "validator_count": len(result.validators),
    }


def metrics_from_validation_result(
    run_id: str,
    task_id: str,
    result: SceneValidationResult,
) -> list[RunMetric]:
    metrics = [
        _metric(run_id, task_id, "total_score", result.total_score, "validation"),
        _metric(run_id, task_id, "overall_status", result.overall_status.value, "validation"),
        _metric(run_id, task_id, "issues_total", _issues_total(result), "validation"),
        _metric(run_id, task_id, "error_count", _error_count(result), "validation"),
        _metric(
            run_id,
            task_id,
            "validators_passed",
            _summary_or_count(result, "validators_passed", "passed"),
            "validators",
        ),
        _metric(
            run_id,
            task_id,
            "validators_failed",
            _summary_or_count(result, "validators_failed", "failed"),
            "validators",
        ),
        _metric(
            run_id,
            task_id,
            "validators_skipped",
            _summary_or_count(result, "validators_skipped", "skipped"),
            "validators",
        ),
        _metric(
            run_id,
            task_id,
            "validators_total",
            int(result.summary.get("validators_total", len(result.validators))),
            "validators",
        ),
        _metric(
            run_id,
            task_id,
            "validators_run",
            int(
                result.summary.get(
                    "validators_run",
                    sum(1 for validator in result.validators if validator.status.value != "skipped"),
                )
            ),
            "validators",
        ),
        _metric(
            run_id,
            task_id,
            "validation_coverage",
            _validation_coverage(result),
            "validators",
        ),
    ]

    for validator in result.validators:
        metrics.append(
            _metric(
                run_id,
                task_id,
                f"validator.{validator.name}.score",
                validator.score,
                "validator_scores",
            )
        )
        for metric_score in validator.metrics:
            metrics.append(
                _metric(
                    run_id,
                    task_id,
                    f"metric.{validator.name}.{metric_score.name}.score",
                    metric_score.score,
                    "metric_scores",
                )
            )

    return metrics


def _metric(
    run_id: str,
    task_id: str,
    name: str,
    value: float | str | int | bool,
    group: str,
) -> RunMetric:
    return RunMetric(
        run_id=run_id,
        task_id=task_id,
        name=name,
        value=value,
        group=group,
        source="validation_result",
    )


def _issues_total(result: SceneValidationResult) -> int:
    if "issues_total" in result.summary:
        return int(result.summary["issues_total"])
    return len(result.issues)


def _error_count(result: SceneValidationResult) -> int:
    if "error_count" in result.summary:
        return int(result.summary["error_count"])
    return sum(1 for issue in result.issues if issue.severity is ValidationSeverity.ERROR)


def _summary_or_count(
    result: SceneValidationResult,
    summary_key: str,
    status_value: str,
) -> int:
    if summary_key in result.summary:
        return int(result.summary[summary_key])
    return sum(1 for validator in result.validators if validator.status.value == status_value)


def _validation_coverage(result: SceneValidationResult) -> float:
    if "validation_coverage" in result.summary:
        return float(result.summary["validation_coverage"])
    total = len(result.validators)
    if total == 0:
        return 0.0
    run = sum(1 for validator in result.validators if validator.status.value != "skipped")
    return run / total
