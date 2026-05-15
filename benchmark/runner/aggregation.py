from benchmark.metrics.aggregate import aggregate_run_results
from benchmark.runner.models import ExperimentResult, RunResult


def summarize_runs(runs: list[RunResult]) -> dict[str, object]:
    summary = aggregate_run_results(runs)
    return summary.model_dump(exclude={"metrics"})


def summarize_experiment(result: ExperimentResult) -> dict[str, object]:
    return summarize_runs(result.runs)
