from pathlib import Path

from benchmark.metrics.aggregate import aggregate_run_results
from benchmark.metrics.export import (
    write_metrics_csv,
    write_run_results_json,
    write_summary_csv,
    write_summary_json,
)
from benchmark.runner.experiment_runner import ExperimentRunner
from benchmark.runner.models import ExperimentConfig, ExperimentResult, RunConfig, RunResult
from benchmark.runner.paths import RunArtifactLayout


class BatchRunner:
    """Runs all entries in an ExperimentConfig sequentially."""

    def __init__(self, runner: ExperimentRunner | None = None) -> None:
        self.runner = runner or ExperimentRunner()

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        results: list[RunResult] = []
        for run_config in config.runs:
            results.append(self.runner.run(run_config))

        metrics_summary = aggregate_run_results(results)
        result = ExperimentResult(
            experiment_id=config.experiment_id,
            runs=results,
            summary=metrics_summary.model_dump(exclude={"metrics"}),
        )
        output_dir = _experiment_output_dir(config.runs)
        _write_experiment_result(result, output_dir / "experiment_result.json")
        write_run_results_json(results, output_dir / "run_results.json")
        write_summary_json(metrics_summary, output_dir / "summary.json")
        write_summary_csv(results, output_dir / "summary.csv")
        write_metrics_csv(metrics_summary.metrics, output_dir / "metrics.csv")
        return result

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        return self.run_experiment(config)


def _experiment_output_dir(runs: list[RunConfig]) -> Path:
    if not runs:
        return Path("artifacts") / "experiments"

    roots = {
        RunArtifactLayout.from_run_output_dir(run.output_dir, run.run_id).root
        for run in runs
    }
    if len(roots) == 1:
        return roots.pop()
    return RunArtifactLayout.from_run_output_dir(runs[0].output_dir, runs[0].run_id).root


def _write_experiment_result(result: ExperimentResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
