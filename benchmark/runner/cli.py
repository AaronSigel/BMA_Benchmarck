import argparse
from pathlib import Path

from pydantic import ValidationError

from benchmark.metrics.aggregate import aggregate_run_results
from benchmark.runner.batch_runner import BatchRunner
from benchmark.runner.config_loader import load_experiment_config, load_run_config
from benchmark.runner.errors import RunnerConfigError
from benchmark.runner.experiment_runner import ExperimentRunner
from benchmark.runner.models import ExperimentResult, RunResult, RunStatus


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run(args.config)
    if args.command == "experiment":
        return _experiment(args.config)
    if args.command == "summarize":
        return _summarize(args.results)

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark experiments.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run one benchmark run config.")
    run_parser.add_argument("--config", type=Path, required=True)

    experiment_parser = subparsers.add_parser(
        "experiment",
        help="Run all runs from an experiment config.",
    )
    experiment_parser.add_argument("--config", type=Path, required=True)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Print a summary for an experiment_result.json file.",
    )
    summarize_parser.add_argument("--results", type=Path, required=True)

    return parser


def _run(config_path: Path) -> int:
    try:
        config = load_run_config(config_path)
        result = ExperimentRunner().run(config)
    except (RunnerConfigError, OSError, ValidationError) as error:
        print(f"ERROR: {error}")
        return 1

    print(_format_run_result(result))
    return 1 if result.status is RunStatus.ERROR else 0


def _experiment(config_path: Path) -> int:
    try:
        config = load_experiment_config(config_path)
        result = BatchRunner().run_experiment(config)
    except (RunnerConfigError, OSError, ValidationError) as error:
        print(f"ERROR: {error}")
        return 1

    print(_format_experiment_result(result))
    return 1 if result.summary.get("error_runs", 0) else 0


def _summarize(results_path: Path) -> int:
    try:
        result = _load_experiment_result(results_path)
    except (OSError, ValidationError) as error:
        print(f"ERROR: {error}")
        return 1

    print(_format_experiment_result(result))
    return 0


def _load_experiment_result(path: Path) -> ExperimentResult:
    try:
        return ExperimentResult.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise OSError(f"Failed to read experiment result {path}: {error}") from error


def _format_run_result(result: RunResult) -> str:
    lines = [
        f"run_id: {result.run_id}",
        f"task_id: {result.task_id}",
        f"status: {result.status.value}",
        f"execution_mode: {result.execution_mode.value}",
        f"total_score: {_format_optional_float(result.total_score)}",
        f"overall_status: {result.overall_status or ''}",
        f"duration_sec: {_format_optional_float(result.duration_sec)}",
    ]
    if result.validation_result_path is not None:
        lines.append(f"validation_result_path: {result.validation_result_path}")
    if result.scene_snapshot_path is not None:
        lines.append(f"scene_snapshot_path: {result.scene_snapshot_path}")
    if result.error:
        lines.append(f"error: {result.error}")
    return "\n".join(lines)


def _format_experiment_result(result: ExperimentResult) -> str:
    summary = result.summary or aggregate_run_results(result.runs).model_dump(exclude={"metrics"})
    return "\n".join(
        [
            f"experiment_id: {result.experiment_id}",
            f"total_runs: {summary.get('total_runs', 0)}",
            f"passed: {summary.get('passed_runs', 0)}",
            f"failed: {summary.get('failed_runs', 0)}",
            f"error: {summary.get('error_runs', 0)}",
            f"average_score: {_format_optional_float(summary.get('average_score'))}",
        ]
    )


def _format_optional_float(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
