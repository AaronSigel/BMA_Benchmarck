import argparse
from pathlib import Path

from pydantic import ValidationError

from benchmark.env import load_project_dotenv
from benchmark.metrics.aggregate import aggregate_run_results
from benchmark.runner.batch_runner import BatchRunner
from benchmark.runner.config_loader import load_experiment_config, load_run_config
from benchmark.runner.errors import RunnerConfigError
from benchmark.runner.experiment_runner import ExperimentRunner
from benchmark.runner.models import ExperimentConfig, ExperimentResult, RunResult, RunStatus
from benchmark.runner.paths import RunArtifactLayout


def main(argv: list[str] | None = None) -> int:
    load_project_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run(args.config)
    if args.command == "experiment":
        return _experiment(args)
    if args.command == "matrix":
        return _matrix(args)
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
    experiment_parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analysis layer after batch run; writes experiment_analysis.json.",
    )
    experiment_parser.add_argument(
        "--report",
        action="store_true",
        help="Build Markdown and HTML summary report (implies --analyze).",
    )

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Print a summary for an experiment_result.json file.",
    )
    summarize_parser.add_argument("--results", type=Path, required=True)

    matrix_parser = subparsers.add_parser(
        "matrix",
        help="Run a Stage 8 experiment matrix.",
    )
    matrix_parser.add_argument("--config", type=Path, required=True)
    matrix_parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run Stage 7 analysis after the matrix batch run.",
    )
    matrix_parser.add_argument(
        "--report",
        action="store_true",
        help="Build Markdown and HTML reports after the matrix batch run (implies --analyze).",
    )
    matrix_parser.add_argument(
        "--fail-fast-profile-preflight",
        action="store_true",
        help="Stop before running if any selected MCP profile fails preflight.",
    )
    matrix_parser.add_argument("--resume", action="store_true", help="Resume an existing matrix output directory.")
    matrix_parser.add_argument("--clean-output", action="store_true", help="Remove existing output before running.")
    matrix_parser.add_argument("--max-estimated-cost", type=float)
    matrix_parser.add_argument("--max-runtime-minutes", type=float)

    return parser


def _experiment_output_dir(config: ExperimentConfig) -> Path:
    """Derive the experiment-level output directory from run configs."""
    if not config.runs:
        return Path("artifacts") / "experiments"
    roots = {
        RunArtifactLayout.from_run_output_dir(run.output_dir, run.run_id).root
        for run in config.runs
    }
    return roots.pop() if len(roots) == 1 else RunArtifactLayout.from_run_output_dir(
        config.runs[0].output_dir, config.runs[0].run_id
    ).root


def _run_post_analysis(output_dir: Path, experiment_id: str, build_report: bool) -> None:
    """Run analysis layer and optionally build Markdown/HTML reports."""
    from benchmark.analysis.comparison import analyze_experiment
    from benchmark.analysis.export import write_experiment_analysis_json
    from benchmark.analysis.models import ReportConfig
    from benchmark.analysis.report_builder import build_html_report, build_markdown_report

    print("Running post-experiment analysis…")
    analysis = analyze_experiment(output_dir)
    out_json = output_dir / "experiment_analysis.json"
    write_experiment_analysis_json(analysis, out_json)
    print(f"  experiment_analysis.json → {out_json}")
    print(
        f"  runs={analysis.summary.total_runs}"
        f"  passed={analysis.summary.successful_runs}"
        f"  avg_score={analysis.summary.average_scene_score}"
    )

    if build_report:
        config = ReportConfig(
            title=f"Experiment: {experiment_id}",
            input_dir=output_dir,
            output_dir=output_dir,
            formats=["markdown", "html"],
        )
        md_path = output_dir / "report.md"
        md_path.write_text(build_markdown_report(analysis, config), encoding="utf-8")
        print(f"  report.md        → {md_path}")

        html_path = output_dir / "report.html"
        html_path.write_text(build_html_report(analysis, config), encoding="utf-8")
        print(f"  report.html      → {html_path}")


def _run(config_path: Path) -> int:
    try:
        config = load_run_config(config_path)
        result = ExperimentRunner().run(config)
    except (RunnerConfigError, OSError, ValidationError) as error:
        print(f"ERROR: {error}")
        return 1

    print(_format_run_result(result))
    return 1 if result.status is RunStatus.ERROR else 0


def _experiment(args: argparse.Namespace) -> int:
    config_path: Path = args.config
    do_analyze: bool = args.analyze or args.report
    do_report: bool = args.report

    try:
        config = load_experiment_config(config_path)
        result = BatchRunner().run_experiment(config)
    except (RunnerConfigError, OSError, ValidationError) as error:
        print(f"ERROR: {error}")
        return 1

    print(_format_experiment_result(result))

    if do_analyze:
        output_dir = _experiment_output_dir(config)
        try:
            _run_post_analysis(output_dir, config.experiment_id, do_report)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: post-analysis failed: {exc}")

    return 1 if result.summary.get("error_runs", 0) else 0


def _matrix(args: argparse.Namespace) -> int:
    from benchmark.experiments.e2e_runner import E2EBenchmarkRunner
    from benchmark.experiments.matrix import load_matrix

    config_path: Path = args.config
    try:
        runner = E2EBenchmarkRunner()
        matrix = load_matrix(config_path)
        report_ready_default = bool(matrix.metadata.get("report_ready_mvp"))
        if args.report or report_ready_default:
            report_path = runner.run_and_report(
                config_path,
                fail_fast_profile_preflight=args.fail_fast_profile_preflight,
                resume=args.resume,
                clean_output=args.clean_output,
            )
            print(f"report: {report_path}")
            return 0
        if args.analyze:
            analysis = runner.run_and_analyze(
                config_path,
                fail_fast_profile_preflight=args.fail_fast_profile_preflight,
                resume=args.resume,
                clean_output=args.clean_output,
            )
            print(
                "\n".join(
                    [
                        f"experiment_id: {analysis.experiment_id}",
                        f"total_runs: {analysis.summary.total_runs}",
                        f"passed: {analysis.summary.successful_runs}",
                        f"failed: {analysis.summary.failed_runs}",
                        f"error: {analysis.summary.error_runs}",
                    ]
                )
            )
            return 0
        result = runner.run(
            config_path,
            fail_fast_profile_preflight=args.fail_fast_profile_preflight,
            resume=args.resume,
            clean_output=args.clean_output,
        )
    except (RuntimeError, OSError, ValueError, ValidationError) as error:
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
