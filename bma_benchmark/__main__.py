from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark.env import load_project_dotenv


def main() -> int:
    load_project_dotenv()
    parser = _parser()
    args = parser.parse_args(sys.argv[1:])

    if args.command == "run-matrix":
        from benchmark.runner.cli import main as runner_main

        argv = ["matrix", "--config", str(args.config)]
        if args.resume:
            argv.append("--resume")
        if args.clean_output:
            argv.append("--clean-output")
        if args.report:
            argv.append("--report")
        if args.analyze:
            argv.append("--analyze")
        return runner_main(argv)

    if args.command == "preflight":
        from benchmark.experiments.generator import generate_experiment_config
        from benchmark.experiments.matrix import load_matrix
        from benchmark.experiments.preflight import write_preflight_report

        matrix = load_matrix(args.config)
        config = generate_experiment_config(matrix)
        report = write_preflight_report(config, matrix.output_root)
        print(f"preflight: {matrix.output_root / 'preflight_report.json'}")
        return 0 if report.get("status") != "failed" else 1

    if args.command == "analyze":
        from benchmark.experiments.e2e_runner import run_analysis

        analysis = run_analysis(args.input)
        print(f"experiment_analysis: {args.input / 'experiment_analysis.json'}")
        print(f"runs: {analysis.summary.total_runs}")
        return 0

    if args.command == "build-report":
        from benchmark.experiments.e2e_runner import build_reports, run_analysis
        from benchmark.experiments.models import ExperimentMatrix

        matrix = ExperimentMatrix(matrix_id=args.input.name, output_root=args.input, report_config_path=args.config)
        report = build_reports(run_analysis(args.input), matrix)
        print(f"report: {report}")
        return 0

    if args.command in {"validate-report-bundle", "validate-bundle"}:
        from benchmark.analysis.report_bundle_validator import validate_report_bundle_result

        bundle = args.bundle if args.command == "validate-report-bundle" else args.input
        result = validate_report_bundle_result(bundle)
        if result["status"] != "passed":
            for check in result["checks"]:
                if check.get("status") == "failed":
                    print(f"ERROR: {check.get('message') or check.get('name')}")
            return 1
        print("report_bundle valid")
        return 0

    if args.command == "compare-bundles":
        from benchmark.analysis.bundle_compare import compare_bundles

        result = compare_bundles(args.bundle_a, args.bundle_b, args.output)
        out = Path(args.output) if args.output else args.bundle_b.parent
        print(f"comparison: {out / 'comparison_report.md'}")
        print(f"total_runs_delta: {result['total_runs']['delta']}")
        return 0

    if args.command == "list-strategies":
        from benchmark.agent.strategies.registry import STRATEGY_REGISTRY

        for entry in STRATEGY_REGISTRY.entries():
            print(f"{entry['name']}\t{entry['class']}")
        return 0

    parser.print_help()
    return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bma-benchmark")
    sub = parser.add_subparsers(dest="command")

    run_matrix = sub.add_parser("run-matrix")
    run_matrix.add_argument("--config", type=Path, required=True)
    run_matrix.add_argument("--resume", action="store_true")
    run_matrix.add_argument("--clean-output", action="store_true")
    run_matrix.add_argument("--analyze", action="store_true")
    run_matrix.add_argument("--report", action="store_true")

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--config", type=Path, required=True)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--input", type=Path, required=True)

    build_report = sub.add_parser("build-report")
    build_report.add_argument("--input", type=Path, required=True)
    build_report.add_argument("--config", type=Path, default=Path("configs/reports/default_report.yaml"))

    validate = sub.add_parser("validate-report-bundle")
    validate.add_argument("bundle", type=Path)

    validate_alias = sub.add_parser("validate-bundle")
    validate_alias.add_argument("--input", type=Path, required=True)

    compare = sub.add_parser("compare-bundles")
    compare.add_argument("bundle_a", type=Path)
    compare.add_argument("bundle_b", type=Path)
    compare.add_argument("--output", type=Path)

    sub.add_parser("list-strategies")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
