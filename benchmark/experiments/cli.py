from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.experiments.e2e_runner import E2EBenchmarkRunner
from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.matrix import load_matrix
from benchmark.experiments.readiness import check_matrix_readiness, write_readiness_result
from benchmark.runner.config_loader import dump_experiment_config


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        matrix = load_matrix(args.matrix)
        config = generate_experiment_config(matrix)
        dump_experiment_config(config, args.output)
        print(f"wrote {args.output}")
        return 0

    if args.command == "readiness":
        result = check_matrix_readiness(load_matrix(args.matrix))
        if args.output:
            write_readiness_result(result, args.output)
            print(f"wrote {args.output}")
        print("status: pass" if result.ok else "status: fail")
        for warning in result.warnings:
            print(f"WARNING: {warning}")
        for error in result.errors:
            print(f"ERROR: {error}")
        return 0 if result.ok else 1

    if args.command == "run":
        result = E2EBenchmarkRunner().run(args.matrix)
        print(result.model_dump_json(indent=2))
        return 0

    if args.command == "run-and-analyze":
        analysis = E2EBenchmarkRunner().run_and_analyze(args.matrix)
        print(analysis.model_dump_json(indent=2))
        return 0

    if args.command == "run-and-report":
        report_path = E2EBenchmarkRunner().run_and_report(args.matrix)
        print(f"report: {report_path}")
        return 0

    if args.command == "list-matrices":
        for matrix_path in _list_matrices(args.directory):
            matrix = load_matrix(matrix_path)
            print(f"{matrix.matrix_id}\t{matrix_path}\t{matrix.title or ''}")
        return 0

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage benchmark experiment matrices.")
    subparsers = parser.add_subparsers(dest="command")

    generate = subparsers.add_parser("generate", help="Generate an ExperimentConfig from a matrix.")
    generate.add_argument("--matrix", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)

    readiness = subparsers.add_parser("readiness", help="Check matrix readiness.")
    readiness.add_argument("--matrix", type=Path, required=True)
    readiness.add_argument("--output", type=Path)

    run = subparsers.add_parser("run", help="Run a matrix through the E2E orchestrator.")
    run.add_argument("--matrix", type=Path, required=True)

    run_and_analyze = subparsers.add_parser(
        "run-and-analyze",
        help="Run a matrix and write experiment_analysis.json.",
    )
    run_and_analyze.add_argument("--matrix", type=Path, required=True)

    run_and_report = subparsers.add_parser(
        "run-and-report",
        help="Run a matrix, analyze results, and build reports.",
    )
    run_and_report.add_argument("--matrix", type=Path, required=True)

    list_matrices = subparsers.add_parser("list-matrices", help="List matrix YAML files.")
    list_matrices.add_argument("--directory", type=Path, default=Path("configs/matrices"))

    return parser


def _list_matrices(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )


if __name__ == "__main__":
    raise SystemExit(main())
