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

    if args.command == "lighting-report":
        import json as json_module

        from benchmark.analysis.run_analysis import summarize_lighting_failures

        task_ids = [item.strip() for item in args.tasks.split(",") if item.strip()] if args.tasks else None
        report = summarize_lighting_failures(args.input, task_ids=task_ids, sample_limit=args.sample_limit)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json_module.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"lighting_report: {args.output}")
        else:
            print(json_module.dumps(report, indent=2, ensure_ascii=False))
        return 0

    if args.command == "build-report":
        from benchmark.experiments.e2e_runner import build_reports, run_analysis
        from benchmark.experiments.models import ExperimentMatrix

        matrix = ExperimentMatrix(matrix_id=args.input.name, output_root=args.input, report_config_path=args.config)
        report = build_reports(run_analysis(args.input), matrix)
        print(f"report: {report}")
        return 0

    if args.command == "audit-validators":
        from bma_benchmark.validation_audit.collector import collect_validator_audit
        from bma_benchmark.validation_audit.writers import write_validator_audit

        report = collect_validator_audit(args.tasks_dir)
        write_validator_audit(report, args.out)
        print(f"validator_audit: {args.out}")
        print(f"rows: {len(report.rows)}")
        return 0

    if args.command == "build-scene-gallery":
        from bma_benchmark.reporting.scene_examples.cli import build_scene_gallery

        return build_scene_gallery(args.input, args.out, args.examples_per_status)

    if args.command == "build-evidence-pack":
        from bma_benchmark.reporting.evidence_pack.cli import main as evidence_main

        return evidence_main([
            "--experiment", str(args.experiment),
            "--out", str(args.out),
            *(["--config", str(args.config)] if args.config else []),
            *(["--tasks-dir", str(args.tasks_dir)] if args.tasks_dir else []),
        ])

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

    if args.command == "merge-runs":
        from benchmark.experiments.merge_runs import merge_experiment_runs

        result = merge_experiment_runs(
            base=args.base,
            replacement=args.replacement,
            output=args.output,
            replace_agent=args.replace_agent,
            replacement_reason=args.replacement_reason,
            rebuild_reports=not args.no_report,
        )
        print(f"merged: {result['output_root']}")
        print(f"total_runs: {result['total_runs']}")
        if result.get("report_bundle"):
            print(f"report_bundle: {result['report_bundle']}")
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

    lighting_report = sub.add_parser("lighting-report")
    lighting_report.add_argument("--input", type=Path, required=True)
    lighting_report.add_argument(
        "--tasks",
        type=str,
        default="lighting_001_area_light,lighting_003_three_point_lighting",
    )
    lighting_report.add_argument("--output", type=Path)
    lighting_report.add_argument("--sample-limit", type=int, default=5)

    build_report = sub.add_parser("build-report")
    build_report.add_argument("--input", type=Path, required=True)
    build_report.add_argument("--config", type=Path, default=Path("configs/reports/default_report.yaml"))

    audit_validators = sub.add_parser("audit-validators")
    audit_validators.add_argument("--tasks-dir", type=Path, required=True)
    audit_validators.add_argument("--out", type=Path, required=True)

    scene_gallery = sub.add_parser("build-scene-gallery")
    scene_gallery.add_argument("--input", type=Path, required=True)
    scene_gallery.add_argument("--out", type=Path, required=True)
    scene_gallery.add_argument("--examples-per-status", type=int, default=4)

    evidence_pack = sub.add_parser("build-evidence-pack")
    evidence_pack.add_argument("--experiment", type=Path, required=True)
    evidence_pack.add_argument("--out", type=Path, default=Path("artifacts/report_evidence_pack"))
    evidence_pack.add_argument("--config", type=Path, default=None)
    evidence_pack.add_argument("--tasks-dir", type=Path, default=Path("tasks"))

    validate = sub.add_parser("validate-report-bundle")
    validate.add_argument("bundle", type=Path)

    validate_alias = sub.add_parser("validate-bundle")
    validate_alias.add_argument("--input", type=Path, required=True)

    compare = sub.add_parser("compare-bundles")
    compare.add_argument("bundle_a", type=Path)
    compare.add_argument("bundle_b", type=Path)
    compare.add_argument("--output", type=Path)

    merge_runs = sub.add_parser("merge-runs")
    merge_runs.add_argument("--base", type=Path, required=True)
    merge_runs.add_argument("--replacement", type=Path, required=True)
    merge_runs.add_argument("--replace-agent", type=str, required=True)
    merge_runs.add_argument("--output", type=Path, required=True)
    merge_runs.add_argument("--replacement-reason", type=str, default=None)
    merge_runs.add_argument("--no-report", action="store_true")

    sub.add_parser("list-strategies")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
