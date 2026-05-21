from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark.analysis.models import ComparisonDimension, ReportConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml_config(path: Path) -> ReportConfig:
    import yaml  # type: ignore[import-untyped]
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ReportConfig(**raw)


def _print_comparison_table(report) -> None:  # type: ignore[no-untyped-def]
    """Print a ComparisonReport as a plain text table to stdout."""
    groups = report.groups
    if not groups:
        print("(no groups)")
        return

    headers = ["Value", "Runs", "Success%", "AvgScore", "AvgTools", "AvgDur(s)"]
    rows = [
        [
            g.value,
            str(g.run_count),
            f"{g.success_rate:.1%}" if g.success_rate is not None else "N/A",
            f"{g.avg_score:.4f}" if g.avg_score is not None else "N/A",
            f"{g.avg_tool_calls:.1f}" if g.avg_tool_calls is not None else "N/A",
            f"{g.avg_duration_sec:.1f}" if g.avg_duration_sec is not None else "N/A",
        ]
        for g in groups
    ]

    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_analyze_run(args: argparse.Namespace) -> int:
    from benchmark.analysis.run_analysis import analyze_run
    from benchmark.analysis.trace_reader import load_run_bundle

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: run-dir not found: {run_dir}", file=sys.stderr)
        return 1

    bundle = load_run_bundle(run_dir)
    result = analyze_run(bundle)

    out_dir = Path(args.output) if args.output else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "run_analysis.json"
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"run_analysis.json written to {out_path}")
    return 0


def cmd_analyze_experiment(args: argparse.Namespace) -> int:
    from benchmark.analysis.comparison import analyze_experiment
    from benchmark.analysis.export import write_experiment_analysis_json

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        print(f"Error: experiment-dir not found: {exp_dir}", file=sys.stderr)
        return 1

    result = analyze_experiment(exp_dir)

    out_dir = Path(args.output) if args.output else exp_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiment_analysis.json"
    write_experiment_analysis_json(result, out_path)
    print(f"experiment_analysis.json written to {out_path}")
    print(f"  Runs: {result.summary.total_runs}  Successful: {result.summary.successful_runs}")
    return 0


def cmd_build_report(args: argparse.Namespace) -> int:
    from benchmark.analysis.comparison import analyze_experiment
    from benchmark.analysis.export import (
        write_experiment_analysis_json,
        write_run_metrics_csv,
        write_group_comparison_csv,
        write_error_taxonomy_csv,
    )
    from benchmark.analysis.report_builder import build_markdown_report, build_html_report

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        return 1

    config = _load_yaml_config(config_path)

    # Allow CLI override of input/output dirs
    if args.input:
        from benchmark.analysis.models import ReportConfig as RC
        config = RC(**{**config.model_dump(), "input_dir": args.input})
    if args.output:
        from benchmark.analysis.models import ReportConfig as RC
        config = RC(**{**config.model_dump(), "output_dir": args.output})

    input_dir = Path(config.input_dir)
    if not input_dir.exists():
        print(f"Error: input_dir not found: {input_dir}", file=sys.stderr)
        return 1

    analysis = analyze_experiment(input_dir)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    created: list[str] = []

    if "json" in config.formats:
        p = out / "experiment_analysis.json"
        write_experiment_analysis_json(analysis, p)
        created.append(str(p))

    if "csv" in config.formats:
        p = out / "run_metrics.csv"
        write_run_metrics_csv(analysis.runs, p)
        created.append(str(p))

        # Group comparison CSV for each dimension
        from benchmark.analysis.comparison import compare_runs
        for dim in (ComparisonDimension.STRATEGY, ComparisonDimension.MODEL, ComparisonDimension.TASK_CATEGORY):
            report = compare_runs(analysis.runs, dim)
            p = out / f"comparison_{dim.value}.csv"
            write_group_comparison_csv(report.groups, p)
            created.append(str(p))

        # Error taxonomy CSV
        from collections import Counter
        ec: Counter[str] = Counter()
        for r in analysis.runs:
            for key, val in r.metrics.items():
                if key.startswith("error.") and isinstance(val, int) and val > 0:
                    ec[key[len("error."):]] += val
        if ec:
            p = out / "error_taxonomy.csv"
            write_error_taxonomy_csv(dict(ec), p)
            created.append(str(p))

    if "markdown" in config.formats:
        p = out / "report.md"
        p.write_text(build_markdown_report(analysis, config), encoding="utf-8")
        created.append(str(p))

    if "html" in config.formats:
        p = out / "report.html"
        p.write_text(build_html_report(analysis, config), encoding="utf-8")
        created.append(str(p))

    for path in created:
        print(f"  {path}")
    return 0


def cmd_validate_report_bundle(args: argparse.Namespace) -> int:
    from benchmark.analysis.report_bundle_validator import validate_report_bundle_result

    bundle = Path(args.bundle)
    result = validate_report_bundle_result(bundle)
    passed = result["status"] == "passed"

    failed_checks = [c for c in result["checks"] if c["status"] == "failed"]
    if failed_checks:
        for check in failed_checks:
            print(f"FAIL  {check['name']}: {check.get('message', '')}", file=sys.stderr)
    else:
        print(f"OK  bundle validated: {bundle}")

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"result written to {out}")

    return 0 if passed else 1


def cmd_compare_bundles(args: argparse.Namespace) -> int:
    from benchmark.analysis.bundle_compare import compare_bundles

    bundle_a = Path(args.bundle_a)
    bundle_b = Path(args.bundle_b)

    for p in (bundle_a, bundle_b):
        if not p.exists():
            print(f"Error: bundle not found: {p}", file=sys.stderr)
            return 1

    output_dir = Path(args.output) if args.output else None
    result = compare_bundles(bundle_a, bundle_b, output_dir=output_dir)

    out_root = output_dir or bundle_b.parent
    print(f"total_runs: a={result['total_runs']['a']}  b={result['total_runs']['b']}  delta={result['total_runs']['delta']:+d}")
    clean = result["clean_pass_rate"]
    print(f"clean_pass_rate: a={clean['a']:.1%}  b={clean['b']:.1%}  delta={clean['delta']:+.1%}")
    print(f"comparison_report.json → {out_root / 'comparison_report.json'}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from benchmark.analysis.comparison import analyze_experiment, compare_runs

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        return 1

    group_by = (args.group_by or "strategy").replace("-", "_")
    try:
        dimension = ComparisonDimension(group_by)
    except ValueError:
        valid = ", ".join(d.value for d in ComparisonDimension)
        print(f"Error: unknown dimension '{group_by}'. Valid: {valid}", file=sys.stderr)
        return 1

    analysis = analyze_experiment(input_dir)
    if not analysis.runs:
        print("No runs found.", file=sys.stderr)
        return 1

    report = compare_runs(analysis.runs, dimension)
    print(f"\nComparison by: {dimension.value}  ({len(analysis.runs)} runs)\n")
    _print_comparison_table(report)
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.analysis.cli",
        description="BMA Benchmark analysis CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # analyze-run
    p_ar = sub.add_parser("analyze-run", help="Analyze a single run directory")
    p_ar.add_argument("--run-dir", required=True, help="Path to the run directory")
    p_ar.add_argument("--output", help="Directory to write run_analysis.json (default: --run-dir)")

    # analyze-experiment
    p_ae = sub.add_parser("analyze-experiment", help="Analyze all runs under an experiment directory")
    p_ae.add_argument("--experiment-dir", required=True, help="Path to the experiment directory")
    p_ae.add_argument("--output", help="Directory to write experiment_analysis.json (default: --experiment-dir)")

    # build-report
    p_br = sub.add_parser("build-report", help="Build reports from a YAML config")
    p_br.add_argument("--config", required=True, help="Path to report config YAML")
    p_br.add_argument("--input", help="Override input_dir from config")
    p_br.add_argument("--output", help="Override output_dir from config")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare runs grouped by a dimension")
    p_cmp.add_argument("--input", required=True, help="Experiment directory")
    p_cmp.add_argument(
        "--group-by",
        default="strategy",
        choices=[d.value for d in ComparisonDimension],
        help="Grouping dimension (default: strategy)",
    )

    # validate-report-bundle
    p_vrb = sub.add_parser("validate-report-bundle", help="Validate a report bundle directory.")
    p_vrb.add_argument("--bundle", required=True, help="Path to the report bundle directory")
    p_vrb.add_argument("--output", help="Write full validation result JSON to this file")

    # compare-bundles
    p_cb = sub.add_parser("compare-bundles", help="Compare two report bundles and write a diff report.")
    p_cb.add_argument("--bundle-a", required=True, help="Path to the baseline bundle")
    p_cb.add_argument("--bundle-b", required=True, help="Path to the comparison bundle")
    p_cb.add_argument("--output", help="Directory to write comparison_report.json/.md (default: bundle-b parent)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "analyze-run": cmd_analyze_run,
        "analyze-experiment": cmd_analyze_experiment,
        "build-report": cmd_build_report,
        "compare": cmd_compare,
        "validate-report-bundle": cmd_validate_report_bundle,
        "compare-bundles": cmd_compare_bundles,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 0

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
