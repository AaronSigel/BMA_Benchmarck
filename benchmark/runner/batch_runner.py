from pathlib import Path
import json

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
from benchmark.runner.artifact_manifest import validate_run_artifact_manifest
from benchmark.analysis.trace_reader import read_run_result


class BatchRunner:
    """Runs all entries in an ExperimentConfig sequentially."""

    def __init__(self, runner: ExperimentRunner | None = None) -> None:
        self.runner = runner or ExperimentRunner()

    def run_experiment(self, config: ExperimentConfig, *, resume: bool = False) -> ExperimentResult:
        results: list[RunResult] = []
        resume_status: dict[str, str] = {}
        resume_entries: list[dict[str, str]] = []
        for run_config in config.runs:
            if resume:
                existing, reason = _load_existing_run(run_config)
                if existing is not None:
                    results.append(existing)
                    resume_status[run_config.run_id] = "skipped_existing"
                    resume_entries.append({"run_id": run_config.run_id, "status": "completed_existing"})
                    continue
                resume_status[run_config.run_id] = reason
                resume_entries.append({"run_id": run_config.run_id, "status": reason})
            result = self.runner.run(run_config)
            if resume and resume_status.get(run_config.run_id) in {"rerun_missing", "rerun_incomplete", "rerun_corrupted"} and result.status.value == "error":
                resume_status[run_config.run_id] = "rerun_failed_again"
                resume_entries[-1]["status"] = "rerun_failed_again"
            elif resume:
                previous = resume_status.get(run_config.run_id, "rerun_missing")
                resume_status[run_config.run_id] = previous
                resume_entries[-1]["result_status"] = result.status.value
            results.append(result)

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
        if resume:
            (output_dir / "resume_status.json").write_text(
                json.dumps(resume_status, indent=2),
                encoding="utf-8",
            )
            _write_resume_report(output_dir, resume_entries)
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


def _load_existing_run(config: RunConfig) -> tuple[RunResult | None, str]:
    layout = RunArtifactLayout.from_run_output_dir(config.output_dir, config.run_id)
    run_dir = layout.run_dir()
    if not run_dir.exists() or not layout.artifact_manifest_json().exists():
        return None, "rerun_missing"
    ok, errors = validate_run_artifact_manifest(run_dir)
    if not ok:
        corrupt = any("invalid" in error for error in errors)
        return None, "rerun_corrupted" if corrupt else "rerun_incomplete"
    try:
        return read_run_result(layout.run_result_json()), "skipped_existing"
    except Exception:
        return None, "rerun_corrupted"


def _write_resume_report(output_dir: Path, entries: list[dict[str, str]]) -> None:
    counts: dict[str, int] = {}
    for entry in entries:
        status = entry.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    payload = {
        "total_runs": len(entries),
        "status_counts": counts,
        "runs": entries,
    }
    (output_dir / "resume_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Resume Report", "", "| Status | Count |", "| --- | --- |"]
    for status, count in sorted(counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "| Run | Status | Result |", "| --- | --- | --- |"])
    for entry in entries:
        lines.append(f"| {entry.get('run_id')} | {entry.get('status')} | {entry.get('result_status', '')} |")
    (output_dir / "resume_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
