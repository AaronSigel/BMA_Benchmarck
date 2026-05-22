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
        from benchmark.mcp.socket_watchdog import get_watchdog_counters, increment_runs_since_restart, reset_watchdog_counters
        reset_watchdog_counters()
        worker_lifecycle = config.metadata.get("worker_lifecycle") if isinstance(config.metadata.get("worker_lifecycle"), dict) else {}
        restart_every_n = int(worker_lifecycle.get("restart_every_n_runs", 0) or 0)
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
            if _should_retry_run_on_infra_failure(run_config, worker_lifecycle, result):
                _restart_worker_for_run(run_config, worker_lifecycle)
                original = result
                result = self.runner.run(run_config)
                result = _annotate_infra_retry(result, original=original)
            if resume and resume_status.get(run_config.run_id) in {"rerun_missing", "rerun_incomplete", "rerun_corrupted"} and result.status.value == "error":
                resume_status[run_config.run_id] = "rerun_failed_again"
                resume_entries[-1]["status"] = "rerun_failed_again"
            elif resume:
                previous = resume_status.get(run_config.run_id, "rerun_missing")
                resume_status[run_config.run_id] = previous
                resume_entries[-1]["result_status"] = result.status.value
            results.append(result)
            if not resume or resume_status.get(run_config.run_id) != "skipped_existing":
                increment_runs_since_restart()
                if restart_every_n > 0 and run_config.mcp_config_path is not None:
                    _maybe_proactive_worker_restart(run_config, restart_every_n, worker_lifecycle)

        metrics_summary = aggregate_run_results(results)
        from benchmark.mcp.socket_watchdog import get_runs_since_last_restart
        watchdog_counters = get_watchdog_counters()
        result = ExperimentResult(
            experiment_id=config.experiment_id,
            runs=results,
            summary={
                **metrics_summary.model_dump(exclude={"metrics"}),
                **watchdog_counters.to_dict(),
                "runs_since_last_restart": get_runs_since_last_restart(),
            },
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


def _maybe_proactive_worker_restart(
    run_config: RunConfig,
    every_n_runs: int,
    worker_lifecycle: dict,
) -> None:
    from benchmark.mcp.config import McpServerConfig
    from benchmark.mcp.server_adapter import ExternalBlenderMcpServerAdapter
    from benchmark.mcp.socket_watchdog import BlenderSocketWatchdog, get_runs_since_last_restart
    import yaml

    if get_runs_since_last_restart() < every_n_runs:
        return
    if run_config.mcp_config_path is None:
        return
    raw = yaml.safe_load(run_config.mcp_config_path.read_text(encoding="utf-8")) or {}
    mcp_config = McpServerConfig(**{k: v for k, v in raw.items() if k != "env"})
    watchdog = BlenderSocketWatchdog(
        mcp_config,
        mcp_config_path=run_config.mcp_config_path,
        worker_lifecycle=worker_lifecycle,
    )
    adapter = ExternalBlenderMcpServerAdapter(mcp_config, watchdog=watchdog)
    watchdog.proactive_restart_if_due(every_n_runs, adapter=adapter)


def _run_has_infra_failure(result: RunResult) -> bool:
    summary = result.summary if isinstance(result.summary, dict) else {}
    structured = summary.get("structured_error")
    if isinstance(structured, dict) and structured.get("is_infra_failure"):
        return True
    return False


def _should_retry_run_on_infra_failure(
    run_config: RunConfig,
    worker_lifecycle: dict,
    result: RunResult,
) -> bool:
    max_retries = int(worker_lifecycle.get("retry_run_on_infra_failure", 0) or 0)
    if max_retries <= 0:
        return False
    if result.status.value not in {"error", "failed"}:
        return False
    if not _run_has_infra_failure(result):
        return False
    max_duration = float(worker_lifecycle.get("retry_run_max_duration_sec", 300) or 300)
    duration = result.duration_sec if result.duration_sec is not None else 0.0
    if duration > max_duration:
        return False
    return run_config.mcp_config_path is not None


def _restart_worker_for_run(run_config: RunConfig, worker_lifecycle: dict) -> None:
    from benchmark.mcp.config import McpServerConfig
    from benchmark.mcp.server_adapter import ExternalBlenderMcpServerAdapter
    from benchmark.mcp.socket_watchdog import BlenderSocketWatchdog
    import yaml

    if run_config.mcp_config_path is None:
        return
    raw = yaml.safe_load(run_config.mcp_config_path.read_text(encoding="utf-8")) or {}
    mcp_config = McpServerConfig(**{k: v for k, v in raw.items() if k != "env"})
    watchdog = BlenderSocketWatchdog(
        mcp_config,
        mcp_config_path=run_config.mcp_config_path,
        worker_lifecycle=worker_lifecycle,
    )
    adapter = ExternalBlenderMcpServerAdapter(mcp_config, watchdog=watchdog)
    if watchdog.restart_worker(reason="infra_run_retry"):
        watchdog.verify_after_restart(adapter)


def _annotate_infra_retry(result: RunResult, *, original: RunResult) -> RunResult:
    summary = dict(result.summary or {})
    original_structured = (original.summary or {}).get("structured_error")
    original_error_type = None
    if isinstance(original_structured, dict):
        original_error_type = original_structured.get("error_type")
    summary["infra_retry"] = {
        "attempt": 2,
        "original_error_type": original_error_type,
        "original_status": original.status.value,
    }
    return result.model_copy(update={"summary": summary})


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
        "resume_enabled": True,
        "total_runs": len(entries),
        "completed_existing": counts.get("completed_existing", 0),
        "skipped_existing": counts.get("skipped_existing", 0),
        "rerun_missing": counts.get("rerun_missing", 0),
        "rerun_incomplete": counts.get("rerun_incomplete", 0),
        "rerun_corrupted": counts.get("rerun_corrupted", 0),
        "rerun_failed_again": counts.get("rerun_failed_again", 0),
        "runs": entries,
    }
    (output_dir / "resume_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Resume Report", "", "| Status | Count |", "| --- | --- |"]
    for key in ("completed_existing", "skipped_existing", "rerun_missing", "rerun_incomplete", "rerun_corrupted", "rerun_failed_again"):
        count = counts.get(key, 0)
        if count:
            lines.append(f"| {key} | {count} |")
    for status, count in sorted(counts.items()):
        if status not in ("completed_existing", "skipped_existing", "rerun_missing", "rerun_incomplete", "rerun_corrupted", "rerun_failed_again"):
            lines.append(f"| {status} | {count} |")
    lines.extend(["", "| Run | Status | Result |", "| --- | --- | --- |"])
    for entry in entries:
        lines.append(f"| {entry.get('run_id')} | {entry.get('status')} | {entry.get('result_status', '')} |")
    (output_dir / "resume_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
