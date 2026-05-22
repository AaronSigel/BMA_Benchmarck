from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from benchmark.analysis.models import ExperimentAnalysisResult
from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.manifests import write_manifest_for_matrix
from benchmark.experiments.matrix import load_matrix
from benchmark.experiments.preflight import (
    PreflightError,
    collect_runtime_metadata,
    latest_artifact_mtime,
    prepare_output_root,
    run_contract_smoke_for_experiment,
    filter_config_by_profile_preflight,
    run_profile_preflight_for_experiment,
    write_preflight_report,
)
from benchmark.experiments.readiness import check_matrix_readiness
from benchmark.experiments.models import ExperimentMatrix
from benchmark.runner.models import ExperimentConfig, ExperimentResult


@dataclass(frozen=True)
class PreparedExperiment:
    matrix: ExperimentMatrix
    config: ExperimentConfig
    manifest_path: Path
    metadata: dict[str, Any]


class E2EBenchmarkRunner:
    """Stage 8 orchestration wrapper around the existing runner stack."""

    def __init__(self, batch_runner: object | None = None) -> None:
        self.batch_runner = batch_runner

    def prepare(self, matrix_path: Path | str) -> ExperimentConfig:
        return self._prepare(matrix_path).config

    def run(self, matrix_path: Path | str, *, fail_fast_profile_preflight: bool = False, resume: bool = False, clean_output: bool = False) -> ExperimentResult:
        prepared = self._prepare(matrix_path, fail_fast_profile_preflight=fail_fast_profile_preflight, resume=resume, clean_output=clean_output)
        return self._run_batch(prepared.config, resume=resume)

    def run_and_analyze(self, matrix_path: Path | str, *, fail_fast_profile_preflight: bool = False, resume: bool = False, clean_output: bool = False) -> ExperimentAnalysisResult:
        prepared = self._prepare(matrix_path, fail_fast_profile_preflight=fail_fast_profile_preflight, resume=resume, clean_output=clean_output)
        self._run_batch(prepared.config, resume=resume)
        return run_analysis(prepared.matrix.output_root)

    def run_and_report(self, matrix_path: Path | str, *, clean_output: bool = False, fail_fast_profile_preflight: bool = False, resume: bool = False) -> Path:
        prepared = self._prepare(matrix_path, clean_output=clean_output, run_contract_smoke=True, fail_fast_profile_preflight=fail_fast_profile_preflight, resume=resume)
        self._run_batch(prepared.config, resume=resume)
        _refresh_manifest(prepared)
        analysis = _run_analysis(prepared.matrix.output_root)
        return _build_reports(analysis, prepared.matrix)

    def _prepare(
        self,
        matrix_path: Path | str,
        *,
        clean_output: bool = False,
        run_contract_smoke: bool = False,
        fail_fast_profile_preflight: bool = False,
        resume: bool = False,
    ) -> PreparedExperiment:
        matrix = load_matrix(matrix_path)
        if resume:
            matrix = _matrix_for_resume(matrix)
        elif bool(matrix.metadata.get("timestamp_output_root")):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            matrix = matrix.model_copy(
                update={"output_root": Path("artifacts") / f"{matrix.matrix_id}_{stamp}"}
            )
        freshness = prepare_output_root(matrix.output_root, clean_output=clean_output, allow_existing=resume)
        readiness = check_matrix_readiness(matrix)
        if not readiness.ok:
            raise RuntimeError("; ".join(readiness.errors))
        config = generate_experiment_config(matrix)
        preflight: dict[str, Any] = {
            "artifact_freshness": freshness,
            "planned_runs": len(config.runs),
            "expected_runs": matrix.metadata.get("expected_runs"),
            "readiness_gates": matrix.metadata.get("readiness_gates"),
        }
        preflight_report = write_preflight_report(config, matrix.output_root)
        preflight["preflight_report"] = preflight_report
        preflight_enabled = bool(matrix.metadata.get("preflight", {}).get("enabled") if isinstance(matrix.metadata.get("preflight"), dict) else matrix.metadata.get("preflight_enabled"))
        if preflight_enabled or fail_fast_profile_preflight or run_contract_smoke:
            profile_preflight = run_profile_preflight_for_experiment(config)
            preflight["mcp_profile_preflight"] = profile_preflight
            preflight_cfg = matrix.metadata.get("preflight", {})
            preflight_mode = preflight_cfg.get("mode", "diagnostic") if isinstance(preflight_cfg, dict) else "diagnostic"
            preflight_fail_fast = bool(
                fail_fast_profile_preflight
                or (
                    isinstance(preflight_cfg, dict)
                    and (preflight_cfg.get("fail_fast") or preflight_cfg.get("fail_fast_on_profile_error"))
                    and preflight_mode == "strict"
                )
            )
            if not profile_preflight["ok"] and preflight_fail_fast:
                failed = [
                    item.get("profile")
                    for item in profile_preflight.get("profiles", [])
                    if item.get("preflight_status") != "passed"
                ]
                raise RuntimeError(f"MCP profile preflight failed: {', '.join(str(x) for x in failed)}")
            if preflight_mode == "strict" and preflight_fail_fast:
                config = filter_config_by_profile_preflight(config, profile_preflight)
            else:
                config = _mark_profile_preflight(config, profile_preflight)
            preflight["skipped_by_preflight"] = preflight["planned_runs"] - len(config.runs)
        if run_contract_smoke:
            contract_smoke = run_contract_smoke_for_experiment(config, matrix.output_root)
            preflight["mcp_contract_smoke"] = contract_smoke
            if not contract_smoke.get("ok"):
                contract_error = str(contract_smoke.get("error", "unknown"))
                if "execute_code failed" in contract_error or "reset" in contract_error.lower():
                    raise PreflightError(
                        "Blender harness scene reset failed during preflight: "
                        + contract_error
                    )
                _preflight_cfg = matrix.metadata.get("preflight") or {}
                _smoke_fail_fast = bool(
                    fail_fast_profile_preflight
                    or (isinstance(_preflight_cfg, dict) and _preflight_cfg.get("fail_fast"))
                )
                if _smoke_fail_fast:
                    raise PreflightError(
                        "Socket is reachable, but running server does not support expected BMA contract: "
                        + contract_error
                    )
        preflight["runtime"] = collect_runtime_metadata(config, matrix.output_root)
        manifest_path = write_manifest_for_matrix(
            matrix,
            matrix.output_root / "manifest.json",
            metadata=preflight,
        )
        return PreparedExperiment(
            matrix=matrix,
            config=config,
            manifest_path=manifest_path,
            metadata=preflight,
        )

    def _run_batch(self, config: ExperimentConfig, *, resume: bool = False) -> ExperimentResult:
        runner = self.batch_runner or _default_batch_runner()
        try:
            return runner.run_experiment(config, resume=resume)
        except TypeError as exc:
            if "resume" not in str(exc):
                raise
            return runner.run_experiment(config)


def _default_batch_runner():
    from benchmark.runner.batch_runner import BatchRunner

    return BatchRunner()


def _mark_profile_preflight(config: ExperimentConfig, preflight: dict[str, Any]) -> ExperimentConfig:
    failed = {
        item["profile"]: item.get("reason")
        for item in preflight.get("profiles", [])
        if item.get("preflight_status") != "passed" and item.get("profile")
    }
    if not failed:
        return config
    runs = []
    for run in config.runs:
        if run.mcp_profile in failed:
            runs.append(
                run.model_copy(
                    update={
                        "metadata": {
                            **run.metadata,
                            "profile_preflight_failed": True,
                            "profile_preflight_reason": failed[run.mcp_profile],
                        }
                    }
                )
            )
        else:
            runs.append(run)
    return config.model_copy(update={"runs": runs, "metadata": {**config.metadata, "profile_preflight": preflight}})


def run_analysis(output_root: Path) -> ExperimentAnalysisResult:
    import json

    from benchmark.analysis.comparison import analyze_experiment, build_infra_reliability_payload
    from benchmark.analysis.export import write_experiment_analysis_json, write_run_metrics_csv

    analysis = analyze_experiment(output_root)
    write_experiment_analysis_json(analysis, output_root / "experiment_analysis.json")
    write_run_metrics_csv(analysis.runs, output_root / "summary.csv")
    summary_payload = {
        **analysis.summary.model_dump(mode="json"),
        "infra_reliability": build_infra_reliability_payload(analysis.summary, analysis.metadata),
    }
    (output_root / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return analysis


def _refresh_manifest(prepared: PreparedExperiment) -> None:
    metadata = dict(prepared.metadata)
    runtime = collect_runtime_metadata(prepared.config, prepared.matrix.output_root)
    runtime["latest_run_file_mtime"] = latest_artifact_mtime(prepared.matrix.output_root)
    metadata["runtime"] = runtime
    write_manifest_for_matrix(
        prepared.matrix,
        prepared.manifest_path,
        metadata=metadata,
    )


def build_reports(analysis: ExperimentAnalysisResult, matrix: ExperimentMatrix) -> Path:
    from benchmark.analysis.report_bundle import create_report_bundle, write_figures, write_report_text_ru
    from benchmark.analysis.report_builder import build_html_report, build_markdown_report
    from benchmark.analysis.report_bundle_validator import validate_report_bundle_result

    report_config = _report_config(matrix)
    markdown_path = Path(report_config.output_dir) / "report.md"
    html_path = Path(report_config.output_dir) / "report.html"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown_report(analysis, report_config), encoding="utf-8")
    html_path.write_text(build_html_report(analysis, report_config), encoding="utf-8")
    text_path = Path(report_config.output_dir) / "report_text_ru.md"
    write_report_text_ru(analysis, text_path)
    write_figures(analysis, Path(report_config.output_dir) / "figures")
    bundle = create_report_bundle(
        Path(report_config.output_dir),
        analysis,
        [
            Path(report_config.output_dir) / "summary.csv",
            Path(report_config.output_dir) / "summary.json",
            Path(report_config.output_dir) / "experiment_analysis.json",
            markdown_path,
            html_path,
            text_path,
            Path(report_config.output_dir) / "preflight_report.json",
            Path(report_config.output_dir) / "preflight_report.md",
            Path(report_config.output_dir) / "resume_report.json",
            Path(report_config.output_dir) / "resume_report.md",
        ],
    )
    validation = validate_report_bundle_result(bundle)
    _append_bundle_validation_section(markdown_path, validation)
    bundle_report = bundle / "report.md"
    if bundle_report.exists():
        _append_bundle_validation_section(bundle_report, validation)
    return markdown_path


_run_analysis = run_analysis
_build_reports = build_reports


def _append_bundle_validation_section(path: Path, validation: dict[str, Any]) -> None:
    status = validation.get("status", "unknown")
    structural_validity = validation.get("structural_validity", "unknown")
    readiness_ok = validation.get("readiness_ok", "unknown")
    failed_gates = validation.get("failed_gates", [])
    warning_gates = validation.get("warning_gates", [])
    failed = sum(1 for check in validation.get("checks", []) if isinstance(check, dict) and check.get("status") == "failed")
    failed_gate_lines = ""
    if isinstance(failed_gates, list) and failed_gates:
        failed_gate_lines = "\n".join(
            f"| {gate.get('name')} | {gate.get('expected')} | {gate.get('actual')} | {gate.get('severity', 'blocking')} |"
            for gate in failed_gates
            if isinstance(gate, dict)
        )
        failed_gate_lines = (
            "\n\n### Failed readiness gates\n\n"
            "| Gate | Expected | Actual | Severity |\n| --- | --- | --- | --- |\n"
            + failed_gate_lines
        )
    warning_gate_lines = ""
    if isinstance(warning_gates, list) and warning_gates:
        warning_gate_lines = "\n".join(
            f"| {gate.get('name')} | {gate.get('expected')} | {gate.get('actual')} | {gate.get('severity', 'warning')} |"
            for gate in warning_gates
            if isinstance(gate, dict)
        )
        warning_gate_lines = (
            "\n\n### Warning readiness gates\n\n"
            "| Gate | Expected | Actual | Severity |\n| --- | --- | --- | --- |\n"
            + warning_gate_lines
        )
    section = (
        "\n## Bundle Validation\n\n"
        "| Metric | Value |\n| --- | --- |\n"
        f"| status | {status} |\n"
        f"| structural_validity | {structural_validity} |\n"
        f"| readiness_ok | {readiness_ok} |\n"
        f"| failed_checks | {failed} |\n"
        f"{failed_gate_lines}{warning_gate_lines}\n"
    )
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if "## Bundle Validation" in text:
        text = text.split("## Bundle Validation", 1)[0].rstrip() + "\n"
    path.write_text(text.rstrip() + section, encoding="utf-8")


def _matrix_for_resume(matrix: ExperimentMatrix) -> ExperimentMatrix:
    if not bool(matrix.metadata.get("timestamp_output_root")):
        return matrix
    parent = matrix.output_root.parent
    prefix = f"{matrix.matrix_id}_"
    candidates = sorted(
        (path for path in parent.glob(f"{prefix}*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return matrix.model_copy(update={"metadata": {**matrix.metadata, "timestamp_output_root": False}})
    return matrix.model_copy(
        update={
            "output_root": candidates[-1],
            "metadata": {**matrix.metadata, "timestamp_output_root": False, "resume_from": str(candidates[-1])},
        }
    )


def _report_config(matrix: ExperimentMatrix):
    from benchmark.analysis.models import ReportConfig

    data: dict[str, Any] = {}
    if matrix.report_config_path is not None:
        data = yaml.safe_load(Path(matrix.report_config_path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise RuntimeError(f"report_config_path must contain a YAML mapping: {matrix.report_config_path}")
    data.update(
        {
            "title": data.get("title", f"Experiment: {matrix.matrix_id}"),
            "input_dir": matrix.output_root,
            "output_dir": matrix.output_root,
            "formats": ["markdown", "html"],
        }
    )
    return ReportConfig(**data)
