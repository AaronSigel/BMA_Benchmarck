from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from benchmark.analysis.models import ExperimentAnalysisResult
from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.manifests import write_manifest_for_matrix
from benchmark.experiments.matrix import load_matrix
from benchmark.experiments.preflight import (
    collect_runtime_metadata,
    latest_artifact_mtime,
    prepare_output_root,
    run_contract_smoke_for_experiment,
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

    def run(self, matrix_path: Path | str) -> ExperimentResult:
        prepared = self._prepare(matrix_path)
        return self._run_batch(prepared.config)

    def run_and_analyze(self, matrix_path: Path | str) -> ExperimentAnalysisResult:
        prepared = self._prepare(matrix_path)
        self._run_batch(prepared.config)
        return _run_analysis(prepared.matrix.output_root)

    def run_and_report(self, matrix_path: Path | str, *, clean_output: bool = False) -> Path:
        prepared = self._prepare(matrix_path, clean_output=clean_output, run_contract_smoke=True)
        self._run_batch(prepared.config)
        _refresh_manifest(prepared)
        analysis = _run_analysis(prepared.matrix.output_root)
        return _build_reports(analysis, prepared.matrix)

    def _prepare(
        self,
        matrix_path: Path | str,
        *,
        clean_output: bool = False,
        run_contract_smoke: bool = False,
    ) -> PreparedExperiment:
        matrix = load_matrix(matrix_path)
        freshness = prepare_output_root(matrix.output_root, clean_output=clean_output)
        readiness = check_matrix_readiness(matrix)
        if not readiness.ok:
            raise RuntimeError("; ".join(readiness.errors))
        config = generate_experiment_config(matrix)
        preflight: dict[str, Any] = {
            "artifact_freshness": freshness,
        }
        if run_contract_smoke:
            preflight["mcp_contract_smoke"] = run_contract_smoke_for_experiment(
                config,
                matrix.output_root,
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

    def _run_batch(self, config: ExperimentConfig) -> ExperimentResult:
        runner = self.batch_runner or _default_batch_runner()
        return runner.run_experiment(config)


def _default_batch_runner():
    from benchmark.runner.batch_runner import BatchRunner

    return BatchRunner()


def _run_analysis(output_root: Path) -> ExperimentAnalysisResult:
    from benchmark.analysis.comparison import analyze_experiment
    from benchmark.analysis.export import write_experiment_analysis_json

    analysis = analyze_experiment(output_root)
    write_experiment_analysis_json(analysis, output_root / "experiment_analysis.json")
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


def _build_reports(analysis: ExperimentAnalysisResult, matrix: ExperimentMatrix) -> Path:
    from benchmark.analysis.report_builder import build_html_report, build_markdown_report

    report_config = _report_config(matrix)
    markdown_path = Path(report_config.output_dir) / "report.md"
    html_path = Path(report_config.output_dir) / "report.html"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown_report(analysis, report_config), encoding="utf-8")
    html_path.write_text(build_html_report(analysis, report_config), encoding="utf-8")
    return markdown_path


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
