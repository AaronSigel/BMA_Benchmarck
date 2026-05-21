from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark.analysis.comparison import analyze_run_results
from benchmark.analysis.export import write_run_metrics_csv
from benchmark.analysis.models import RunAnalysisResult
from benchmark.analysis.report_bundle import create_report_bundle, write_figures, write_report_text_ru
from benchmark.analysis.report_builder import build_markdown_report
from benchmark.analysis.run_analysis import _classify_pass_type
from benchmark.analysis.models import ReportConfig
from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.matrix import load_matrix


def _run(run_id: str, pass_type: str, **kwargs) -> RunAnalysisResult:
    return RunAnalysisResult(
        run_id=run_id,
        task_id=kwargs.get("task_id", "geometry_001_basic_primitives"),
        agent_id=kwargs.get("agent_id", "a"),
        strategy=kwargs.get("strategy", "plan_and_execute"),
        model="google/gemini-2.5-flash-lite",
        mcp_profile=kwargs.get("mcp_profile", "minimal"),
        pass_type=pass_type,
        run_status=kwargs.get("run_status", "passed"),
        scene_status=kwargs.get("scene_status", "passed"),
        agent_status=kwargs.get("agent_status", "completed"),
        total_score=kwargs.get("score", 1.0),
        duration_sec=kwargs.get("duration_sec", 2.0),
        tool_call_count=3,
        llm_call_count=1,
        invalid_tool_call_count=0,
        metrics=kwargs.get("metrics", {"provider_cost_available": True, "provider_reported_cost_usd": 0.001}),
        issues=kwargs.get("issues", []),
        artifacts=kwargs.get("artifacts", []),
    )


def test_pass_type_classifier_uses_report_ready_values() -> None:
    assert _classify_pass_type("passed", "passed", "completed", []) == "clean_pass"
    assert _classify_pass_type("passed", "passed", "completed", [{"code": "object_missing"}]) == "soft_pass"
    assert _classify_pass_type("failed", "failed", "completed", []) == "failed_validation"
    assert _classify_pass_type("error", "not_available", "runtime_error", []) == "runtime_error"


def test_status_counts_sum_to_total_runs() -> None:
    analysis = analyze_run_results([
        _run("r1", "clean_pass"),
        _run("r2", "soft_pass"),
        _run("r3", "failed_validation", run_status="failed", scene_status="failed"),
        _run("r4", "runtime_error", run_status="error", scene_status="not_available"),
    ])
    s = analysis.summary
    assert s.clean_pass_count + s.soft_pass_count + s.failed_validation_count + s.runtime_error_count == s.total_runs
    assert s.reported_success_rate == 0.5
    assert s.strict_success_rate == 0.25
    assert s.failure_rate == 0.5


def test_summary_csv_contains_report_ready_columns_and_issues(tmp_path: Path) -> None:
    out = tmp_path / "summary.csv"
    run = _run(
        "r1",
        "failed_validation",
        task_id="export_002_glb_file",
        run_status="failed",
        scene_status="failed",
        issues=[{"code": "export_missing"}, {"code": "material_missing"}],
        artifacts=[str(tmp_path / "r1" / "run_result.json")],
    )
    write_run_metrics_csv([run], out)
    row = next(csv.DictReader(out.open(encoding="utf-8")))
    for column in (
        "pass_type",
        "run_status",
        "scene_status",
        "agent_status",
        "score",
        "validation_issues",
        "agent_issues",
        "tool_issues",
        "export_issues",
        "all_issues",
        "export_status",
        "export_failure_type",
        "artifact_dir",
    ):
        assert column in row
    assert row["pass_type"] == "failed_validation"
    assert row["export_failure_type"] == "export_file_missing"


def test_report_bundle_contains_required_files_and_figures(tmp_path: Path) -> None:
    analysis = analyze_run_results([
        _run("r1", "clean_pass", strategy="plan_and_execute"),
        _run("r2", "runtime_error", strategy="react", run_status="error", scene_status="not_available"),
    ], experiment_id="diagnostic_repeat_gemini_v5")
    root = tmp_path / "diagnostic_repeat_gemini_v5_20260521_120000"
    root.mkdir()
    (root / "summary.json").write_text(analysis.summary.model_dump_json(), encoding="utf-8")
    (root / "experiment_analysis.json").write_text(analysis.model_dump_json(), encoding="utf-8")
    write_run_metrics_csv(analysis.runs, root / "summary.csv")
    (root / "report.md").write_text(build_markdown_report(analysis, ReportConfig(title="MVP", output_dir=root)), encoding="utf-8")
    (root / "report.html").write_text("<html></html>", encoding="utf-8")
    write_report_text_ru(analysis, root / "report_text_ru.md")
    write_figures(analysis, root / "figures")
    bundle = create_report_bundle(root, analysis, [
        root / "summary.csv",
        root / "summary.json",
        root / "experiment_analysis.json",
        root / "report.md",
        root / "report.html",
        root / "report_text_ru.md",
    ])
    for name in ("summary.csv", "summary.json", "experiment_analysis.json", "report.md", "report.html", "report_text_ru.md", "manifest.json", "README_REPORT.md"):
        assert (bundle / name).is_file()
    for name in ("success_by_strategy.png", "success_by_profile.png", "success_by_category.png", "top_validation_issues.png", "cost_by_strategy.png"):
        assert (bundle / "figures" / name).stat().st_size > 0
    assert "pass_type" in (bundle / "README_REPORT.md").read_text(encoding="utf-8")


def test_diagnostic_repeat_gemini_v5_generates_270_runs() -> None:
    matrix = load_matrix("configs/matrices/diagnostic_repeat_gemini_v5.yaml")
    config = generate_experiment_config(matrix)
    assert matrix.metadata["expected_runs"] == 270
    assert matrix.metadata["report_ready_mvp"] is True
    assert matrix.metadata["timestamp_output_root"] is True
    assert len(config.runs) == 270
