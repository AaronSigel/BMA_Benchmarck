from __future__ import annotations

import json
from pathlib import Path

from benchmark.analysis.comparison import analyze_run_results
from benchmark.analysis.export import write_run_metrics_csv
from benchmark.analysis.models import ReportConfig, RunAnalysisResult
from benchmark.analysis.report_bundle import create_report_bundle, write_figures, write_report_text_ru
from benchmark.analysis.report_bundle_validator import validate_report_bundle_result
from benchmark.analysis.report_builder import build_html_report, build_markdown_report


def _run(run_id: str = "run_clean_001") -> RunAnalysisResult:
    return RunAnalysisResult(
        run_id=run_id,
        task_id="geometry_002_positions",
        agent_id="agent",
        strategy="direct",
        model="model",
        mcp_profile="minimal",
        pass_type="clean_pass",
        run_status="passed",
        scene_status="passed",
        agent_status="completed",
        total_score=1.0,
        success=True,
    )


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "experiment"
    root.mkdir()
    run = _run()
    analysis = analyze_run_results([run], experiment_id="experiment")
    write_run_metrics_csv(analysis.runs, root / "summary.csv")
    (root / "summary.json").write_text(analysis.summary.model_dump_json(), encoding="utf-8")
    (root / "experiment_analysis.json").write_text(analysis.model_dump_json(), encoding="utf-8")
    (root / "report.md").write_text(build_markdown_report(analysis, ReportConfig(title="Test", output_dir=root)), encoding="utf-8")
    (root / "report.html").write_text(build_html_report(analysis, ReportConfig(title="Test", output_dir=root)), encoding="utf-8")
    write_report_text_ru(analysis, root / "report_text_ru.md")
    write_figures(analysis, root / "figures")
    (root / "manifest.json").write_text(json.dumps({
        "experiment_id": "experiment",
        "benchmark_protocol_version": "1.0",
        "task_schema_version": "1.0",
        "validator_version": "1.0",
        "tool_contract_version": "1.0",
        "report_schema_version": "1.0",
        "matrix_config_hash": "hash",
        "task_set_hash": "hash",
        "tool_contract_hash": "hash",
    }), encoding="utf-8")
    run_dir = root / run.run_id
    run_dir.mkdir()
    (run_dir / "artifact_manifest.json").write_text(json.dumps({
        "run_id": run.run_id,
        "task_id": run.task_id,
        "status": "clean_pass",
        "artifacts": {
            "run_result": {"path": "run_result.json", "exists": True, "required": True},
            "metrics": {"path": "metrics.json", "exists": True, "required": True},
        },
        "files": ["run_result.json", "metrics.json"],
    }), encoding="utf-8")
    (run_dir / "run_result.json").write_text(json.dumps({
        "run_id": run.run_id,
        "task_id": run.task_id,
        "status": "passed",
        "total_score": 1.0,
    }), encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    return create_report_bundle(root, analysis, [
        root / "summary.csv",
        root / "summary.json",
        root / "experiment_analysis.json",
        root / "report.md",
        root / "report.html",
        root / "report_text_ru.md",
    ])


def test_report_bundle_contains_validator_audit_and_scene_examples(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)

    assert (bundle / "validator_audit" / "validator_inventory.csv").is_file()
    assert (bundle / "scene_examples" / "scene_examples.json").is_file()


def test_validate_report_bundle_fails_when_required_extension_missing(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "validator_audit" / "validator_inventory.csv").unlink()

    result = validate_report_bundle_result(bundle, write_result=False)

    assert result["status"] == "failed"


def test_validate_report_bundle_allows_missing_scene_png_when_metadata_exists(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    for path in (bundle / "scene_examples").glob("*.png"):
        path.unlink()

    result = validate_report_bundle_result(bundle, write_result=False)

    assert result["status"] == "passed"
    assert any(check["status"] == "warning" for check in result["checks"])
