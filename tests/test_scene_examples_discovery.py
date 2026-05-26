import json
from pathlib import Path

from bma_benchmark.reporting.scene_examples.discovery import discover_runs


def test_discovery_finds_run_result_and_artifact_manifest(tmp_path: Path) -> None:
    run = tmp_path / "run_clean_001"
    run.mkdir()
    (run / "run_result.json").write_text(json.dumps({
        "run_id": "run_clean_001",
        "task_id": "geometry_002_positions",
        "status": "passed",
        "total_score": 1.0,
    }), encoding="utf-8")
    (run / "artifact_manifest.json").write_text(json.dumps({
        "run_id": "run_clean_001",
        "task_id": "geometry_002_positions",
        "status": "clean_pass",
        "model": "m",
        "strategy": "s",
        "mcp_profile": "minimal",
        "artifacts": {"validation_result": {"path": "validation_result.json"}},
    }), encoding="utf-8")
    (run / "validation_result.json").write_text(json.dumps({
        "task_id": "geometry_002_positions",
        "overall_status": "passed",
        "total_score": 1.0,
        "validators": [],
        "issues": [],
        "summary": {},
    }), encoding="utf-8")

    refs = discover_runs(tmp_path)

    assert len(refs) == 1
    assert refs[0].run_id == "run_clean_001"
    assert refs[0].artifact_manifest["status"] == "clean_pass"
    assert refs[0].render_path is None
    assert refs[0].render_missing_reason
