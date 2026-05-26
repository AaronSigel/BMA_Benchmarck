from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bma_benchmark.reporting.evidence_pack.completeness import write_completeness_check
from bma_benchmark.reporting.evidence_pack.sanity import run_validator_sanity_suite
from bma_benchmark.reporting.evidence_pack.selection import select_evidence_examples
from bma_benchmark.reporting.scene_examples.discovery import discover_runs
from bma_benchmark.reporting.scene_examples.models import SceneExampleSelectionConfig


def _write_run(
    root: Path,
    run_id: str,
    *,
    task_id: str,
    pass_type: str,
    score: float,
    strategy: str,
    issue_code: str | None = None,
) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checks = []
    if issue_code:
        checks.append({
            "validator_name": "transform_validator",
            "check_name": "location",
            "field": "location.x",
            "expected": 2.0,
            "actual": 0.0,
            "passed": False,
            "issue_code": issue_code,
        })
    validation = {
        "task_id": task_id,
        "total_score": score,
        "issues": [{"code": issue_code}] if issue_code else [],
        "check_table": checks,
    }
    (run_dir / "validation_result.json").write_text(json.dumps(validation), encoding="utf-8")
    (run_dir / "run_result.json").write_text(
        json.dumps({"run_id": run_id, "task_id": task_id, "status": pass_type, "total_score": score}),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps({"run_id": run_id, "task_id": task_id, "status": pass_type}),
        encoding="utf-8",
    )


def _build_mock_experiment(root: Path) -> None:
    rows = [
        ("run_clean_geo", "geometry_002_positions", "clean_pass", 0.98, "react"),
        ("run_clean_mat", "materials_004_multiple_objects", "clean_pass", 0.97, "plan_and_execute"),
        ("run_clean_light", "lighting_003_three_point_lighting", "clean_pass", 0.96, "react"),
        ("run_clean_cam", "camera_003_composition_view", "clean_pass", 0.99, "plan_execute_react_repair"),
        ("run_soft_export", "export_002_glb_file", "soft_pass", 0.91, "react"),
        ("run_fail_geo", "geometry_002_positions", "failed_validation", 0.4, "direct_tool_calling", "location_mismatch"),
        ("run_fail_mat", "materials_004_multiple_objects", "failed_validation", 0.3, "direct_tool_calling", "object_material_missing"),
        ("run_fail_light", "lighting_003_three_point_lighting", "failed_validation", 0.2, "direct_tool_calling", "light_missing"),
        ("run_fail_cam", "camera_003_composition_view", "failed_validation", 0.25, "direct_tool_calling", "camera_direction_mismatch"),
    ]
    summary_lines = [
        "run_id,task_id,model,strategy,mcp_profile,pass_type,score,duration_sec,tool_call_count,error_type",
    ]
    for item in rows:
        run_id, task_id, pass_type, score, strategy = item[:5]
        issue = item[5] if len(item) > 5 else None
        _write_run(root, run_id, task_id=task_id, pass_type=pass_type, score=score, strategy=strategy, issue_code=issue)
        summary_lines.append(
            f"{run_id},{task_id},google/gemini-2.5-flash-lite,{strategy},full,{pass_type},{score},1.0,3,"
        )
    (root / "summary.csv").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


class EvidencePackTests(unittest.TestCase):
    def test_sanity_suite_has_ten_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_validator_sanity_suite(Path(tmp) / "sanity", tasks_root=Path("tasks"))
            self.assertEqual(len(result.cases), 10)
            self.assertTrue(result.all_passed_as_expected)

    def test_evidence_selection_prefers_export_soft_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            root.mkdir()
            _build_mock_experiment(root)
            runs = discover_runs(root)
            bundle = select_evidence_examples(runs, SceneExampleSelectionConfig(examples_per_status=4))
            soft = [e for e in bundle.examples if e.pass_type == "soft_pass"]
            self.assertTrue(soft)
            self.assertEqual(soft[0].task_id, "export_002_glb_file")

    def test_check_excerpt_includes_expected_actual(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            root.mkdir()
            _build_mock_experiment(root)
            runs = discover_runs(root)
            bundle = select_evidence_examples(runs, SceneExampleSelectionConfig(examples_per_status=4))
            failed = [e for e in bundle.examples if e.pass_type == "failed_validation"]
            self.assertTrue(failed)
            row = failed[0].check_table_excerpt[0]
            self.assertIn("expected", row)
            self.assertIn("actual", row)

    def test_completeness_check_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            pack = Path(tmp) / "pack"
            root.mkdir()
            pack.mkdir()
            _build_mock_experiment(root)
            runs = discover_runs(root)
            bundle = select_evidence_examples(runs, SceneExampleSelectionConfig())
            payload = write_completeness_check(
                pack,
                runs=runs,
                examples=bundle.examples,
                sanity_result=run_validator_sanity_suite(Path(tmp) / "sanity", tasks_root=Path("tasks")),
                figure_paths={},
                table_paths={},
                expected_runs=9,
            )
            self.assertEqual(payload["demo_runs_found"], 9)
            self.assertEqual(payload["validator_sanity_cases_found"], 10)
            self.assertTrue((pack / "completeness_check.json").is_file())


if __name__ == "__main__":
    unittest.main()
