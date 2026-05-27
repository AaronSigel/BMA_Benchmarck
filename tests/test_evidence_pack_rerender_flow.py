from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bma_benchmark.reporting.evidence_pack.builder import _refresh_example_images, build_evidence_pack
from bma_benchmark.reporting.rendering.render_plan import RenderedSceneArtifacts
from bma_benchmark.reporting.scene_examples.models import SceneExample


def _write_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(120, 180, 220)).save(path)


class EvidencePackRerenderFlowTests(unittest.TestCase):
    def test_refresh_example_images_fills_viewport_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run1"
            run.mkdir()
            _write_png(run / "viewport.png")
            example = SceneExample(
                run_id="run1",
                task_id="geometry_002_positions",
                pass_type="clean_pass",
                run_dir=run,
                selection_reason="test",
                render_missing_reason="no render/viewport image found",
            )
            _refresh_example_images(example, run)
            self.assertEqual(example.viewport_path, run / "viewport.png")
            self.assertIsNone(example.render_missing_reason)

    @patch("bma_benchmark.reporting.evidence_pack.builder.render_scene_artifacts")
    def test_build_evidence_pack_calls_rerender_when_enabled(self, mock_render) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "experiment"
            out = Path(tmp) / "out"
            run = experiment / "run_clean"
            run.mkdir(parents=True)
            (run / "final_scene.blend").write_bytes(b"blend")
            (run / "run_result.json").write_text(
                json.dumps({"run_id": "run_clean", "task_id": "geometry_002_positions", "status": "passed"}),
                encoding="utf-8",
            )
            (run / "artifact_manifest.json").write_text(
                json.dumps({"run_id": "run_clean", "task_id": "geometry_002_positions", "status": "clean_pass"}),
                encoding="utf-8",
            )
            (run / "metrics.json").write_text("{}", encoding="utf-8")
            (run / "validation_result.json").write_text(
                json.dumps({"overall_status": "passed", "total_score": 1.0, "issues": [], "check_table": []}),
                encoding="utf-8",
            )
            (experiment / "summary.csv").write_text(
                "run_id,task_id,model,strategy,mcp_profile,pass_type,score\n"
                "run_clean,geometry_002_positions,m,s,full,clean_pass,1.0\n",
                encoding="utf-8",
            )

            def _render(run_dir, **kwargs):
                _write_png(run_dir / "viewport.png")
                return RenderedSceneArtifacts(
                    run_id=run_dir.name,
                    run_dir=run_dir,
                    viewport_path=run_dir / "viewport.png",
                    status="rendered",
                )

            mock_render.side_effect = _render
            result = build_evidence_pack(experiment, out, render_missing_with_blender=True)
            self.assertTrue(mock_render.called)
            self.assertGreaterEqual(result["selected_examples_with_images"], 1)

    @patch("bma_benchmark.reporting.evidence_pack.builder.render_scene_artifacts")
    def test_build_evidence_pack_records_render_missing_reason(self, mock_render) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "experiment"
            out = Path(tmp) / "out"
            run = experiment / "run_fail"
            run.mkdir(parents=True)
            (run / "run_result.json").write_text(
                json.dumps({"run_id": "run_fail", "task_id": "geometry_002_positions", "status": "failed"}),
                encoding="utf-8",
            )
            (run / "artifact_manifest.json").write_text(
                json.dumps({"run_id": "run_fail", "task_id": "geometry_002_positions", "status": "failed_validation"}),
                encoding="utf-8",
            )
            (run / "metrics.json").write_text("{}", encoding="utf-8")
            (run / "validation_result.json").write_text(
                json.dumps(
                    {
                        "overall_status": "failed",
                        "total_score": 0.2,
                        "issues": [{"code": "location_mismatch"}],
                        "check_table": [
                            {
                                "check_name": "location",
                                "expected": 1,
                                "actual": 0,
                                "passed": False,
                                "issue_code": "location_mismatch",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (experiment / "summary.csv").write_text(
                "run_id,task_id,model,strategy,mcp_profile,pass_type,score\n"
                "run_fail,geometry_002_positions,m,direct_tool_calling,full,failed_validation,0.2\n",
                encoding="utf-8",
            )
            mock_render.return_value = RenderedSceneArtifacts(
                run_id="run_fail",
                run_dir=run,
                status="failed",
                reason="no .blend or .glb source for visual rendering",
            )
            result = build_evidence_pack(experiment, out, render_missing_with_blender=True)
            self.assertTrue(mock_render.called)
            self.assertFalse(result["visual_evidence_complete"])


if __name__ == "__main__":
    unittest.main()
