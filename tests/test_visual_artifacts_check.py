from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bma_benchmark.reporting.visual_artifacts.check import collect_visual_artifact_rows, write_visual_artifacts_check


def _write_run(root: Path, run_id: str, *, task_id: str, pass_type: str) -> None:
    run = root / run_id
    run.mkdir(parents=True)
    (run / "run_result.json").write_text(
        json.dumps({"run_id": run_id, "task_id": task_id, "status": pass_type}),
        encoding="utf-8",
    )
    (run / "artifact_manifest.json").write_text(
        json.dumps({"run_id": run_id, "task_id": task_id, "status": pass_type}),
        encoding="utf-8",
    )
    (run / "metrics.json").write_text("{}", encoding="utf-8")


class VisualArtifactsCheckTests(unittest.TestCase):
    def test_visual_artifacts_check_outputs_and_visual_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root, "ready_run", task_id="geometry_002_positions", pass_type="clean_pass")
            ready = root / "ready_run"
            (ready / "final_scene.blend").write_bytes(b"blend")
            (ready / "viewport.png").write_bytes(b"png")
            _write_run(root, "missing_run", task_id="camera_003_composition_view", pass_type="failed_validation")

            rows = collect_visual_artifact_rows(root)
            by_id = {row["run_id"]: row for row in rows}
            self.assertTrue(by_id["ready_run"]["visual_ready"])
            self.assertFalse(by_id["missing_run"]["visual_ready"])
            self.assertEqual(by_id["missing_run"]["missing_reason"], "missing final_scene.blend")

            csv_path, json_path = write_visual_artifacts_check(root)
            self.assertTrue(csv_path.is_file())
            self.assertTrue(json_path.is_file())


if __name__ == "__main__":
    unittest.main()
