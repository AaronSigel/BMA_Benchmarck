from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bma_benchmark.reporting.evidence_pack.completeness import write_completeness_check
from bma_benchmark.reporting.evidence_pack.figures import render_evidence_figures
from bma_benchmark.reporting.evidence_pack.sanity import SanitySuiteResult
from bma_benchmark.reporting.scene_examples.models import SceneExample


class NoPlaceholderFiguresTests(unittest.TestCase):
    def test_no_placeholder_figures_when_images_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            example = SceneExample(
                run_id="run1",
                task_id="geometry_002_positions",
                pass_type="clean_pass",
                run_dir=root / "run1",
                selection_reason="test",
                render_missing_reason="no render/viewport image found",
            )
            figures_dir = root / "figures"
            warnings: list[str] = []
            paths = render_evidence_figures([example], figures_dir, warnings=warnings)
            self.assertNotIn("clean_pass_examples.png", paths)
            self.assertTrue((figures_dir / "clean_pass_examples_not_available.md").is_file())

            completeness = write_completeness_check(
                root,
                runs=[],
                examples=[example],
                sanity_result=SanitySuiteResult(cases=[], all_passed_as_expected=True),
                figure_paths=paths,
                table_paths={},
                visual_warnings=warnings,
            )
            self.assertFalse(completeness["visual_evidence_complete"])
            self.assertEqual(completeness["visual_evidence_status"], "incomplete")
            payload = json.loads((root / "completeness_check.json").read_text(encoding="utf-8"))
            self.assertFalse(payload["figures_with_real_scene_images"]["clean_pass_examples.png"])


if __name__ == "__main__":
    unittest.main()
