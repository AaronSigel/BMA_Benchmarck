from __future__ import annotations

import json
import unittest
from pathlib import Path

from bma_benchmark.reporting.scene_examples.image_finder import (
    _is_excluded_png,
    find_scene_image,
    find_scene_images,
)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x01\x01\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


class ImageFinderManifestTests(unittest.TestCase):
    def test_viewport_from_manifest(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run1"
            run.mkdir()
            _write_png(run / "viewport.png")
            (run / "artifact_manifest.json").write_text(
                json.dumps({"artifacts": {"viewport": {"path": "viewport.png", "exists": True}}}),
                encoding="utf-8",
            )
            _, viewport_path, reason = find_scene_images(run)
            self.assertEqual(viewport_path, run / "viewport.png")
            self.assertIsNone(reason)

    def test_final_render_from_manifest(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run1"
            run.mkdir()
            _write_png(run / "final_render.png")
            (run / "artifact_manifest.json").write_text(
                json.dumps({"artifacts": {"final_render": {"path": "final_render.png", "exists": True}}}),
                encoding="utf-8",
            )
            render_path, _, reason = find_scene_images(run)
            self.assertEqual(render_path, run / "final_render.png")
            self.assertIsNone(reason)

    def test_ignores_card_and_examples_png(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run1"
            run.mkdir()
            _write_png(run / "foo_card.png")
            _write_png(run / "clean_pass_examples.png")
            image, reason = find_scene_image(run)
            self.assertIsNone(image)
            self.assertTrue(reason)
            self.assertTrue(_is_excluded_png("foo_card.png"))
            self.assertTrue(_is_excluded_png("clean_pass_examples.png"))


if __name__ == "__main__":
    unittest.main()
