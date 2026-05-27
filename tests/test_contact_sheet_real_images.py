from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bma_benchmark.reporting.scene_examples.contact_sheet import build_contact_sheet
from bma_benchmark.reporting.scene_examples.models import SceneExample


def _write_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color=(120, 180, 220)).save(path)


class ContactSheetRealImagesTests(unittest.TestCase):
    def test_contact_sheet_builds_from_render_not_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            render = root / "render.png"
            card = root / "card.png"
            _write_png(render)
            _write_png(card)
            example = SceneExample(
                run_id="run1",
                task_id="geometry_002_positions",
                pass_type="clean_pass",
                model="google/gemini-2.5-flash-lite",
                strategy="react",
                scene_score=0.98,
                run_dir=root,
                render_path=render,
                card_path=card,
                selection_reason="test",
            )
            out = root / "sheet.png"
            result = build_contact_sheet([example], out, "Clean pass examples")
            self.assertEqual(result, out)
            self.assertTrue(out.is_file())
            from PIL import Image

            sheet = Image.open(out).convert("RGB")
            render_img = Image.open(render).convert("RGB")
            self.assertEqual(sheet.size[0], 1800)
            self.assertNotEqual(render_img.getpixel((0, 0)), (242, 244, 247))


if __name__ == "__main__":
    unittest.main()
