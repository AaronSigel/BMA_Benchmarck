from pathlib import Path

import pytest

from bma_benchmark.reporting.scene_examples.card_renderer import OptionalImageDependencyError, render_scene_card
from bma_benchmark.reporting.scene_examples.models import SceneExample


pytest.importorskip("PIL")


def _example(tmp_path: Path, image: Path | None) -> SceneExample:
    return SceneExample(
        run_id="run_clean_001",
        task_id="geometry_002_positions",
        pass_type="clean_pass",
        scene_score=1.0,
        model="model",
        strategy="strategy",
        mcp_profile="minimal",
        run_dir=tmp_path,
        render_path=image,
        top_issues=[],
        selection_reason="test",
    )


def test_scene_card_created_with_image(tmp_path: Path) -> None:
    from PIL import Image

    image = tmp_path / "render.png"
    Image.new("RGB", (16, 16), "red").save(image)

    out = render_scene_card(_example(tmp_path, image), tmp_path / "card.png")

    assert out.is_file()
    assert out.stat().st_size > 0


def test_scene_card_created_with_placeholder(tmp_path: Path) -> None:
    out = render_scene_card(_example(tmp_path, None), tmp_path / "card.png")

    assert out.is_file()
    assert out.stat().st_size > 0
