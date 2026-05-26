from __future__ import annotations

from pathlib import Path

from bma_benchmark.reporting.scene_examples.card_renderer import _load_pillow
from bma_benchmark.reporting.scene_examples.models import SceneExample


def build_contact_sheet(examples: list[SceneExample], out_path: Path, title: str) -> Path:
    Image, ImageDraw, ImageFont = _load_pillow()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = 2
    thumb_w, thumb_h = 420, 300
    rows = max(1, (len(examples) + cols - 1) // cols)
    width = cols * thumb_w + 48
    height = rows * (thumb_h + 70) + 80
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((24, 24), title, fill=(20, 24, 32), font=font)
    for idx, example in enumerate(examples):
        col = idx % cols
        row = idx // cols
        x = 24 + col * thumb_w
        y = 60 + row * (thumb_h + 70)
        source = example.card_path or example.render_path or example.viewport_path
        if source and Path(source).is_file():
            img = Image.open(source).convert("RGB")
            img.thumbnail((thumb_w - 20, thumb_h))
            sheet.paste(img, (x, y))
        else:
            draw.rectangle((x, y, x + thumb_w - 20, y + thumb_h), fill=(242, 244, 247), outline=(180, 186, 196))
            draw.text((x + 120, y + 135), "image unavailable", fill=(80, 86, 96), font=font)
        label = f"{example.task_id} | {example.pass_type} | {_score(example.scene_score)}"
        draw.text((x, y + thumb_h + 8), label[:70], fill=(20, 24, 32), font=font)
    sheet.save(out_path)
    return out_path


def _score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"
