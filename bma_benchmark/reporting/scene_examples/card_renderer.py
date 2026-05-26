from __future__ import annotations

from pathlib import Path

from bma_benchmark.reporting.scene_examples.models import SceneExample


class OptionalImageDependencyError(RuntimeError):
    pass


def _load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise OptionalImageDependencyError(
            "Pillow is required to render scene cards. Install with: pip install pillow"
        ) from exc
    return Image, ImageDraw, ImageFont


def render_scene_card(example: SceneExample, out_path: Path) -> Path:
    Image, ImageDraw, ImageFont = _load_pillow()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 900, 620
    card = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    image_box = (24, 24, 876, 390)
    source = example.render_path or example.viewport_path
    if source and Path(source).is_file():
        img = Image.open(source).convert("RGB")
        img.thumbnail((image_box[2] - image_box[0], image_box[3] - image_box[1]))
        x = image_box[0] + ((image_box[2] - image_box[0]) - img.width) // 2
        y = image_box[1] + ((image_box[3] - image_box[1]) - img.height) // 2
        card.paste(img, (x, y))
    else:
        draw.rectangle(image_box, fill=(242, 244, 247), outline=(180, 186, 196))
        draw.text((340, 190), "image unavailable", fill=(80, 86, 96), font=title_font)

    y = 415
    lines = [
        f"{example.task_id}  [{example.pass_type}]",
        f"score: {_na(example.scene_score)}   model: {_na(example.model)}",
        f"strategy: {_na(example.strategy)}   mcp: {_na(example.mcp_profile)}",
    ]
    if example.top_issues:
        lines.append("issues: " + ", ".join(example.top_issues[:3]))
    else:
        lines.append("issues: none")
    for line in lines:
        draw.text((24, y), _clip(line, 130), fill=(20, 24, 32), font=font)
        y += 34
    card.save(out_path)
    return out_path


def _na(value) -> str:
    return "N/A" if value is None else str(value)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
