from __future__ import annotations

from pathlib import Path

from bma_benchmark.reporting.scene_examples.card_renderer import _load_pillow
from bma_benchmark.reporting.scene_examples.models import SceneExample

SHEET_WIDTH = 1800
CARD_WIDTH = 820
CARD_HEIGHT = 520
IMAGE_HEIGHT = 390
TEXT_HEIGHT = 130
MIN_FONT_SIZE = 22


def build_contact_sheet(
    examples: list[SceneExample],
    out_path: Path,
    title: str,
    *,
    warnings: list[str] | None = None,
) -> Path | None:
    with_images = [example for example in examples if _scene_image(example) is not None]
    if not with_images:
        _write_not_available(out_path, title, examples, warnings=warnings)
        return None

    Image, ImageDraw, ImageFont = _load_pillow()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    font, title_font, font_warning = _load_fonts(ImageFont, warnings)

    cols = 2
    gap = 40
    header = 72
    rows = max(1, (len(with_images) + cols - 1) // cols)
    height = header + rows * (CARD_HEIGHT + gap) + gap
    sheet = Image.new("RGB", (SHEET_WIDTH, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 24), title, fill=(20, 24, 32), font=title_font)

    for idx, example in enumerate(with_images):
        col = idx % cols
        row = idx // cols
        x = gap + col * (CARD_WIDTH + gap)
        y = header + row * (CARD_HEIGHT + gap)
        _draw_card(sheet, Image, draw, example, x, y, font)

    sheet.save(out_path)
    return out_path


def _draw_card(sheet, Image, draw, example: SceneExample, x: int, y: int, font) -> None:
    image_box = (x, y, x + CARD_WIDTH, y + IMAGE_HEIGHT)
    source = _scene_image(example)
    draw.rectangle(image_box, fill=(250, 250, 252), outline=(210, 214, 222))
    if source is not None:
        img = Image.open(source).convert("RGB")
        img = _fit_image(img, CARD_WIDTH, IMAGE_HEIGHT)
        paste_x = x + (CARD_WIDTH - img.width) // 2
        paste_y = y + (IMAGE_HEIGHT - img.height) // 2
        sheet.paste(img, (paste_x, paste_y))

    text_y = y + IMAGE_HEIGHT + 8
    lines = [
        example.task_id,
        f"{example.pass_type} | score: {_score(example.scene_score)}",
        f"{short_model_name(example.model)} | {example.strategy or 'N/A'}",
        f"issues: {_issues(example)}",
    ]
    for line in lines:
        draw.text((x + 8, text_y), _clip(line, 72), fill=(20, 24, 32), font=font)
        text_y += MIN_FONT_SIZE + 6


def _scene_image(example: SceneExample) -> Path | None:
    for candidate in (example.render_path, example.viewport_path):
        if candidate and Path(candidate).is_file() and Path(candidate).stat().st_size > 0:
            return Path(candidate)
    return None


def _fit_image(img, max_w: int, max_h: int):
    fitted = img.copy()
    fitted.thumbnail((max_w, max_h))
    return fitted


def _load_fonts(ImageFont, warnings: list[str] | None):
    font_paths = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    ]
    for path in font_paths:
        if path.is_file():
            return (
                ImageFont.truetype(str(path), MIN_FONT_SIZE),
                ImageFont.truetype(str(path), MIN_FONT_SIZE + 4),
                None,
            )
    if warnings is not None:
        warnings.append("DejaVuSans not found; using default bitmap font for contact sheets")
    default = ImageFont.load_default()
    return default, default, "default font"


def short_model_name(model: str | None) -> str:
    if not model:
        return "N/A"
    if "/" in model:
        model = model.split("/", 1)[1]
    replacements = {
        "mistral-small-3.2-24b-instruct": "mistral-small-3.2",
    }
    return replacements.get(model, model)


def _issues(example: SceneExample) -> str:
    if example.top_issues:
        return ", ".join(example.top_issues[:2])
    return "none"


def _score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _write_not_available(
    out_path: Path,
    title: str,
    examples: list[SceneExample],
    *,
    warnings: list[str] | None = None,
) -> None:
    md_path = out_path.with_name(out_path.stem + "_not_available.md")
    lines = [
        f"# {title}",
        "",
        "Visual evidence was not generated because selected examples have no render/viewport images.",
        "",
        "## Selected runs",
        "",
    ]
    for example in examples:
        reason = example.render_missing_reason or "no render/viewport image found"
        lines.append(f"- `{example.run_id}` ({example.task_id}): {reason}")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if warnings is not None:
        warnings.append(f"{out_path.name} was not generated because all selected examples miss render/viewport images")
