from __future__ import annotations

from pathlib import Path

from bma_benchmark.reporting.scene_examples.card_renderer import _load_pillow
from bma_benchmark.reporting.scene_examples.contact_sheet import (
    CARD_HEIGHT,
    CARD_WIDTH,
    IMAGE_HEIGHT,
    MIN_FONT_SIZE,
    SHEET_WIDTH,
    _clip,
    _fit_image,
    _load_fonts,
    _scene_image,
    short_model_name,
)
from benchmark.validation.check_labels import display_check_id, display_object_ref
from bma_benchmark.reporting.scene_examples.models import SceneExample


def render_soft_pass_export_figure(
    examples: list[SceneExample],
    out_path: Path,
    *,
    warnings: list[str] | None = None,
) -> Path:
    Image, ImageDraw, ImageFont = _load_pillow()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    font, title_font, _ = _load_fonts(ImageFont, warnings)

    gap = 40
    header = 72
    meta_x = gap + CARD_WIDTH + gap
    height = header + len(examples) * (CARD_HEIGHT + gap) + gap

    canvas = Image.new("RGB", (SHEET_WIDTH, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 24), "Soft pass export examples", fill=(20, 24, 32), font=title_font)

    for idx, example in enumerate(examples):
        y = header + idx * (CARD_HEIGHT + gap)
        x = gap
        image_box = (x, y, x + CARD_WIDTH, y + IMAGE_HEIGHT)
        draw.rectangle(image_box, fill=(250, 250, 252), outline=(210, 214, 222))

        source = _scene_image(example)
        if source is not None:
            img = Image.open(source).convert("RGB")
            img = _fit_image(img, CARD_WIDTH, IMAGE_HEIGHT)
            paste_x = x + (CARD_WIDTH - img.width) // 2
            paste_y = y + (IMAGE_HEIGHT - img.height) // 2
            canvas.paste(img, (paste_x, paste_y))

        glb_exists = "yes" if example.glb_path and example.glb_path.is_file() else "no"
        glb_import = _glb_import_status(example)
        diagnostic = _diagnostic_reason(example)
        lines = [
            example.task_id,
            f"model: {short_model_name(example.model)}",
            f"strategy: {example.strategy or 'N/A'}",
            f"pass_type: soft_pass  score: {_fmt(example.scene_score)}",
            f"diagnostic: {diagnostic}",
            f"export exists: {glb_exists}",
            f"glb import back: {glb_import}",
        ]
        ty = y + 12
        for line in lines:
            draw.text((meta_x, ty), _clip(line, 72), fill=(20, 24, 32), font=font)
            ty += MIN_FONT_SIZE + 8

    canvas.save(out_path)
    return out_path


def render_validator_expected_actual_figure(
    example: SceneExample,
    out_path: Path,
    *,
    warnings: list[str] | None = None,
) -> Path:
    Image, ImageDraw, ImageFont = _load_pillow()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    font, title_font, _ = _load_fonts(ImageFont, warnings)

    gap = 40
    header = 72
    image_band_h = 500
    table_x = gap
    table_w = SHEET_WIDTH - 2 * gap
    col_widths = [220, 140, 200, 200, 70, 120]
    col_x = _column_starts(table_x, col_widths, padding=8)

    rows = _prioritize_failed_rows(_check_rows(example), limit=18)
    row_step = MIN_FONT_SIZE + 12
    table_header_h = MIN_FONT_SIZE + 28
    table_body_h = max(row_step, len(rows) * row_step)
    height = header + image_band_h + gap + table_header_h + table_body_h + gap

    canvas = Image.new("RGB", (SHEET_WIDTH, height), "white")
    draw = ImageDraw.Draw(canvas)
    render_source = _render_source(example)
    pass_label = example.pass_type or "unknown"
    score_label = _fmt(example.scene_score)
    draw.text(
        (gap, 24),
        (
            f"{example.task_id} | validator checks | "
            f"pass_type: {pass_label} | score: {score_label} | render_source: {render_source}"
        ),
        fill=(20, 24, 32),
        font=title_font,
    )

    y0 = header
    image_box = (gap, y0, gap + table_w, y0 + image_band_h)
    draw.rectangle(image_box, fill=(250, 250, 252), outline=(210, 214, 222))
    source = _scene_image(example)
    if source is not None:
        img = Image.open(source).convert("RGB")
        img = _fit_image(img, table_w, image_band_h)
        paste_x = gap + (table_w - img.width) // 2
        paste_y = y0 + (image_band_h - img.height) // 2
        canvas.paste(img, (paste_x, paste_y))

    ty = y0 + image_band_h + gap
    headers = ["check", "object", "expected", "actual", "status", "issue"]
    for idx, header_label in enumerate(headers):
        draw.text((col_x[idx], ty), header_label, fill=(60, 66, 76), font=font)
    ty += MIN_FONT_SIZE + 10
    draw.line((table_x, ty, table_x + table_w, ty), fill=(180, 186, 196))
    ty += 12

    for row in rows:
        check_label = display_check_id(row)
        object_label = display_object_ref(row)
        expected = _format_cell(row.get("expected"))
        actual = _format_cell(row.get("actual"))
        status = _row_status(row)
        issue = str(row.get("issue_code") or "none")
        values = [check_label, object_label, expected, actual, status, issue]
        for idx, value in enumerate(values):
            fill = _status_color(status) if idx == 4 else (20, 24, 32)
            draw.text(
                (col_x[idx], ty),
                _clip_to_width(draw, value, font, col_widths[idx]),
                fill=fill,
                font=font,
            )
        ty += row_step

    canvas.save(out_path)
    return out_path


def _column_starts(origin: int, widths: list[int], *, padding: int) -> list[int]:
    starts = [origin]
    for width in widths[:-1]:
        starts.append(starts[-1] + width + padding)
    return starts


def _clip_to_width(draw, text: str, font, max_width: int) -> str:
    value = str(text)
    if _text_width(draw, value, font) <= max_width:
        return value
    ell = "..."
    while value and _text_width(draw, value + ell, font) > max_width:
        value = value[:-1]
    return value + ell if value else ell


def _text_width(draw, text: str, font) -> float:
    try:
        return float(draw.textlength(text, font=font))
    except AttributeError:
        bbox = draw.textbbox((0, 0), text, font=font)
        return float(bbox[2] - bbox[0])


def _format_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = [f"{key}={value[key]}" for key in sorted(value.keys())]
        return ", ".join(parts)
    return str(value)


def _is_export_row(row: dict) -> bool:
    validator = str(row.get("validator_name") or "")
    check_id = display_check_id(row)
    return validator in {"export_validator", "glb_import_back_validator"} or check_id.startswith(
        ("export.", "import_back.")
    )


def _row_status(row: dict) -> str:
    status = row.get("status")
    if status:
        return str(status)
    return "pass" if row.get("passed") else "fail"


def _status_color(status: str) -> tuple[int, int, int]:
    if status == "pass":
        return (20, 120, 60)
    if status == "skip":
        return (140, 140, 140)
    return (180, 40, 40)


def _prioritize_failed_rows(rows: list[dict], *, limit: int) -> list[dict]:
    export_rows = [row for row in rows if isinstance(row, dict) and _is_export_row(row)]
    failed = [
        row for row in rows
        if isinstance(row, dict) and _row_status(row) == "fail"
    ]
    skipped = [
        row for row in rows
        if isinstance(row, dict) and _row_status(row) == "skip"
    ]
    passed = [
        row for row in rows
        if isinstance(row, dict) and _row_status(row) == "pass"
    ]
    ordered: list[dict] = []
    seen: set[int] = set()
    max_passed_visible = max(4, limit // 3)
    buckets = (export_rows, failed, skipped, passed[:max_passed_visible])
    for bucket in buckets:
        for row in bucket:
            key = id(row)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(row)
    return ordered[:limit]


def _check_rows(example: SceneExample) -> list[dict]:
    import json

    if example.validation_result_path and example.validation_result_path.is_file():
        try:
            data = json.loads(example.validation_result_path.read_text(encoding="utf-8"))
            checks = data.get("check_table") or []
            if not checks:
                checks = [
                    row
                    for validator in data.get("validators") or []
                    if isinstance(validator, dict)
                    for row in validator.get("check_table") or []
                ]
            if checks:
                return [row for row in checks if isinstance(row, dict)]
        except (OSError, json.JSONDecodeError):
            pass
    return list(example.check_table_excerpt or [])


def _render_source(example: SceneExample) -> str:
    import json

    if example.validation_result_path and example.validation_result_path.is_file():
        try:
            data = json.loads(example.validation_result_path.read_text(encoding="utf-8"))
            summary = data.get("summary") or {}
            if isinstance(summary, dict) and summary.get("render_source"):
                return str(summary["render_source"])
        except (OSError, json.JSONDecodeError):
            pass
    if example.render_path and example.render_path.is_file():
        return "final_scene"
    if example.viewport_path and example.viewport_path.is_file():
        return "final_scene"
    return "unknown"


def _check_label(row: dict) -> str:
    parts = [str(row.get("check_name") or "")]
    field = row.get("field")
    if field:
        parts.append(str(field))
    return ".".join(p for p in parts if p)


def _glb_import_status(example: SceneExample) -> str:
    import json

    if not example.validation_result_path:
        return "unknown"
    try:
        data = json.loads(example.validation_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    for validator in data.get("validators") or []:
        if isinstance(validator, dict) and validator.get("name") == "glb_import_back_validator":
            status = validator.get("status")
            return "pass" if status == "passed" else "fail"
    return "n/a"


def _diagnostic_reason(example: SceneExample) -> str:
    run_result_path = example.run_dir / "run_result.json"
    if not run_result_path.is_file():
        return "agent diagnostic termination"
    import json

    try:
        data = json.loads(run_result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "agent diagnostic termination"
    summary = data.get("summary") or {}
    return str(
        summary.get("error_type")
        or summary.get("failure_stage")
        or data.get("status")
        or "agent diagnostic termination"
    )


def _fmt(value) -> str:
    return "N/A" if value is None else f"{float(value):.3f}"


def _cell(value) -> str:
    return _format_cell(value)
