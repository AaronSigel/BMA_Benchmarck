from __future__ import annotations

from pathlib import Path

from bma_benchmark.reporting.scene_examples.card_renderer import _load_pillow
from bma_benchmark.reporting.scene_examples.models import SceneExample


def render_soft_pass_export_figure(examples: list[SceneExample], out_path: Path) -> Path:
    Image, ImageDraw, ImageFont = _load_pillow()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, card_h = 920, 680
    height = max(card_h, len(examples) * card_h + 40)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    y = 20
    for example in examples:
        source = example.render_path or example.viewport_path
        box = (20, y, 440, y + 360)
        if source and Path(source).is_file():
            img = Image.open(source).convert("RGB")
            img.thumbnail((box[2] - box[0], box[3] - box[1]))
            canvas.paste(img, (box[0], box[1] + (box[3] - box[1] - img.height) // 2))
        else:
            draw.rectangle(box, fill=(242, 244, 247), outline=(180, 186, 196))

        glb_exists = "yes" if example.glb_path and example.glb_path.is_file() else "no"
        glb_import = _glb_import_status(example)
        diagnostic = _diagnostic_reason(example)
        lines = [
            f"task: {example.task_id}",
            f"model: {example.model}  strategy: {example.strategy}",
            f"pass_type: soft_pass  scene_score: {_fmt(example.scene_score)}",
            f"diagnostic reason: {diagnostic}",
            f"export file exists: {glb_exists}",
            f"glb import back: {glb_import}",
        ]
        ty = y + 20
        for line in lines:
            draw.text((460, ty), _clip(line, 55), fill=(20, 24, 32), font=font)
            ty += 28
        y += card_h

    canvas.save(out_path)
    return out_path


def render_validator_expected_actual_figure(example: SceneExample, out_path: Path) -> Path:
    Image, ImageDraw, ImageFont = _load_pillow()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1400, 720
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    source = example.render_path or example.viewport_path
    left_box = (20, 20, 660, 680)
    if source and Path(source).is_file():
        img = Image.open(source).convert("RGB")
        img.thumbnail((left_box[2] - left_box[0], left_box[3] - left_box[1]))
        x = left_box[0] + (left_box[2] - left_box[0] - img.width) // 2
        y = left_box[1] + (left_box[3] - left_box[1] - img.height) // 2
        canvas.paste(img, (x, y))
    else:
        draw.rectangle(left_box, fill=(242, 244, 247), outline=(180, 186, 196))
        draw.text((280, 340), "image unavailable", fill=(80, 86, 96), font=font)

    draw.text((690, 20), f"{example.task_id} | validator checks", fill=(20, 24, 32), font=font)
    headers = ["check", "expected", "actual", "status"]
    col_x = [690, 900, 1050, 1200]
    ty = 50
    for idx, header in enumerate(headers):
        draw.text((col_x[idx], ty), header, fill=(60, 66, 76), font=font)
    ty += 24
    draw.line((690, ty, 1360, ty), fill=(180, 186, 196))
    ty += 8

    rows = _check_rows(example)[:12]
    for row in rows:
        check_label = _check_label(row)
        expected = _cell(row.get("expected"))
        actual = _cell(row.get("actual"))
        status = "pass" if row.get("passed") else "fail"
        draw.text((col_x[0], ty), _clip(check_label, 28), fill=(20, 24, 32), font=font)
        draw.text((col_x[1], ty), _clip(expected, 18), fill=(20, 24, 32), font=font)
        draw.text((col_x[2], ty), _clip(actual, 18), fill=(20, 24, 32), font=font)
        color = (20, 120, 60) if status == "pass" else (180, 40, 40)
        draw.text((col_x[3], ty), status, fill=color, font=font)
        ty += 22

    canvas.save(out_path)
    return out_path


def _check_rows(example: SceneExample) -> list[dict]:
    import json

    if example.validation_result_path and example.validation_result_path.is_file():
        try:
            data = json.loads(example.validation_result_path.read_text(encoding="utf-8"))
            checks = data.get("check_table") or []
            if checks:
                return [row for row in checks if isinstance(row, dict)]
        except (OSError, json.JSONDecodeError):
            pass
    return list(example.check_table_excerpt or [])


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
        if isinstance(validator, dict) and validator.get("name") == "glb_import_back":
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
    if value is None:
        return ""
    return str(value)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
