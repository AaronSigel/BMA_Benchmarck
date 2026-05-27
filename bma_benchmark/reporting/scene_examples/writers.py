from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from bma_benchmark.reporting.scene_examples.card_renderer import (
    OptionalImageDependencyError,
    _load_pillow,
    render_scene_card,
)
from bma_benchmark.reporting.scene_examples.contact_sheet import build_contact_sheet
from bma_benchmark.reporting.scene_examples.models import SceneExample, SceneExampleBundle

FIELDS = [
    "run_id",
    "task_id",
    "category",
    "model",
    "strategy",
    "mcp_profile",
    "pass_type",
    "scene_score",
    "strict_success",
    "run_dir",
    "snapshot_path",
    "validation_result_path",
    "render_path",
    "viewport_path",
    "blend_path",
    "glb_path",
    "top_issues",
    "selection_reason",
    "render_missing_reason",
    "card_path",
    "thumbnail_path",
]


def write_scene_examples(bundle: SceneExampleBundle, out_dir: Path, *, render_images: bool = True) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        write_json(bundle, out_dir / "scene_examples.json"),
        write_csv(bundle, out_dir / "scene_examples.csv"),
        write_markdown(bundle, out_dir / "scene_examples.md"),
    ]
    if render_images:
        try:
            paths.extend(render_scene_images(bundle.examples, out_dir))
            paths[0] = write_json(bundle, out_dir / "scene_examples.json")
            paths[1] = write_csv(bundle, out_dir / "scene_examples.csv")
            paths[2] = write_markdown(bundle, out_dir / "scene_examples.md")
        except OptionalImageDependencyError as exc:
            bundle.warnings.append(str(exc))
            write_json(bundle, out_dir / "scene_examples.json")
            write_markdown(bundle, out_dir / "scene_examples.md")
    return paths


def render_scene_images(examples: list[SceneExample], out_dir: Path) -> list[Path]:
    Image, _, _ = _load_pillow()

    paths: list[Path] = []
    cards = out_dir / "cards"
    thumbs = out_dir / "thumbnails"
    cards.mkdir(parents=True, exist_ok=True)
    thumbs.mkdir(parents=True, exist_ok=True)
    for example in examples:
        card_path = cards / f"{_safe(example.run_id)}_card.png"
        render_scene_card(example, card_path)
        thumb_path = thumbs / f"{_safe(example.run_id)}.png"
        img = Image.open(card_path).convert("RGB")
        img.thumbnail((320, 220))
        img.save(thumb_path)
        example.card_path = card_path
        example.thumbnail_path = thumb_path
        paths.extend([card_path, thumb_path])
    groups = {
        "clean_pass_examples.png": [e for e in examples if e.pass_type == "clean_pass"],
        "soft_pass_examples.png": [e for e in examples if e.pass_type == "soft_pass"],
        "failed_validation_examples.png": [e for e in examples if e.pass_type == "failed_validation"],
        "mixed_scene_examples.png": examples,
    }
    for name, group in groups.items():
        sheet_path = build_contact_sheet(group, out_dir / name, name.replace("_", " ").replace(".png", ""))
        if sheet_path is not None:
            paths.append(sheet_path)
    return paths


def write_json(bundle: SceneExampleBundle, path: Path) -> Path:
    path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_csv(bundle: SceneExampleBundle, path: Path) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for example in bundle.examples:
            data = example.model_dump(mode="json")
            data["top_issues"] = ";".join(data.get("top_issues") or [])
            writer.writerow({field: data.get(field) for field in FIELDS})
    return path


def write_markdown(bundle: SceneExampleBundle, path: Path) -> Path:
    lines = [
        "# Scene Examples",
        "",
        "| Run | Task | Pass Type | Score | Model | Image | Issues |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for example in bundle.examples:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    example.run_id,
                    example.task_id,
                    example.pass_type,
                    example.scene_score,
                    example.model,
                    example.render_path or example.viewport_path or example.render_missing_reason,
                    ", ".join(example.top_issues[:3]) or "none",
                )
            )
            + " |"
        )
    if bundle.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in bundle.warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")
