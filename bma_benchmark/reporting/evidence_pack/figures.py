from __future__ import annotations

from pathlib import Path

from bma_benchmark.reporting.evidence_pack.figure_renderers import (
    render_soft_pass_export_figure,
    render_validator_expected_actual_figure,
)
from bma_benchmark.reporting.scene_examples.card_renderer import render_scene_card
from bma_benchmark.reporting.scene_examples.contact_sheet import build_contact_sheet
from bma_benchmark.reporting.scene_examples.models import SceneExample


def render_evidence_figures(examples: list[SceneExample], figures_dir: Path) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    clean = [e for e in examples if e.pass_type == "clean_pass"]
    failed = [e for e in examples if e.pass_type == "failed_validation"]
    soft = [e for e in examples if e.pass_type == "soft_pass"]

    for group in (clean, failed, soft):
        for ex in group:
            if ex.card_path is None:
                card_path = figures_dir / f"{_safe(ex.run_id)}_card.png"
                render_scene_card(ex, card_path)
                ex.card_path = card_path

    if clean:
        paths["clean_pass_examples.png"] = build_contact_sheet(
            clean[:6], figures_dir / "clean_pass_examples.png", "Clean pass examples"
        )
    if failed:
        paths["failed_validation_examples.png"] = build_contact_sheet(
            failed[:6], figures_dir / "failed_validation_examples.png", "Failed validation examples"
        )
    if soft:
        paths["soft_pass_export_example.png"] = render_soft_pass_export_figure(
            soft[:2], figures_dir / "soft_pass_export_example.png"
        )

    validator_example = _best_validator_example(failed or examples)
    if validator_example:
        paths["validator_expected_actual_example.png"] = render_validator_expected_actual_figure(
            validator_example, figures_dir / "validator_expected_actual_example.png"
        )

    return paths


def _best_validator_example(examples: list[SceneExample]) -> SceneExample | None:
    best: SceneExample | None = None
    best_score = -1
    for ex in examples:
        validation = _load_validation(ex)
        checks = validation.get("check_table") or ex.check_table_excerpt or []
        failed_with_values = sum(
            1 for row in checks
            if isinstance(row, dict)
            and row.get("passed") is False
            and row.get("expected") is not None
            and row.get("actual") is not None
        )
        if failed_with_values > best_score:
            best_score = failed_with_values
            best = ex
    return best or (examples[0] if examples else None)


def _load_validation(example: SceneExample) -> dict:
    if example.validation_result_path and example.validation_result_path.is_file():
        import json

        try:
            return json.loads(example.validation_result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def _safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
