from __future__ import annotations

from pathlib import Path

from bma_benchmark.reporting.evidence_pack.figure_renderers import (
    render_soft_pass_export_figure,
    render_validator_expected_actual_figure,
)
from bma_benchmark.reporting.scene_examples.contact_sheet import _scene_image, build_contact_sheet
from bma_benchmark.reporting.scene_examples.models import SceneExample


def render_evidence_figures(
    examples: list[SceneExample],
    figures_dir: Path,
    *,
    validator_pool: list[SceneExample] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    clean = [e for e in examples if e.pass_type == "clean_pass"]
    failed = [e for e in examples if e.pass_type == "failed_validation"]
    soft = [e for e in examples if e.pass_type == "soft_pass"]

    if clean:
        path = build_contact_sheet(
            clean[:6],
            figures_dir / "clean_pass_examples.png",
            "Clean pass examples",
            warnings=warnings,
        )
        if path is not None:
            paths["clean_pass_examples.png"] = path

    if failed:
        path = build_contact_sheet(
            failed[:6],
            figures_dir / "failed_validation_examples.png",
            "Failed validation examples",
            warnings=warnings,
        )
        if path is not None:
            paths["failed_validation_examples.png"] = path

    if soft:
        soft_with_images = [e for e in soft[:2] if _scene_image(e) is not None]
        if soft_with_images:
            paths["soft_pass_export_example.png"] = render_soft_pass_export_figure(
                soft_with_images,
                figures_dir / "soft_pass_export_example.png",
            )
        else:
            _write_figure_not_available(
                figures_dir / "soft_pass_export_example.png",
                "Soft pass export example",
                soft[:2],
                warnings=warnings,
            )

    validator_example = _best_validator_example(validator_pool or examples)
    if validator_example and _scene_image(validator_example) is not None:
        paths["validator_expected_actual_example.png"] = render_validator_expected_actual_figure(
            validator_example,
            figures_dir / "validator_expected_actual_example.png",
        )
    elif validator_example:
        _write_figure_not_available(
            figures_dir / "validator_expected_actual_example.png",
            "Validator expected vs actual example",
            [validator_example],
            warnings=warnings,
        )

    return paths


def _write_figure_not_available(
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
        "Figure was not generated because no render/viewport image was available.",
        "",
    ]
    for example in examples:
        reason = example.render_missing_reason or "no render/viewport image found"
        lines.append(f"- `{example.run_id}` ({example.task_id}): {reason}")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if warnings is not None:
        warnings.append(f"{out_path.name} was not generated because selected examples miss render/viewport images")


def _best_validator_example(examples: list[SceneExample]) -> SceneExample | None:
    """Пример с частичным прохождением: есть и pass, и fail (не «всё провалено»)."""
    best: SceneExample | None = None
    best_score = -1
    for ex in examples:
        validation = _load_validation(ex)
        passed_rows, failed_rows, _skip_rows = _check_counts(validation, ex)
        if passed_rows == 0 or failed_rows == 0:
            continue
        total_score = validation.get("total_score")
        balance = min(passed_rows, failed_rows)
        score = balance * 20 + passed_rows + failed_rows
        if ex.pass_type == "soft_pass":
            score += 200
        if total_score is not None and 0.6 <= float(total_score) < 1.0:
            score += 150
        if validation.get("overall_status") == "warning":
            score += 100
        if failed_rows == 1:
            score += 80
        if score > best_score:
            best_score = score
            best = ex

    if best is not None:
        return best

    for ex in examples:
        validation = _load_validation(ex)
        passed_rows, failed_rows, _skip_rows = _check_counts(validation, ex)
        if passed_rows > 0 and failed_rows > 0:
            return ex
    return examples[0] if examples else None


def _check_counts(example_validation: dict, ex: SceneExample) -> tuple[int, int, int]:
    checks = example_validation.get("check_table") or ex.check_table_excerpt or []
    if not checks:
        checks = [
            row
            for validator in example_validation.get("validators") or []
            if isinstance(validator, dict)
            for row in validator.get("check_table") or []
        ]
    passed = 0
    failed = 0
    skipped = 0
    for row in checks:
        if not isinstance(row, dict):
            continue
        status = row.get("status")
        if status == "skip" or str(status) == "skip":
            skipped += 1
        elif row.get("passed"):
            passed += 1
        else:
            failed += 1
    return passed, failed, skipped


def _load_validation(example: SceneExample) -> dict:
    if example.validation_result_path and example.validation_result_path.is_file():
        import json

        try:
            return json.loads(example.validation_result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}
