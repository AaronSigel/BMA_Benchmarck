"""Тесты сборки article_exports_validator."""

from __future__ import annotations

import json
from pathlib import Path

from bma_benchmark.reporting.article_exports_validator.builder import build_validator_article_pack


def test_export_pack_from_real_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    main = root / "artifacts/experiments/final_multimodel_openrouter_v1_merged"
    evidence = root / "artifacts/report_evidence_pack"
    render = root / "artifacts/experiments/report_demo_slice"
    if not (main / "summary.csv").is_file():
        return
    if not (evidence / "figures" / "clean_pass_examples.png").is_file():
        return

    out = tmp_path / "article_exports_validator"
    result = build_validator_article_pack(
        main_experiment=main,
        evidence_pack=evidence,
        render_experiment=render,
        out_dir=out,
    )
    assert result["figures_copied"] == 4
    assert (out / "README.md").is_file()
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "3600" in readme
    assert "не содержит PNG" in readme or "Содержит PNG:** нет" in readme

    manifest = _read_csv(out / "tables" / "render_artifacts_manifest.csv")
    fig_rows = [r for r in manifest if r["filename"].startswith("fig_")]
    assert len(fig_rows) == 4
    assert all(r["is_from_main_3600_run"] == "false" for r in fig_rows)

    bundle = json.loads((out / "json" / "validator_article_bundle.json").read_text(encoding="utf-8"))
    assert bundle["source_separation"]["main_3600_run"]["contains_png_renders"] is False
    assert bundle["source_separation"]["render_matrix"]["contains_png_renders"] is True
    assert len(bundle["validator_issue_frequency"]) > 0
    assert len(bundle["validator_checks_examples"]) > 0

    assert (out / "case_studies" / "clean_pass_case.md").is_file()
    assert (out / "case_studies" / "failed_validation_case.md").is_file()
    assert (out / "case_studies" / "validator_checks_case.md").is_file()


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
