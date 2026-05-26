from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from benchmark.experiments.matrix import load_matrix

from bma_benchmark.reporting.evidence_pack.completeness import write_completeness_check
from bma_benchmark.reporting.evidence_pack.figures import render_evidence_figures
from bma_benchmark.reporting.evidence_pack.manifest import write_run_manifest
from bma_benchmark.reporting.evidence_pack.readme import write_readme
from bma_benchmark.reporting.evidence_pack.sanity import run_validator_sanity_suite
from bma_benchmark.reporting.evidence_pack.selection import select_evidence_examples
from bma_benchmark.reporting.evidence_pack.tables import write_evidence_tables
from bma_benchmark.reporting.scene_examples.discovery import discover_runs
from bma_benchmark.reporting.scene_examples.models import SceneExampleSelectionConfig
from bma_benchmark.reporting.scene_examples.writers import render_scene_images


def build_evidence_pack(
    experiment_dir: Path,
    out_dir: Path,
    *,
    config_path: Path | None = None,
    tasks_root: Path = Path("tasks"),
) -> dict:
    """Собирает artifacts/report_evidence_pack из результатов демо-среза."""
    experiment_dir = Path(experiment_dir)
    out_dir = Path(out_dir)
    matrix = load_matrix(config_path) if config_path else None

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "demo_slice").mkdir(exist_ok=True)
    (out_dir / "selected_examples").mkdir(exist_ok=True)

    sanity_result = run_validator_sanity_suite(out_dir / "validator_sanity", tasks_root=tasks_root)
    runs = discover_runs(experiment_dir)
    selection_config = SceneExampleSelectionConfig(examples_per_status=6)
    bundle = select_evidence_examples(runs, selection_config)

    cards_dir = out_dir / "selected_examples" / "cards"
    render_scene_images(bundle.examples, cards_dir)

    table_paths = write_evidence_tables(
        experiment_dir,
        out_dir,
        runs,
        bundle.examples,
        matrix,
        sanity_result,
    )
    figure_paths = render_evidence_figures(bundle.examples, out_dir / "figures")
    _copy_selected_logs(bundle.examples, out_dir / "logs")
    _link_experiment(experiment_dir, out_dir / "demo_slice")

    manifest = write_run_manifest(
        out_dir / "run_manifest.json",
        experiment_dir=experiment_dir,
        config_path=config_path,
        matrix=matrix,
    )
    write_readme(
        out_dir / "README.md",
        manifest=manifest,
        experiment_dir=experiment_dir,
        out_dir=out_dir,
        bundle=bundle,
        sanity_result=sanity_result,
        table_paths=table_paths,
    )
    completeness = write_completeness_check(
        out_dir,
        runs=runs,
        examples=bundle.examples,
        sanity_result=sanity_result,
        figure_paths=figure_paths,
        table_paths=table_paths,
        expected_runs=matrix.metadata.get("expected_runs", 100) if matrix else 100,
    )
    return completeness


def _copy_selected_logs(examples, logs_dir: Path) -> None:
    for sub in ("selected_validation_results", "selected_scene_snapshots", "selected_run_results"):
        (logs_dir / sub).mkdir(parents=True, exist_ok=True)
    for idx, example in enumerate(examples, start=1):
        prefix = f"{idx:02d}_{example.run_id}"
        if example.validation_result_path and example.validation_result_path.is_file():
            shutil.copy2(
                example.validation_result_path,
                logs_dir / "selected_validation_results" / f"{prefix}.json",
            )
        if example.snapshot_path and example.snapshot_path.is_file():
            shutil.copy2(
                example.snapshot_path,
                logs_dir / "selected_scene_snapshots" / f"{prefix}.json",
            )
        run_result = example.run_dir / "run_result.json"
        if run_result.is_file():
            shutil.copy2(run_result, logs_dir / "selected_run_results" / f"{prefix}.json")


def _link_experiment(experiment_dir: Path, demo_slice_dir: Path) -> None:
    target = demo_slice_dir / "experiment"
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(experiment_dir.resolve(), target_is_directory=True)
    except OSError:
        (demo_slice_dir / "experiment_path.txt").write_text(str(experiment_dir.resolve()), encoding="utf-8")
