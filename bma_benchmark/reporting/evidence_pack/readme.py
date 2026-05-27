from __future__ import annotations

from pathlib import Path
from typing import Any

from bma_benchmark.reporting.evidence_pack.sanity import SanitySuiteResult
from bma_benchmark.reporting.scene_examples.models import SceneExampleBundle


def write_readme(
    path: Path,
    *,
    manifest: dict[str, Any],
    experiment_dir: Path,
    out_dir: Path,
    bundle: SceneExampleBundle,
    sanity_result: SanitySuiteResult,
    table_paths: dict[str, Path],
    completeness: dict[str, Any] | None = None,
) -> None:
    lines = [
        "# Report Evidence Pack",
        "",
        "Демонстрационный пакет данных для вставки в отчёт BMA-Benchmark.",
        "",
        "> **Важно:** этот демонстрационный срез (100 runs) **не заменяет** основную экспериментальную матрицу на 3600 прогонов.",
        "",
        "## Provenance",
        "",
        f"- Git commit: `{manifest.get('git_commit', 'unknown')}`",
        f"- Created at: {manifest.get('created_at', 'unknown')}",
        f"- Matrix config: `{manifest.get('matrix_config', 'unknown')}`",
        f"- Matrix ID: `{manifest.get('matrix_id', 'unknown')}`",
        "",
        "## Matrix composition",
        "",
        f"- Tasks: {', '.join(manifest.get('tasks') or [])}",
        f"- Models: {', '.join(manifest.get('models') or [])}",
        "- Strategies: direct_tool_calling, plan_and_execute, react, plan_execute_react_repair",
        f"- MCP profile: {', '.join(manifest.get('mcp_profiles') or ['full'])}",
        f"- Repetitions: {manifest.get('repetitions', 1)}",
        "",
        "## Commands",
        "",
        "```bash",
        "python -m bma_benchmark preflight --config configs/matrices/report_demo_slice.yaml",
        "python -m bma_benchmark run-matrix --config configs/matrices/report_demo_slice.yaml --analyze --report",
        "python -m bma_benchmark build-evidence-pack \\",
        "  --experiment artifacts/experiments/report_demo_slice \\",
        "  --config configs/matrices/report_demo_slice.yaml \\",
        "  --out artifacts/report_evidence_pack \\",
        "  --render-missing-with-blender --render-mode viewport",
        "```",
        "",
        "## Visual evidence",
        "",
    ]
    if completeness:
        complete = completeness.get("visual_evidence_complete")
        lines.append(f"- Status: `{completeness.get('visual_evidence_status', 'unknown')}`")
        lines.append(f"- Complete: `{complete}`")
        lines.append(
            f"- Selected examples with images: {completeness.get('selected_examples_with_images', 0)}"
            f"/{completeness.get('selected_examples_total', 0)}"
        )
        if not complete:
            lines.extend([
                "",
                "Visual evidence images are incomplete: no render/viewport images were available "
                "for all required figure groups.",
            ])
            for warning in completeness.get("warnings") or []:
                lines.append(f"- Warning: {warning}")
        lines.append("")
    else:
        lines.append("")

    lines.extend([
        "## Paths",
        "",
        f"- Raw experiment: `{experiment_dir}`",
        f"- Evidence pack: `{out_dir}`",
        f"- Selected examples: `{out_dir / 'selected_examples'}`",
        f"- Validator sanity: `{out_dir / 'validator_sanity'}`",
        "",
        "## Selected examples",
        "",
        f"- Total selected: {len(bundle.examples)}",
        f"- Clean pass: {sum(1 for e in bundle.examples if e.pass_type == 'clean_pass')}",
        f"- Failed validation: {sum(1 for e in bundle.examples if e.pass_type == 'failed_validation')}",
        f"- Soft pass: {sum(1 for e in bundle.examples if e.pass_type == 'soft_pass')}",
        "",
    ])
    if bundle.warnings:
        lines.extend(["## Selection warnings", ""])
        lines.extend(f"- {w}" for w in bundle.warnings)
        lines.append("")

    lines.extend([
        "## Validator sanity",
        "",
        f"- Cases: {len(sanity_result.cases)}",
        f"- All passed as expected: {sanity_result.all_passed_as_expected}",
        "",
        "## Tables",
        "",
    ])
    for name, p in sorted(table_paths.items()):
        lines.append(f"- `{p.relative_to(out_dir)}`")

    missing = out_dir / "tables" / "missing_artifacts.csv"
    if missing.is_file():
        lines.extend([
            "",
            "## Missing artifacts",
            "",
            f"See `{missing.relative_to(out_dir)}` for runs without render/viewport or validation files.",
        ])

    if not sanity_result.all_passed_as_expected:
        lines.extend([
            "",
            "## Sanity failures",
            "",
            f"See `{out_dir / 'validator_sanity' / 'FAILED_SANITY_CASES.md'}`.",
        ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
