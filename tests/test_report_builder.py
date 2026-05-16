"""Tests for benchmark.analysis.report_builder."""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.analysis.models import (
    ExperimentAnalysisResult,
    ExperimentSummary,
    ReportConfig,
    RunAnalysisResult,
)
from benchmark.analysis.report_builder import (
    build_html_report,
    build_markdown_report,
    build_summary_table,
)

FIXTURES = Path(__file__).parent / "fixtures" / "analysis"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    run_id: str = "r1",
    task_id: str = "t1",
    agent_id: str = "a",
    strategy: str = "react",
    success: bool | None = True,
    total_score: float | None = 1.0,
    tool_call_count: int = 5,
    llm_call_count: int = 3,
    duration_sec: float | None = 10.0,
    error_count: int = 0,
    metrics: dict | None = None,
    model: str | None = "mock",
    mcp_profile: str | None = None,
) -> RunAnalysisResult:
    return RunAnalysisResult(
        run_id=run_id,
        task_id=task_id,
        agent_id=agent_id,
        strategy=strategy,
        success=success,
        total_score=total_score,
        tool_call_count=tool_call_count,
        llm_call_count=llm_call_count,
        duration_sec=duration_sec,
        error_count=error_count,
        metrics=metrics or {},
        model=model,
        mcp_profile=mcp_profile,
    )


def _experiment(
    *runs: RunAnalysisResult,
    exp_id: str = "exp1",
    summary: ExperimentSummary | None = None,
) -> ExperimentAnalysisResult:
    s = summary or ExperimentSummary(
        total_runs=len(runs),
        successful_runs=sum(1 for r in runs if r.success is True),
        failed_runs=sum(1 for r in runs if r.success is False),
        average_scene_score=None,
    )
    return ExperimentAnalysisResult(
        experiment_id=exp_id,
        runs=list(runs),
        summary=s,
    )


def _default_config(**kw) -> ReportConfig:
    return ReportConfig(**kw)


# ---------------------------------------------------------------------------
# build_summary_table
# ---------------------------------------------------------------------------


class TestBuildSummaryTable:
    def test_one_row_per_run(self):
        runs = [_run("r1"), _run("r2")]
        rows = build_summary_table(runs)
        assert len(rows) == 2

    def test_empty_returns_empty(self):
        assert build_summary_table([]) == []

    def test_row_has_run_id(self):
        rows = build_summary_table([_run(run_id="my-run")])
        assert rows[0]["run_id"] == "my-run"

    def test_row_has_total_score(self):
        rows = build_summary_table([_run(total_score=0.75)])
        assert rows[0]["total_score"] == pytest.approx(0.75)

    def test_row_has_tool_call_count(self):
        rows = build_summary_table([_run(tool_call_count=9)])
        assert rows[0]["tool_call_count"] == 9


# ---------------------------------------------------------------------------
# build_markdown_report — structure
# ---------------------------------------------------------------------------


class TestBuildMarkdownReportStructure:
    def test_contains_title(self):
        exp = _experiment(_run())
        cfg = _default_config(title="My Report")
        md = build_markdown_report(exp, cfg)
        assert "My Report" in md

    def test_contains_experiment_id(self):
        exp = _experiment(_run(), exp_id="test-exp")
        md = build_markdown_report(exp, _default_config())
        assert "test-exp" in md

    def test_summary_section_present(self):
        exp = _experiment(_run())
        md = build_markdown_report(exp, _default_config())
        assert "## 1. Summary" in md

    def test_best_worst_section_present(self):
        exp = _experiment(_run())
        md = build_markdown_report(exp, _default_config())
        assert "## 2. Best / Worst Runs" in md

    def test_run_details_section_present_by_default(self):
        exp = _experiment(_run())
        md = build_markdown_report(exp, _default_config())
        assert "## 8. Run Details" in md

    def test_run_details_section_hidden_when_disabled(self):
        exp = _experiment(_run())
        md = build_markdown_report(exp, _default_config(include_runs=False))
        assert "## 8. Run Details" not in md

    def test_group_comparison_present_by_default(self):
        exp = _experiment(_run())
        md = build_markdown_report(exp, _default_config())
        assert "## 3. Strategy Comparison" in md

    def test_group_comparison_hidden_when_disabled(self):
        exp = _experiment(_run())
        md = build_markdown_report(exp, _default_config(include_group_comparison=False))
        assert "## 3. Strategy Comparison" not in md

    def test_error_taxonomy_section_present_by_default(self):
        exp = _experiment(_run())
        md = build_markdown_report(exp, _default_config())
        assert "## 7. Error Taxonomy" in md

    def test_error_taxonomy_hidden_when_disabled(self):
        exp = _experiment(_run())
        md = build_markdown_report(exp, _default_config(include_error_taxonomy=False))
        assert "## 7. Error Taxonomy" not in md

    def test_artifact_links_section_present_by_default(self):
        exp = _experiment(_run())
        md = build_markdown_report(exp, _default_config())
        assert "## 9. Artifact Links" in md

    def test_artifact_links_hidden_when_disabled(self):
        exp = _experiment(_run())
        md = build_markdown_report(exp, _default_config(include_artifact_links=False))
        assert "## 9. Artifact Links" not in md

    def test_no_runs_no_crash(self):
        exp = _experiment()
        md = build_markdown_report(exp, _default_config())
        assert "## 1. Summary" in md


# ---------------------------------------------------------------------------
# build_markdown_report — content correctness
# ---------------------------------------------------------------------------


class TestBuildMarkdownReportContent:
    def test_run_id_in_details(self):
        exp = _experiment(_run(run_id="special-run"))
        md = build_markdown_report(exp, _default_config())
        assert "special-run" in md

    def test_summary_total_runs(self):
        exp = _experiment(_run("r1"), _run("r2"))
        exp = exp.model_copy(update={"summary": ExperimentSummary(total_runs=2)})
        md = build_markdown_report(exp, _default_config())
        assert "2" in md

    def test_error_taxonomy_lists_errors(self):
        run = _run(metrics={"error.tool_disabled": 3})
        exp = _experiment(run)
        md = build_markdown_report(exp, _default_config())
        assert "tool_disabled" in md

    def test_na_for_none_score(self):
        run = _run(total_score=None)
        exp = _experiment(run)
        md = build_markdown_report(exp, _default_config())
        assert "N/A" in md

    def test_multiple_runs_all_appear(self):
        exp = _experiment(_run("r1"), _run("r2"), _run("r3"))
        md = build_markdown_report(exp, _default_config())
        assert "r1" in md
        assert "r2" in md
        assert "r3" in md


# ---------------------------------------------------------------------------
# build_html_report — structure
# ---------------------------------------------------------------------------


class TestBuildHtmlReportStructure:
    def test_contains_doctype(self):
        exp = _experiment(_run())
        html = build_html_report(exp, _default_config())
        assert "<!DOCTYPE html>" in html

    def test_contains_title_in_head(self):
        exp = _experiment(_run())
        html = build_html_report(exp, _default_config(title="HTML Rep"))
        assert "<title>HTML Rep</title>" in html

    def test_contains_summary_heading(self):
        exp = _experiment(_run())
        html = build_html_report(exp, _default_config())
        assert "1. Summary" in html

    def test_contains_best_worst_heading(self):
        exp = _experiment(_run())
        html = build_html_report(exp, _default_config())
        assert "2. Best / Worst Runs" in html

    def test_group_comparison_present_by_default(self):
        exp = _experiment(_run())
        html = build_html_report(exp, _default_config())
        assert "3. Strategy Comparison" in html

    def test_group_comparison_absent_when_disabled(self):
        exp = _experiment(_run())
        html = build_html_report(exp, _default_config(include_group_comparison=False))
        assert "3. Strategy Comparison" not in html

    def test_error_taxonomy_present_by_default(self):
        exp = _experiment(_run())
        html = build_html_report(exp, _default_config())
        assert "7. Error Taxonomy" in html

    def test_error_taxonomy_absent_when_disabled(self):
        exp = _experiment(_run())
        html = build_html_report(exp, _default_config(include_error_taxonomy=False))
        assert "7. Error Taxonomy" not in html

    def test_run_details_present_by_default(self):
        exp = _experiment(_run())
        html = build_html_report(exp, _default_config())
        assert "8. Run Details" in html

    def test_run_details_absent_when_disabled(self):
        exp = _experiment(_run())
        html = build_html_report(exp, _default_config(include_runs=False))
        assert "8. Run Details" not in html

    def test_artifact_links_present_by_default(self):
        exp = _experiment(_run())
        html = build_html_report(exp, _default_config())
        assert "9. Artifact Links" in html

    def test_artifact_links_absent_when_disabled(self):
        exp = _experiment(_run())
        html = build_html_report(exp, _default_config(include_artifact_links=False))
        assert "9. Artifact Links" not in html

    def test_no_runs_no_crash(self):
        exp = _experiment()
        html = build_html_report(exp, _default_config())
        assert "<!DOCTYPE html>" in html


# ---------------------------------------------------------------------------
# build_html_report — content correctness
# ---------------------------------------------------------------------------


class TestBuildHtmlReportContent:
    def test_experiment_id_present(self):
        exp = _experiment(_run(), exp_id="html-exp")
        html = build_html_report(exp, _default_config())
        assert "html-exp" in html

    def test_run_id_in_table(self):
        exp = _experiment(_run(run_id="unique-run"))
        html = build_html_report(exp, _default_config())
        assert "unique-run" in html

    def test_pass_span_for_success_true(self):
        exp = _experiment(_run(success=True))
        html = build_html_report(exp, _default_config())
        assert 'class="pass"' in html

    def test_fail_span_for_success_false(self):
        exp = _experiment(_run(success=False, total_score=0.0))
        html = build_html_report(exp, _default_config())
        assert 'class="fail"' in html

    def test_na_span_for_none_score(self):
        exp = _experiment(_run(total_score=None, success=None))
        html = build_html_report(exp, _default_config())
        assert 'class="na"' in html

    def test_error_taxonomy_rows_populated(self):
        run = _run(metrics={"error.llm_timeout": 2})
        exp = _experiment(run)
        html = build_html_report(exp, _default_config())
        assert "llm_timeout" in html

    def test_css_style_block_present(self):
        exp = _experiment(_run())
        html = build_html_report(exp, _default_config())
        assert "<style>" in html

    def test_artifact_link_rendered(self):
        run = RunAnalysisResult(
            run_id="r1", task_id="t1", agent_id="a", strategy="react",
            artifacts=["/tmp/artifact.json"],
        )
        exp = _experiment(run)
        html = build_html_report(exp, _default_config())
        assert "/tmp/artifact.json" in html


# ---------------------------------------------------------------------------
# Fixture-based integration
# ---------------------------------------------------------------------------


class TestReportBuilderWithFixtures:
    def _load_analysis(self) -> ExperimentAnalysisResult:
        from benchmark.analysis.comparison import analyze_experiment
        return analyze_experiment(FIXTURES)

    def test_markdown_no_crash(self):
        analysis = self._load_analysis()
        md = build_markdown_report(analysis, _default_config(title="Fixture Report"))
        assert "Fixture Report" in md

    def test_html_no_crash(self):
        analysis = self._load_analysis()
        html = build_html_report(analysis, _default_config(title="Fixture HTML"))
        assert "Fixture HTML" in html

    def test_markdown_contains_run_ids_from_fixture(self):
        analysis = self._load_analysis()
        md = build_markdown_report(analysis, _default_config())
        run_ids = [r.run_id for r in analysis.runs]
        for rid in run_ids:
            assert rid in md
