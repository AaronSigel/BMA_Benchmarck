"""Tests for benchmark.analysis.export."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmark.analysis.export import (
    _most_common_error,
    _run_to_row,
    to_csv,
    to_html,
    to_json,
    to_markdown,
    write_error_taxonomy_csv,
    write_experiment_analysis_json,
    write_group_comparison_csv,
    write_run_analysis_json,
    write_run_metrics_csv,
)
from benchmark.analysis.models import (
    ComparisonDimension,
    ComparisonGroup,
    ErrorCategory,
    ErrorRecord,
    ExperimentAnalysisResult,
    ExperimentSummary,
    RunAnalysisResult,
)

_NOW = "2026-05-16T10:00:00Z"


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
    )


def _group(value: str = "react", run_count: int = 3) -> ComparisonGroup:
    return ComparisonGroup(
        dimension=ComparisonDimension.STRATEGY,
        value=value,
        run_count=run_count,
        success_rate=0.67,
        avg_score=0.8,
        avg_tool_calls=5.0,
        avg_duration_sec=12.0,
    )


def _experiment(*runs: RunAnalysisResult, exp_id: str = "exp1") -> ExperimentAnalysisResult:
    return ExperimentAnalysisResult(
        experiment_id=exp_id,
        runs=list(runs),
        summary=ExperimentSummary(total_runs=len(runs)),
    )


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------


class TestToJson:
    def test_dict_serialised(self):
        out = to_json({"key": "val"})
        assert json.loads(out) == {"key": "val"}

    def test_pydantic_model_serialised(self):
        run = _run()
        out = to_json(run)
        d = json.loads(out)
        assert d["run_id"] == "r1"

    def test_list_of_models_serialised(self):
        runs = [_run("r1"), _run("r2")]
        out = to_json(runs)
        lst = json.loads(out)
        assert len(lst) == 2
        assert lst[0]["run_id"] == "r1"

    def test_indent_applied(self):
        out = to_json({"a": 1}, indent=4)
        assert "    " in out

    def test_path_serialised_as_string(self):
        out = to_json({"p": Path("/tmp/x")})
        d = json.loads(out)
        assert d["p"] == "/tmp/x"


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------


class TestToCsv:
    def test_empty_returns_empty_string(self):
        assert to_csv([]) == ""

    def test_header_present(self):
        out = to_csv([{"a": 1, "b": 2}])
        lines = out.splitlines()
        assert lines[0] == "a,b"

    def test_data_row_present(self):
        out = to_csv([{"a": 1, "b": 2}])
        lines = out.splitlines()
        assert lines[1] == "1,2"

    def test_multiple_rows(self):
        rows = [{"x": i} for i in range(5)]
        out = to_csv(rows)
        assert len(out.splitlines()) == 6  # header + 5 rows


# ---------------------------------------------------------------------------
# to_markdown
# ---------------------------------------------------------------------------


class TestToMarkdown:
    def test_empty_no_title(self):
        out = to_markdown([])
        assert "No data" in out

    def test_empty_with_title(self):
        out = to_markdown([], title="My Report")
        assert "My Report" in out
        assert "No data" in out

    def test_header_row(self):
        out = to_markdown([{"col1": "a", "col2": "b"}])
        assert "col1" in out
        assert "col2" in out

    def test_separator_row(self):
        out = to_markdown([{"col1": "a"}])
        assert "---" in out

    def test_data_cell(self):
        out = to_markdown([{"name": "reactor"}])
        assert "reactor" in out

    def test_title_heading(self):
        out = to_markdown([{"x": 1}], title="Title")
        assert out.startswith("# Title")


# ---------------------------------------------------------------------------
# to_html
# ---------------------------------------------------------------------------


class TestToHtml:
    def test_contains_doctype(self):
        out = to_html([{"a": 1}])
        assert "<!DOCTYPE html>" in out

    def test_heading_present_when_title(self):
        out = to_html([{"a": 1}], title="Rep")
        assert "<h1>Rep</h1>" in out

    def test_table_header(self):
        out = to_html([{"col": "v"}])
        assert "<th>col</th>" in out

    def test_table_cell(self):
        out = to_html([{"col": "hello"}])
        assert "<td>hello</td>" in out

    def test_empty_rows_no_crash(self):
        out = to_html([])
        assert "<!DOCTYPE html>" in out


# ---------------------------------------------------------------------------
# _most_common_error
# ---------------------------------------------------------------------------


class TestMostCommonError:
    def test_no_error_metrics_returns_empty(self):
        run = _run(metrics={})
        assert _most_common_error(run) == ""

    def test_single_error_returned(self):
        run = _run(metrics={"error.tool_runtime_error": 3})
        assert _most_common_error(run) == "tool_runtime_error"

    def test_picks_highest_count(self):
        run = _run(metrics={
            "error.tool_disabled": 1,
            "error.tool_runtime_error": 5,
        })
        assert _most_common_error(run) == "tool_runtime_error"

    def test_zero_count_ignored(self):
        run = _run(metrics={"error.tool_disabled": 0, "error.llm_timeout": 2})
        assert _most_common_error(run) == "llm_timeout"

    def test_non_error_key_ignored(self):
        run = _run(metrics={"tool_call_count": 10, "error.agent_step_limit": 1})
        assert _most_common_error(run) == "agent_step_limit"


# ---------------------------------------------------------------------------
# _run_to_row
# ---------------------------------------------------------------------------


class TestRunToRow:
    def test_all_columns_present(self):
        from benchmark.analysis.export import _RUN_METRICS_COLUMNS
        row = _run_to_row(_run())
        for col in _RUN_METRICS_COLUMNS:
            assert col in row

    def test_most_common_error_populated(self):
        run = _run(metrics={"error.tool_disabled": 2})
        row = _run_to_row(run)
        assert row["most_common_error"] == "tool_disabled"

    def test_none_model_becomes_empty_string(self):
        run = RunAnalysisResult(
            run_id="r1", task_id="t1", agent_id="a", strategy="react", model=None,
        )
        row = _run_to_row(run)
        assert row["model"] == ""


# ---------------------------------------------------------------------------
# write_run_analysis_json
# ---------------------------------------------------------------------------


class TestWriteRunAnalysisJson:
    def test_file_created(self, tmp_path):
        out = tmp_path / "run_analysis.json"
        write_run_analysis_json(_run(), out)
        assert out.exists()

    def test_file_round_trips(self, tmp_path):
        out = tmp_path / "run.json"
        run = _run(run_id="rx", task_id="task-x")
        write_run_analysis_json(run, out)
        d = json.loads(out.read_text())
        assert d["run_id"] == "rx"
        assert d["task_id"] == "task-x"

    def test_parent_dirs_created(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "run.json"
        write_run_analysis_json(_run(), out)
        assert out.exists()


# ---------------------------------------------------------------------------
# write_experiment_analysis_json
# ---------------------------------------------------------------------------


class TestWriteExperimentAnalysisJson:
    def test_file_created(self, tmp_path):
        out = tmp_path / "exp.json"
        write_experiment_analysis_json(_experiment(_run()), out)
        assert out.exists()

    def test_experiment_id_preserved(self, tmp_path):
        out = tmp_path / "exp.json"
        write_experiment_analysis_json(_experiment(_run(), exp_id="my-exp"), out)
        d = json.loads(out.read_text())
        assert d["experiment_id"] == "my-exp"

    def test_runs_preserved(self, tmp_path):
        out = tmp_path / "exp.json"
        write_experiment_analysis_json(_experiment(_run("r1"), _run("r2")), out)
        d = json.loads(out.read_text())
        assert len(d["runs"]) == 2


# ---------------------------------------------------------------------------
# write_run_metrics_csv
# ---------------------------------------------------------------------------


class TestWriteRunMetricsCsv:
    def test_file_created(self, tmp_path):
        out = tmp_path / "metrics.csv"
        write_run_metrics_csv([_run()], out)
        assert out.exists()

    def test_header_matches_columns(self, tmp_path):
        from benchmark.analysis.export import _RUN_METRICS_COLUMNS
        out = tmp_path / "metrics.csv"
        write_run_metrics_csv([_run()], out)
        with out.open() as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == _RUN_METRICS_COLUMNS

    def test_one_row_per_run(self, tmp_path):
        out = tmp_path / "metrics.csv"
        write_run_metrics_csv([_run("r1"), _run("r2")], out)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_empty_list_header_only(self, tmp_path):
        out = tmp_path / "metrics.csv"
        write_run_metrics_csv([], out)
        assert out.exists()
        with out.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows == []

    def test_run_id_in_csv(self, tmp_path):
        out = tmp_path / "metrics.csv"
        write_run_metrics_csv([_run(run_id="unique-run")], out)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["run_id"] == "unique-run"


# ---------------------------------------------------------------------------
# write_group_comparison_csv
# ---------------------------------------------------------------------------


class TestWriteGroupComparisonCsv:
    def test_file_created(self, tmp_path):
        out = tmp_path / "groups.csv"
        write_group_comparison_csv([_group()], out)
        assert out.exists()

    def test_dimension_value_present(self, tmp_path):
        out = tmp_path / "groups.csv"
        write_group_comparison_csv([_group("direct_tool_calling")], out)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["value"] == "direct_tool_calling"

    def test_run_count_in_csv(self, tmp_path):
        out = tmp_path / "groups.csv"
        write_group_comparison_csv([_group(run_count=7)], out)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["run_count"] == "7"

    def test_expected_columns(self, tmp_path):
        out = tmp_path / "groups.csv"
        write_group_comparison_csv([_group()], out)
        with out.open() as f:
            reader = csv.DictReader(f)
            assert "dimension" in reader.fieldnames
            assert "success_rate" in reader.fieldnames
            assert "avg_score" in reader.fieldnames


# ---------------------------------------------------------------------------
# write_error_taxonomy_csv — dict form
# ---------------------------------------------------------------------------


class TestWriteErrorTaxonomyCsvDict:
    def test_file_created(self, tmp_path):
        out = tmp_path / "errors.csv"
        write_error_taxonomy_csv({"tool_disabled": 3}, out)
        assert out.exists()

    def test_columns_are_category_and_count(self, tmp_path):
        out = tmp_path / "errors.csv"
        write_error_taxonomy_csv({"tool_disabled": 1}, out)
        with out.open() as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == {"category", "count"}

    def test_category_and_count_values(self, tmp_path):
        out = tmp_path / "errors.csv"
        write_error_taxonomy_csv({"tool_runtime_error": 5}, out)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["category"] == "tool_runtime_error"
        assert rows[0]["count"] == "5"

    def test_empty_dict(self, tmp_path):
        out = tmp_path / "errors.csv"
        write_error_taxonomy_csv({}, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# write_error_taxonomy_csv — list[ErrorRecord] form
# ---------------------------------------------------------------------------


class TestWriteErrorTaxonomyCsvList:
    def _record(self, cat: ErrorCategory = ErrorCategory.TOOL_DISABLED) -> ErrorRecord:
        return ErrorRecord(
            run_id="r1", task_id="t1", step_index=0,
            category=cat, message="boom",
        )

    def test_list_form_file_created(self, tmp_path):
        out = tmp_path / "errors.csv"
        write_error_taxonomy_csv([self._record()], out)
        assert out.exists()

    def test_list_form_columns(self, tmp_path):
        out = tmp_path / "errors.csv"
        write_error_taxonomy_csv([self._record()], out)
        with out.open() as f:
            reader = csv.DictReader(f)
            assert "run_id" in reader.fieldnames
            assert "category" in reader.fieldnames
            assert "message" in reader.fieldnames

    def test_list_form_row_values(self, tmp_path):
        out = tmp_path / "errors.csv"
        write_error_taxonomy_csv([self._record(ErrorCategory.LLM_TIMEOUT)], out)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["category"] == "llm_timeout"
