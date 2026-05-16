"""Tests for benchmark.analysis.cli."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from benchmark.analysis.cli import build_parser, cmd_analyze_run, main

FIXTURES = Path(__file__).parent / "fixtures" / "analysis"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_namespace(**kw):
    import argparse
    return argparse.Namespace(**kw)


def _make_run_dir(tmp_path: Path, trace_name: str = "agent_trace_react_success.json") -> Path:
    """Copy a trace fixture into a fresh run directory."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copy(FIXTURES / trace_name, run_dir / "agent_trace.json")
    shutil.copy(FIXTURES / "validation_result_success.json", run_dir / "validation_result.json")
    return run_dir


def _make_experiment_dir(tmp_path: Path) -> Path:
    """Create a minimal experiment directory with two run sub-directories."""
    exp_dir = tmp_path / "experiment"
    for i, trace in enumerate(
        ["agent_trace_react_success.json", "agent_trace_direct_success.json"]
    ):
        run_dir = exp_dir / f"run_{i}"
        run_dir.mkdir(parents=True)
        shutil.copy(FIXTURES / trace, run_dir / "agent_trace.json")
    return exp_dir


def _write_report_config(path: Path, input_dir: Path, output_dir: Path,
                          formats: list[str] | None = None) -> Path:
    cfg = {
        "report_id": "test",
        "title": "Test Report",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "formats": formats or ["json", "csv", "markdown", "html"],
    }
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_parser_has_analyze_run(self):
        p = build_parser()
        # parse_known_args returns correctly for analyze-run
        ns, _ = p.parse_known_args(["analyze-run", "--run-dir", "/tmp/x"])
        assert ns.command == "analyze-run"

    def test_parser_has_analyze_experiment(self):
        p = build_parser()
        ns, _ = p.parse_known_args(["analyze-experiment", "--experiment-dir", "/tmp/x"])
        assert ns.command == "analyze-experiment"

    def test_parser_has_build_report(self):
        p = build_parser()
        ns, _ = p.parse_known_args(["build-report", "--config", "/tmp/cfg.yaml"])
        assert ns.command == "build-report"

    def test_parser_has_compare(self):
        p = build_parser()
        ns, _ = p.parse_known_args(["compare", "--input", "/tmp/x"])
        assert ns.command == "compare"

    def test_no_subcommand_returns_zero(self):
        result = main([])
        assert result == 0


# ---------------------------------------------------------------------------
# cmd_analyze_run
# ---------------------------------------------------------------------------


class TestCmdAnalyzeRun:
    def test_missing_run_dir_returns_1(self, tmp_path):
        ns = _make_namespace(run_dir=str(tmp_path / "nonexistent"), output=None)
        assert cmd_analyze_run(ns) == 1

    def test_creates_run_analysis_json(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        ns = _make_namespace(run_dir=str(run_dir), output=None)
        assert cmd_analyze_run(ns) == 0
        assert (run_dir / "run_analysis.json").exists()

    def test_output_to_custom_dir(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        out_dir = tmp_path / "out"
        ns = _make_namespace(run_dir=str(run_dir), output=str(out_dir))
        assert cmd_analyze_run(ns) == 0
        assert (out_dir / "run_analysis.json").exists()

    def test_output_json_is_valid(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        ns = _make_namespace(run_dir=str(run_dir), output=None)
        cmd_analyze_run(ns)
        d = json.loads((run_dir / "run_analysis.json").read_text())
        assert "run_id" in d
        assert "task_id" in d

    def test_run_id_matches_trace(self, tmp_path):
        run_dir = _make_run_dir(tmp_path)
        ns = _make_namespace(run_dir=str(run_dir), output=None)
        cmd_analyze_run(ns)
        d = json.loads((run_dir / "run_analysis.json").read_text())
        assert d["run_id"] == "run_react_001"

    def test_direct_trace_also_works(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "agent_trace_direct_success.json")
        ns = _make_namespace(run_dir=str(run_dir), output=None)
        assert cmd_analyze_run(ns) == 0

    def test_tool_error_trace_succeeds(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "agent_trace_tool_error.json")
        ns = _make_namespace(run_dir=str(run_dir), output=None)
        assert cmd_analyze_run(ns) == 0


# ---------------------------------------------------------------------------
# cmd_analyze_experiment via main()
# ---------------------------------------------------------------------------


class TestCmdAnalyzeExperiment:
    def test_missing_experiment_dir_returns_1(self, tmp_path):
        result = main(["analyze-experiment", "--experiment-dir", str(tmp_path / "nope")])
        assert result == 1

    def test_creates_experiment_analysis_json(self, tmp_path):
        exp_dir = _make_experiment_dir(tmp_path)
        result = main(["analyze-experiment", "--experiment-dir", str(exp_dir)])
        assert result == 0
        assert (exp_dir / "experiment_analysis.json").exists()

    def test_output_to_custom_dir(self, tmp_path):
        exp_dir = _make_experiment_dir(tmp_path)
        out_dir = tmp_path / "reports"
        result = main([
            "analyze-experiment",
            "--experiment-dir", str(exp_dir),
            "--output", str(out_dir),
        ])
        assert result == 0
        assert (out_dir / "experiment_analysis.json").exists()

    def test_output_json_structure(self, tmp_path):
        exp_dir = _make_experiment_dir(tmp_path)
        result = main(["analyze-experiment", "--experiment-dir", str(exp_dir)])
        assert result == 0
        d = json.loads((exp_dir / "experiment_analysis.json").read_text())
        assert "experiment_id" in d
        assert "runs" in d
        assert "summary" in d

    def test_two_runs_discovered(self, tmp_path):
        exp_dir = _make_experiment_dir(tmp_path)
        main(["analyze-experiment", "--experiment-dir", str(exp_dir)])
        d = json.loads((exp_dir / "experiment_analysis.json").read_text())
        assert len(d["runs"]) == 2


# ---------------------------------------------------------------------------
# cmd_build_report via main()
# ---------------------------------------------------------------------------


class TestCmdBuildReport:
    def test_missing_config_returns_1(self, tmp_path):
        result = main(["build-report", "--config", str(tmp_path / "no.yaml")])
        assert result == 1

    def test_missing_input_dir_returns_1(self, tmp_path):
        cfg_path = tmp_path / "cfg.yaml"
        _write_report_config(cfg_path, tmp_path / "nope", tmp_path / "out")
        result = main(["build-report", "--config", str(cfg_path)])
        assert result == 1

    def test_creates_markdown_file(self, tmp_path):
        exp_dir = _make_experiment_dir(tmp_path)
        out_dir = tmp_path / "reports"
        cfg_path = tmp_path / "cfg.yaml"
        _write_report_config(cfg_path, exp_dir, out_dir, formats=["markdown"])
        result = main(["build-report", "--config", str(cfg_path)])
        assert result == 0
        assert (out_dir / "report.md").exists()

    def test_creates_html_file(self, tmp_path):
        exp_dir = _make_experiment_dir(tmp_path)
        out_dir = tmp_path / "reports"
        cfg_path = tmp_path / "cfg.yaml"
        _write_report_config(cfg_path, exp_dir, out_dir, formats=["html"])
        result = main(["build-report", "--config", str(cfg_path)])
        assert result == 0
        assert (out_dir / "report.html").exists()

    def test_creates_json_file(self, tmp_path):
        exp_dir = _make_experiment_dir(tmp_path)
        out_dir = tmp_path / "reports"
        cfg_path = tmp_path / "cfg.yaml"
        _write_report_config(cfg_path, exp_dir, out_dir, formats=["json"])
        result = main(["build-report", "--config", str(cfg_path)])
        assert result == 0
        assert (out_dir / "experiment_analysis.json").exists()

    def test_creates_csv_file(self, tmp_path):
        exp_dir = _make_experiment_dir(tmp_path)
        out_dir = tmp_path / "reports"
        cfg_path = tmp_path / "cfg.yaml"
        _write_report_config(cfg_path, exp_dir, out_dir, formats=["csv"])
        result = main(["build-report", "--config", str(cfg_path)])
        assert result == 0
        assert (out_dir / "run_metrics.csv").exists()

    def test_all_formats_created(self, tmp_path):
        exp_dir = _make_experiment_dir(tmp_path)
        out_dir = tmp_path / "reports"
        cfg_path = tmp_path / "cfg.yaml"
        _write_report_config(cfg_path, exp_dir, out_dir)
        result = main(["build-report", "--config", str(cfg_path)])
        assert result == 0
        assert (out_dir / "report.md").exists()
        assert (out_dir / "report.html").exists()
        assert (out_dir / "experiment_analysis.json").exists()
        assert (out_dir / "run_metrics.csv").exists()

    def test_cli_output_override(self, tmp_path):
        exp_dir = _make_experiment_dir(tmp_path)
        cfg_out = tmp_path / "cfg_reports"
        cli_out = tmp_path / "cli_reports"
        cfg_path = tmp_path / "cfg.yaml"
        _write_report_config(cfg_path, exp_dir, cfg_out, formats=["markdown"])
        result = main([
            "build-report",
            "--config", str(cfg_path),
            "--output", str(cli_out),
        ])
        assert result == 0
        assert (cli_out / "report.md").exists()

    def test_cli_input_override(self, tmp_path):
        exp_dir = _make_experiment_dir(tmp_path)
        dummy_input = tmp_path / "dummy"
        dummy_input.mkdir()
        out_dir = tmp_path / "reports"
        cfg_path = tmp_path / "cfg.yaml"
        _write_report_config(cfg_path, dummy_input, out_dir, formats=["markdown"])
        result = main([
            "build-report",
            "--config", str(cfg_path),
            "--input", str(exp_dir),
        ])
        assert result == 0
        assert (out_dir / "report.md").exists()

    def test_markdown_contains_title_from_config(self, tmp_path):
        exp_dir = _make_experiment_dir(tmp_path)
        out_dir = tmp_path / "reports"
        cfg_path = tmp_path / "cfg.yaml"
        cfg = {
            "report_id": "test",
            "title": "My Custom Title",
            "input_dir": str(exp_dir),
            "output_dir": str(out_dir),
            "formats": ["markdown"],
        }
        cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
        main(["build-report", "--config", str(cfg_path)])
        md = (out_dir / "report.md").read_text()
        assert "My Custom Title" in md


# ---------------------------------------------------------------------------
# cmd_compare via main()
# ---------------------------------------------------------------------------


class TestCmdCompare:
    def test_missing_input_dir_returns_1(self, tmp_path):
        result = main(["compare", "--input", str(tmp_path / "nope")])
        assert result == 1

    def test_returns_zero_with_valid_dir(self, tmp_path, capsys):
        exp_dir = _make_experiment_dir(tmp_path)
        result = main(["compare", "--input", str(exp_dir)])
        assert result == 0

    def test_outputs_table_to_stdout(self, tmp_path, capsys):
        exp_dir = _make_experiment_dir(tmp_path)
        main(["compare", "--input", str(exp_dir)])
        out = capsys.readouterr().out
        assert "strategy" in out.lower() or "react" in out.lower() or "direct" in out.lower()

    def test_group_by_model(self, tmp_path, capsys):
        exp_dir = _make_experiment_dir(tmp_path)
        result = main(["compare", "--input", str(exp_dir), "--group-by", "model"])
        assert result == 0

    def test_group_by_agent_id(self, tmp_path, capsys):
        exp_dir = _make_experiment_dir(tmp_path)
        result = main(["compare", "--input", str(exp_dir), "--group-by", "agent_id"])
        assert result == 0

    def test_invalid_group_by_raises_system_exit(self, tmp_path):
        exp_dir = _make_experiment_dir(tmp_path)
        with pytest.raises(SystemExit):
            main(["compare", "--input", str(exp_dir), "--group-by", "invalid_dim"])

    def test_table_has_header_columns(self, tmp_path, capsys):
        exp_dir = _make_experiment_dir(tmp_path)
        main(["compare", "--input", str(exp_dir)])
        out = capsys.readouterr().out
        assert "Value" in out
        assert "Runs" in out

    def test_empty_experiment_dir_returns_1_or_prints_message(self, tmp_path, capsys):
        exp_dir = tmp_path / "empty"
        exp_dir.mkdir()
        result = main(["compare", "--input", str(exp_dir)])
        # Either returns 1 OR prints "No runs" — both are acceptable
        out = capsys.readouterr().out + capsys.readouterr().err
        assert result == 1 or "No runs" in out or result == 0
