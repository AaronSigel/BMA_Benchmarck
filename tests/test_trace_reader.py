"""Tests for benchmark.analysis.trace_reader."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from benchmark.analysis.trace_reader import (
    RunArtifactBundle,
    TraceReadError,
    discover_run_artifacts,
    load_run_bundle,
    read_agent_trace,
    read_run_result,
    read_validation_result,
)

FIXTURES = Path(__file__).parent / "fixtures" / "analysis"


# ---------------------------------------------------------------------------
# read_agent_trace
# ---------------------------------------------------------------------------

class TestReadAgentTrace:
    def test_reads_direct_success_fixture(self):
        trace = read_agent_trace(FIXTURES / "agent_trace_direct_success.json")
        assert trace.run_id == "run_direct_001"
        assert trace.strategy.value == "direct_tool_calling"
        assert trace.success is True
        assert len(trace.steps) == 3

    def test_reads_react_success_fixture(self):
        trace = read_agent_trace(FIXTURES / "agent_trace_react_success.json")
        assert trace.strategy.value == "react"
        assert len(trace.steps) == 6

    def test_reads_tool_error_fixture(self):
        trace = read_agent_trace(FIXTURES / "agent_trace_tool_error.json")
        assert trace.success is False
        error_steps = [s for s in trace.steps if s.error]
        assert len(error_steps) == 2

    def test_reads_plan_execute_fixture(self):
        trace = read_agent_trace(FIXTURES / "agent_trace_plan_execute_success.json")
        assert trace.strategy.value == "plan_and_execute"
        plan_steps = [s for s in trace.steps if s.step_type.value == "plan"]
        assert len(plan_steps) == 1

    def test_missing_file_raises_trace_read_error(self, tmp_path):
        with pytest.raises(TraceReadError, match="Cannot read"):
            read_agent_trace(tmp_path / "nonexistent.json")

    def test_invalid_json_raises_trace_read_error(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json")
        with pytest.raises(TraceReadError):
            read_agent_trace(bad)

    def test_accepts_string_path(self):
        trace = read_agent_trace(str(FIXTURES / "agent_trace_direct_success.json"))
        assert trace.run_id == "run_direct_001"


# ---------------------------------------------------------------------------
# read_run_result
# ---------------------------------------------------------------------------

class TestReadRunResult:
    def test_reads_success_fixture(self):
        rr = read_run_result(FIXTURES / "run_result_success.json")
        assert rr.run_id == "run_direct_001"
        assert rr.status.value == "passed"
        assert rr.total_score == 1.0

    def test_reads_error_fixture(self):
        rr = read_run_result(FIXTURES / "run_result_error.json")
        assert rr.status.value == "error"
        assert rr.total_score is None
        assert rr.error is not None

    def test_missing_file_raises_trace_read_error(self, tmp_path):
        with pytest.raises(TraceReadError):
            read_run_result(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# read_validation_result
# ---------------------------------------------------------------------------

class TestReadValidationResult:
    def test_reads_success_fixture(self):
        v = read_validation_result(FIXTURES / "validation_result_success.json")
        assert v.overall_status.value == "passed"
        assert v.total_score == 1.0
        assert len(v.validators) == 2
        assert all(vr.status.value == "passed" for vr in v.validators)

    def test_reads_partial_fixture(self):
        v = read_validation_result(FIXTURES / "validation_result_partial.json")
        assert v.overall_status.value == "warning"
        assert v.total_score == 0.6
        failed = [vr for vr in v.validators if vr.status.value == "failed"]
        assert len(failed) == 1
        assert failed[0].name == "material_validator"

    def test_missing_file_raises_trace_read_error(self, tmp_path):
        with pytest.raises(TraceReadError):
            read_validation_result(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# discover_run_artifacts
# ---------------------------------------------------------------------------

class TestDiscoverRunArtifacts:
    def test_finds_dirs_with_agent_trace(self, tmp_path):
        run_a = tmp_path / "run_a"
        run_a.mkdir()
        (run_a / "agent_trace.json").write_text(
            (FIXTURES / "agent_trace_direct_success.json").read_text()
        )
        found = discover_run_artifacts(tmp_path)
        assert run_a in found

    def test_finds_dirs_with_run_result(self, tmp_path):
        run_b = tmp_path / "run_b"
        run_b.mkdir()
        (run_b / "run_result.json").write_text(
            (FIXTURES / "run_result_success.json").read_text()
        )
        found = discover_run_artifacts(tmp_path)
        assert run_b in found

    def test_finds_both_types(self, tmp_path):
        for name, src in [
            ("run_trace", "agent_trace_direct_success.json"),
            ("run_result", "run_result_success.json"),
        ]:
            d = tmp_path / name
            d.mkdir()
            fname = "agent_trace.json" if "trace" in src else "run_result.json"
            (d / fname).write_text((FIXTURES / src).read_text())

        found = discover_run_artifacts(tmp_path)
        assert len(found) == 2

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert discover_run_artifacts(tmp_path) == []

    def test_no_duplicates_when_both_files_present(self, tmp_path):
        run_dir = tmp_path / "run_x"
        run_dir.mkdir()
        (run_dir / "agent_trace.json").write_text(
            (FIXTURES / "agent_trace_direct_success.json").read_text()
        )
        (run_dir / "run_result.json").write_text(
            (FIXTURES / "run_result_success.json").read_text()
        )
        found = discover_run_artifacts(tmp_path)
        assert found.count(run_dir) == 1

    def test_returns_sorted_paths(self, tmp_path):
        for name in ["run_c", "run_a", "run_b"]:
            d = tmp_path / name
            d.mkdir()
            (d / "run_result.json").write_text(
                (FIXTURES / "run_result_success.json").read_text()
            )
        found = discover_run_artifacts(tmp_path)
        assert found == sorted(found)


# ---------------------------------------------------------------------------
# load_run_bundle
# ---------------------------------------------------------------------------

class TestLoadRunBundle:
    def _make_run_dir(self, tmp_path, *, with_trace=False, with_run_result=False,
                      with_validation=False) -> Path:
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()
        if with_trace:
            (run_dir / "agent_trace.json").write_text(
                (FIXTURES / "agent_trace_direct_success.json").read_text()
            )
        if with_run_result:
            (run_dir / "run_result.json").write_text(
                (FIXTURES / "run_result_success.json").read_text()
            )
        if with_validation:
            (run_dir / "validation_result.json").write_text(
                (FIXTURES / "validation_result_success.json").read_text()
            )
        return run_dir

    def test_loads_all_artifacts(self, tmp_path):
        run_dir = self._make_run_dir(
            tmp_path, with_trace=True, with_run_result=True, with_validation=True
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            bundle = load_run_bundle(run_dir)
        assert bundle.agent_trace is not None
        assert bundle.run_result is not None
        assert bundle.validation_result is not None

    def test_missing_trace_handled_gracefully(self, tmp_path):
        run_dir = self._make_run_dir(tmp_path, with_run_result=True, with_validation=True)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            bundle = load_run_bundle(run_dir)
        assert bundle.agent_trace is None
        assert bundle.run_result is not None

    def test_missing_validation_emits_warning(self, tmp_path):
        run_dir = self._make_run_dir(tmp_path, with_trace=True, with_run_result=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bundle = load_run_bundle(run_dir)
        assert bundle.validation_result is None
        warning_messages = [str(warning.message) for warning in w]
        assert any("validation_result.json" in m for m in warning_messages)

    def test_missing_run_result_handled_gracefully(self, tmp_path):
        run_dir = self._make_run_dir(tmp_path, with_trace=True, with_validation=True)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            bundle = load_run_bundle(run_dir)
        assert bundle.run_result is None
        assert bundle.agent_trace is not None

    def test_completely_empty_dir_returns_empty_bundle(self, tmp_path):
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            bundle = load_run_bundle(run_dir)
        assert bundle.agent_trace is None
        assert bundle.run_result is None
        assert bundle.validation_result is None

    def test_run_dir_attribute_set_correctly(self, tmp_path):
        run_dir = self._make_run_dir(tmp_path, with_run_result=True)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            bundle = load_run_bundle(run_dir)
        assert bundle.run_dir == run_dir

    def test_accepts_string_path(self, tmp_path):
        run_dir = self._make_run_dir(tmp_path, with_run_result=True)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            bundle = load_run_bundle(str(run_dir))
        assert bundle.run_dir == run_dir

    def test_loads_optional_extras(self, tmp_path):
        run_dir = self._make_run_dir(tmp_path, with_run_result=True)
        (run_dir / "scene_snapshot.json").write_text('{"objects": []}')
        (run_dir / "metrics.json").write_text('{"tool_call_count": 3}')
        (run_dir / "summary.json").write_text('{"strategy": "react"}')
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            bundle = load_run_bundle(run_dir)
        assert bundle.scene_snapshot == {"objects": []}
        assert bundle.metrics == {"tool_call_count": 3}
        assert bundle.summary == {"strategy": "react"}

    def test_invalid_json_in_trace_gracefully_handled(self, tmp_path):
        run_dir = tmp_path / "bad_trace_run"
        run_dir.mkdir()
        (run_dir / "agent_trace.json").write_text("INVALID JSON")
        (run_dir / "run_result.json").write_text(
            (FIXTURES / "run_result_success.json").read_text()
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            bundle = load_run_bundle(run_dir)
        assert bundle.agent_trace is None
        assert bundle.run_result is not None


# ---------------------------------------------------------------------------
# RunArtifactBundle model
# ---------------------------------------------------------------------------

class TestRunArtifactBundle:
    def test_minimal_construction(self, tmp_path):
        bundle = RunArtifactBundle(run_dir=tmp_path)
        assert bundle.agent_trace is None
        assert bundle.run_result is None
        assert bundle.validation_result is None
        assert bundle.scene_snapshot is None
        assert bundle.metrics is None
        assert bundle.summary is None

    def test_with_trace(self):
        trace = read_agent_trace(FIXTURES / "agent_trace_react_success.json")
        bundle = RunArtifactBundle(run_dir=FIXTURES, agent_trace=trace)
        assert bundle.agent_trace is not None
        assert bundle.agent_trace.strategy.value == "react"
