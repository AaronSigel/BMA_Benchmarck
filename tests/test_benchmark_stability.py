"""Tests for benchmark runner stability: P0 items from ТЗ.

Covers:
- structured_error in run_result.json and agent_trace.json
- UnclassifiedError not used for known failures
- Resume mode skip/rerun logic and resume_report.json format
- validate-report-bundle CLI exit codes
- mock_report_ready matrix generates 4 runs without API keys/Blender
- diagnostic_repeat_gemini_v5 description matches repetitions
- analyze / build-report pipeline
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark.agent.models import AgentStrategyName, AgentTrace
from benchmark.runner.controlled_errors import (
    ControlledErrorType,
    normalize_error,
)
from benchmark.runner.models import (
    AgentStatus,
    ExecutionMode,
    RunResult,
    RunStatus,
    SceneStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUN_DEFAULTS = dict(
    task_id="geometry_001_basic_primitives",
    execution_mode=ExecutionMode.AGENT_MCP,
    validation_result_path=None,
    scene_snapshot_path=None,
    artifacts_dir=Path("/tmp/run"),
    started_at="2026-01-01T00:00:00Z",
    finished_at="2026-01-01T00:01:00Z",
    duration_sec=60.0,
    overall_status=None,
)


def _make_run_result(
    run_id: str = "r1",
    status: RunStatus = RunStatus.ERROR,
    error: str | None = "something went wrong",
    structured_error: dict | None = None,
    **kwargs,
) -> RunResult:
    return RunResult(
        run_id=run_id,
        status=status,
        run_status=kwargs.pop("run_status", status),
        agent_status=kwargs.pop("agent_status", AgentStatus.RUNTIME_ERROR),
        scene_status=kwargs.pop("scene_status", SceneStatus.NOT_AVAILABLE),
        error=error,
        structured_error=structured_error,
        **{**_RUN_DEFAULTS, **kwargs},
    )


# ---------------------------------------------------------------------------
# P0.1 — structured_error in RunResult
# ---------------------------------------------------------------------------

class TestRunResultStructuredError:
    def test_run_result_has_structured_error_field(self):
        result = _make_run_result(
            structured_error={
                "error_type": "ReactMaxSteps",
                "message": "ReAct strategy reached max_steps",
                "source": "agent",
                "failure_stage": "agent_execution",
                "recoverable": True,
                "raw_error": "ReAct strategy reached max_steps",
            }
        )
        assert result.structured_error is not None
        assert result.structured_error["error_type"] == "ReactMaxSteps"

    def test_run_result_structured_error_none_by_default(self):
        result = _make_run_result(structured_error=None)
        assert result.structured_error is None

    def test_run_result_json_contains_structured_error(self, tmp_path: Path):
        structured = {
            "error_type": "BlenderSocketUnavailable",
            "message": "No response from Blender socket",
            "source": "blender",
            "failure_stage": "tool_call",
            "recoverable": False,
            "raw_error": "No response from Blender socket",
        }
        result = _make_run_result(
            error="No response from Blender socket",
            structured_error=structured,
        )
        path = tmp_path / "run_result.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        data = json.loads(path.read_text())
        assert "structured_error" in data
        assert data["structured_error"]["error_type"] == "BlenderSocketUnavailable"
        assert data["structured_error"]["source"] == "blender"


# ---------------------------------------------------------------------------
# P0.1 — structured_error in AgentTrace
# ---------------------------------------------------------------------------

class TestAgentTraceStructuredError:
    def test_agent_trace_has_structured_error_field(self):
        trace = AgentTrace(
            run_id="r1",
            task_id="t1",
            agent_id="a",
            strategy=AgentStrategyName.REACT,
            success=False,
            error="ReAct strategy reached max_steps",
            structured_error={
                "error_type": "ReactMaxSteps",
                "message": "ReAct strategy reached max_steps",
                "source": "agent",
                "failure_stage": "agent_execution",
                "recoverable": True,
                "raw_error": "ReAct strategy reached max_steps",
            },
        )
        assert trace.structured_error is not None
        assert trace.structured_error["error_type"] == "ReactMaxSteps"

    def test_agent_trace_structured_error_none_by_default(self):
        trace = AgentTrace(
            run_id="r1",
            task_id="t1",
            agent_id="a",
            strategy=AgentStrategyName.DIRECT_TOOL_CALLING,
        )
        assert trace.structured_error is None

    def test_stub_trace_has_structured_error(self, tmp_path: Path):
        from benchmark.runner.experiment_runner import _write_stub_trace_if_needed
        from benchmark.runner.models import RunConfig

        config = RunConfig(
            run_id="stub_r1",
            task_id="geometry_001_basic_primitives",
            execution_mode=ExecutionMode.AGENT_MCP,
            artifacts_dir=tmp_path,
            output_dir=tmp_path,
            metadata={"agent_id": "mock_agent", "agent_strategy": "direct_tool_calling"},
        )
        from benchmark.runner.paths import RunArtifactLayout
        layout = RunArtifactLayout.from_run_output_dir(tmp_path, "stub_r1")
        layout.ensure()
        structured = {
            "error_type": "SnapshotUnavailable",
            "message": "pre-run scene snapshot could not be collected",
            "source": "blender",
            "failure_stage": "pre_run_snapshot",
            "recoverable": True,
            "raw_error": "pre-run scene snapshot could not be collected",
        }
        _write_stub_trace_if_needed(config, layout, structured, "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z")
        trace_path = layout.run_dir() / "agent_trace.json"
        assert trace_path.exists()
        data = json.loads(trace_path.read_text())
        assert "structured_error" in data
        assert data["structured_error"]["error_type"] == "SnapshotUnavailable"


# ---------------------------------------------------------------------------
# P0.2 — UnclassifiedError not used for known failures
# ---------------------------------------------------------------------------

class TestUnclassifiedErrorNotUsedForKnownFailures:
    @pytest.mark.parametrize("message,expected", [
        ("ReAct strategy reached max_steps", ControlledErrorType.REACT_MAX_STEPS),
        ("reached max_steps after 20 iterations", ControlledErrorType.REACT_MAX_STEPS),
        ("No tool call or JSON action returned by LLM", ControlledErrorType.LLM_PARSE_ERROR),
        ("did not include action in response", ControlledErrorType.LLM_PARSE_ERROR),
        ("failed to parse LLM response", ControlledErrorType.LLM_PARSE_ERROR),
        ("Invalid JSON from 'bma_create_object'", ControlledErrorType.INVALID_TOOL_RESPONSE),
        ("Tool call timed out after 60 seconds", ControlledErrorType.TOOL_TIMEOUT),
        ("pre-run scene snapshot could not be collected", ControlledErrorType.SNAPSHOT_UNAVAILABLE),
        ("scene reset failed: could not reset scene", ControlledErrorType.RESET_SCENE_FAILED),
        ("No response from Blender socket", ControlledErrorType.BLENDER_SOCKET_UNAVAILABLE),
        ("repeated the same action three times", ControlledErrorType.REACT_INVALID_ACTION),
        ("no_progress_detected after 3 steps", ControlledErrorType.REACT_NO_PROGRESS),
    ])
    def test_known_errors_not_unclassified(self, message: str, expected: ControlledErrorType):
        result = normalize_error(message)
        assert result.error_type != ControlledErrorType.UNCLASSIFIED, (
            f"'{message}' was classified as UnclassifiedError, expected {expected.value}"
        )
        assert result.error_type == expected

    def test_raw_error_always_preserved(self):
        message = "completely opaque mysterious thing happened 0xDEADBEEF"
        result = normalize_error(message)
        assert result.raw_error == message
        assert result.error_type == ControlledErrorType.UNCLASSIFIED


# ---------------------------------------------------------------------------
# P0.3 — Resume report format
# ---------------------------------------------------------------------------

class TestResumeReportFormat:
    def test_resume_report_json_has_required_fields(self, tmp_path: Path):
        from benchmark.runner.batch_runner import _write_resume_report

        entries = [
            {"run_id": "r1", "status": "completed_existing"},
            {"run_id": "r2", "status": "completed_existing"},
            {"run_id": "r3", "status": "rerun_missing"},
            {"run_id": "r4", "status": "rerun_incomplete"},
            {"run_id": "r5", "status": "rerun_corrupted"},
            {"run_id": "r6", "status": "rerun_failed_again"},
        ]
        _write_resume_report(tmp_path, entries)
        data = json.loads((tmp_path / "resume_report.json").read_text())
        assert data["resume_enabled"] is True
        assert data["total_runs"] == 6
        assert data["completed_existing"] == 2
        assert data["rerun_missing"] == 1
        assert data["rerun_incomplete"] == 1
        assert data["rerun_corrupted"] == 1
        assert data["rerun_failed_again"] == 1

    def test_resume_report_md_written(self, tmp_path: Path):
        from benchmark.runner.batch_runner import _write_resume_report
        _write_resume_report(tmp_path, [{"run_id": "r1", "status": "completed_existing"}])
        md = (tmp_path / "resume_report.md").read_text()
        assert "# Resume Report" in md
        assert "completed_existing" in md

    def test_resume_skips_completed_runs(self, tmp_path: Path):
        """BatchRunner.run_experiment with resume=True skips existing completed runs."""
        from benchmark.runner.batch_runner import BatchRunner
        from benchmark.runner.models import ExperimentConfig, RunConfig
        from benchmark.runner.artifact_manifest import write_run_artifact_manifest
        from benchmark.runner.paths import RunArtifactLayout

        # Create a fake completed run
        run_config = RunConfig(
            run_id="completed_run",
            task_id="geometry_001_basic_primitives",
            execution_mode=ExecutionMode.EXTERNAL_SNAPSHOT,
            artifacts_dir=tmp_path / "completed_run",
            output_dir=tmp_path / "completed_run",
        )
        layout = RunArtifactLayout.from_run_output_dir(tmp_path / "completed_run", "completed_run")
        layout.ensure()

        # Write a minimal run_result.json
        completed = RunResult(
            run_id="completed_run",
            task_id="geometry_001_basic_primitives",
            status=RunStatus.PASSED,
            run_status=RunStatus.PASSED,
            execution_mode=ExecutionMode.EXTERNAL_SNAPSHOT,
            validation_result_path=None,
            scene_snapshot_path=None,
            artifacts_dir=layout.run_dir(),
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:01:00Z",
            overall_status="passed",
        )
        layout.run_result_json().write_text(completed.model_dump_json(), encoding="utf-8")
        (layout.run_dir() / "metrics.json").write_text("[]", encoding="utf-8")
        write_run_artifact_manifest(completed, layout)

        mock_runner = MagicMock()
        batch = BatchRunner(runner=mock_runner)
        config = ExperimentConfig(experiment_id="test_exp", runs=[run_config])
        batch.run_experiment(config, resume=True)

        # The runner should NOT have been called (run was skipped)
        mock_runner.run.assert_not_called()

    def test_resume_reruns_missing_runs(self, tmp_path: Path):
        """BatchRunner.run_experiment with resume=True reruns missing runs."""
        from benchmark.runner.batch_runner import BatchRunner
        from benchmark.runner.models import ExperimentConfig, RunConfig

        run_config = RunConfig(
            run_id="missing_run",
            task_id="geometry_001_basic_primitives",
            execution_mode=ExecutionMode.EXTERNAL_SNAPSHOT,
            artifacts_dir=tmp_path / "missing_run",
            output_dir=tmp_path / "missing_run",
        )
        mock_result = RunResult(
            run_id="missing_run",
            task_id="geometry_001_basic_primitives",
            status=RunStatus.PASSED,
            run_status=RunStatus.PASSED,
            execution_mode=ExecutionMode.EXTERNAL_SNAPSHOT,
            validation_result_path=None,
            scene_snapshot_path=None,
            artifacts_dir=tmp_path / "missing_run",
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:01:00Z",
            overall_status="passed",
        )
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        batch = BatchRunner(runner=mock_runner)
        config = ExperimentConfig(experiment_id="test_exp", runs=[run_config])
        batch.run_experiment(config, resume=True)

        # Should have been called once for the missing run
        mock_runner.run.assert_called_once()

    def test_resume_report_written_after_resume(self, tmp_path: Path):
        """resume_report.json is written when resume=True."""
        from benchmark.runner.batch_runner import BatchRunner
        from benchmark.runner.models import ExperimentConfig, RunConfig

        run_config = RunConfig(
            run_id="some_run",
            task_id="geometry_001_basic_primitives",
            execution_mode=ExecutionMode.EXTERNAL_SNAPSHOT,
            artifacts_dir=tmp_path / "some_run",
            output_dir=tmp_path / "some_run",
        )
        mock_result = RunResult(
            run_id="some_run",
            task_id="geometry_001_basic_primitives",
            status=RunStatus.PASSED,
            run_status=RunStatus.PASSED,
            execution_mode=ExecutionMode.EXTERNAL_SNAPSHOT,
            validation_result_path=None,
            scene_snapshot_path=None,
            artifacts_dir=tmp_path / "some_run",
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:01:00Z",
            overall_status="passed",
        )
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_result
        batch = BatchRunner(runner=mock_runner)
        config = ExperimentConfig(experiment_id="test_exp", runs=[run_config])
        batch.run_experiment(config, resume=True)

        assert (tmp_path / "resume_report.json").exists()
        data = json.loads((tmp_path / "resume_report.json").read_text())
        assert "resume_enabled" in data
        assert "total_runs" in data


# ---------------------------------------------------------------------------
# P0.4 — validate-report-bundle CLI
# ---------------------------------------------------------------------------

def _build_valid_test_bundle(tmp_path: Path) -> Path:
    """Build a complete valid report bundle for CLI testing."""
    from benchmark.analysis.comparison import analyze_run_results
    from benchmark.analysis.export import write_run_metrics_csv
    from benchmark.analysis.models import ReportConfig, RunAnalysisResult
    from benchmark.analysis.report_bundle import create_report_bundle, write_figures, write_report_text_ru
    from benchmark.analysis.report_builder import build_html_report, build_markdown_report

    runs = [
        RunAnalysisResult(
            run_id="r1", task_id="geometry_001_basic_primitives", agent_id="a",
            strategy="plan_and_execute", model="mock", mcp_profile="minimal",
            pass_type="clean_pass", run_status="passed", scene_status="passed",
            agent_status="completed", total_score=1.0, duration_sec=1.0,
            tool_call_count=2, llm_call_count=1,
            metrics={"provider_cost_available": True, "provider_reported_cost_usd": 0.001},
        ),
        RunAnalysisResult(
            run_id="r2", task_id="geometry_002_positions", agent_id="a",
            strategy="react", model="mock", mcp_profile="minimal",
            pass_type="runtime_error", run_status="error", scene_status="not_available",
            agent_status="runtime_error", total_score=None, duration_sec=0.5,
            tool_call_count=0, llm_call_count=0,
            metrics={"provider_cost_available": False},
        ),
    ]

    # Create per-run artifact manifests so run_artifact_manifests.json has correct count
    for run in runs:
        run_dir = tmp_path / run.run_id
        run_dir.mkdir(exist_ok=True)
        (run_dir / "run_result.json").write_text(json.dumps({"run_id": run.run_id}), encoding="utf-8")
        (run_dir / "metrics.json").write_text("[]", encoding="utf-8")
        artifact_manifest = {
            "run_id": run.run_id,
            "task_id": run.task_id,
            "status": run.pass_type,
            "artifacts": {
                "run_result": {"path": "run_result.json", "exists": True, "required": True},
                "metrics": {"path": "metrics.json", "exists": True, "required": True},
            },
            "files": ["run_result.json", "metrics.json"],
        }
        (run_dir / "artifact_manifest.json").write_text(json.dumps(artifact_manifest), encoding="utf-8")

    # Write manifest.json with required protocol versions and hashes
    (tmp_path / "manifest.json").write_text(json.dumps({
        "experiment_id": "test_bundle_cli",
        "benchmark_protocol_version": "1.0",
        "task_schema_version": "1.0",
        "validator_version": "1.0",
        "tool_contract_version": "1.0",
        "report_schema_version": "1.0",
        "matrix_config_hash": "stub_hash_config",
        "task_set_hash": "stub_hash_tasks",
        "tool_contract_hash": "stub_hash_tools",
    }, indent=2), encoding="utf-8")

    analysis = analyze_run_results(runs, experiment_id="test_bundle_cli")
    cfg = ReportConfig(title="Test", output_dir=tmp_path)
    (tmp_path / "report.md").write_text(build_markdown_report(analysis, cfg), encoding="utf-8")
    (tmp_path / "report.html").write_text(build_html_report(analysis, cfg), encoding="utf-8")
    write_report_text_ru(analysis, tmp_path / "report_text_ru.md")
    write_figures(analysis, tmp_path / "figures")
    write_run_metrics_csv(runs, tmp_path / "summary.csv")
    (tmp_path / "summary.json").write_text(analysis.summary.model_dump_json(), encoding="utf-8")
    (tmp_path / "experiment_analysis.json").write_text(analysis.model_dump_json(), encoding="utf-8")
    return create_report_bundle(tmp_path, analysis, [
        tmp_path / "summary.csv",
        tmp_path / "summary.json",
        tmp_path / "experiment_analysis.json",
        tmp_path / "report.md",
        tmp_path / "report.html",
        tmp_path / "report_text_ru.md",
    ])


class TestValidateReportBundleCli:
    def test_valid_bundle_exits_zero(self, tmp_path: Path):
        """validate-report-bundle on a valid bundle exits 0."""
        bundle = _build_valid_test_bundle(tmp_path)
        proc = subprocess.run(
            [sys.executable, "-m", "bma_benchmark", "validate-report-bundle", str(bundle)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"Expected exit 0, got {proc.returncode}\n{proc.stdout}\n{proc.stderr}"

    def test_missing_bundle_exits_nonzero(self, tmp_path: Path):
        """validate-report-bundle on missing path exits non-zero."""
        proc = subprocess.run(
            [sys.executable, "-m", "bma_benchmark", "validate-report-bundle",
             str(tmp_path / "nonexistent_bundle")],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0

    def test_damaged_bundle_exits_nonzero(self, tmp_path: Path):
        """validate-report-bundle on a directory missing required files exits non-zero."""
        bundle = tmp_path / "report_bundle"
        bundle.mkdir()
        proc = subprocess.run(
            [sys.executable, "-m", "bma_benchmark", "validate-report-bundle", str(bundle)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert proc.stdout or proc.stderr  # Some error output


# ---------------------------------------------------------------------------
# P0.5 — mock_report_ready matrix
# ---------------------------------------------------------------------------

class TestMockReportReadyMatrix:
    def test_mock_matrix_generates_4_runs(self):
        from benchmark.experiments.generator import generate_experiment_config
        from benchmark.experiments.matrix import load_matrix

        matrix = load_matrix("configs/matrices/mock_report_ready.yaml")
        config = generate_experiment_config(matrix)
        assert matrix.metadata["expected_runs"] == 4
        assert len(config.runs) == 4

    def test_mock_matrix_uses_per_task_snapshot_paths(self):
        from benchmark.experiments.generator import generate_experiment_config
        from benchmark.experiments.matrix import load_matrix

        matrix = load_matrix("configs/matrices/mock_report_ready.yaml")
        config = generate_experiment_config(matrix)
        by_task = {r.task_id: r.snapshot_path for r in config.runs}
        assert by_task["geometry_001_basic_primitives"] is not None
        assert "mock_geometry_pass" in str(by_task["geometry_001_basic_primitives"])
        assert "missing_object" in str(by_task["geometry_002_positions"])
        assert "nonexistent" in str(by_task["geometry_004_rotation"])

    def test_mock_matrix_execution_mode_is_external_snapshot(self):
        from benchmark.experiments.generator import generate_experiment_config
        from benchmark.experiments.matrix import load_matrix

        matrix = load_matrix("configs/matrices/mock_report_ready.yaml")
        config = generate_experiment_config(matrix)
        for run in config.runs:
            assert run.execution_mode == ExecutionMode.EXTERNAL_SNAPSHOT


# ---------------------------------------------------------------------------
# P0.6 — diagnostic_repeat_gemini_v5 description matches repetitions
# ---------------------------------------------------------------------------

class TestDiagnosticRepeatDescription:
    def test_description_matches_single_repetition(self):
        from benchmark.experiments.matrix import load_matrix
        matrix = load_matrix("configs/matrices/diagnostic_repeat_gemini_v5.yaml")
        assert matrix.repetitions == 1
        assert "two-repeat" not in matrix.description.lower()
        assert "540" not in matrix.description

    def test_description_mentions_270_runs(self):
        from benchmark.experiments.matrix import load_matrix
        matrix = load_matrix("configs/matrices/diagnostic_repeat_gemini_v5.yaml")
        assert "270" in matrix.description

    def test_expected_runs_is_270(self):
        from benchmark.experiments.matrix import load_matrix
        matrix = load_matrix("configs/matrices/diagnostic_repeat_gemini_v5.yaml")
        assert matrix.metadata["expected_runs"] == 270

    def test_generates_270_runs(self):
        from benchmark.experiments.generator import generate_experiment_config
        from benchmark.experiments.matrix import load_matrix
        matrix = load_matrix("configs/matrices/diagnostic_repeat_gemini_v5.yaml")
        config = generate_experiment_config(matrix)
        assert len(config.runs) == 270
