from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from benchmark.agent.strategies.registry import STRATEGY_REGISTRY
from benchmark.analysis.report_bundle_validator import validate_report_bundle_result
from benchmark.experiments.e2e_runner import E2EBenchmarkRunner
from benchmark.experiments.generator import generate_experiment_config
from benchmark.experiments.matrix import load_matrix
from benchmark.experiments.preflight import build_preflight_report
from benchmark.runner.artifact_manifest import validate_run_artifact_manifest
from benchmark.runner.batch_runner import BatchRunner
from benchmark.runner.config_loader import load_experiment_config
from benchmark.runner.controlled_errors import normalize_error
from benchmark.runner.execution import ExecutionResult
from benchmark.runner.experiment_runner import ExperimentRunner
from benchmark.runner.models import ExecutionMode, ExperimentConfig
from benchmark.runner.models import RunConfig, RunResult, RunStatus


def test_mock_report_ready_matrix_is_offline_and_small() -> None:
    matrix = load_matrix("configs/matrices/mock_report_ready.yaml")
    config = generate_experiment_config(matrix)
    assert len(config.runs) == matrix.metadata["expected_runs"]
    assert {run.execution_mode.value for run in config.runs} == {"external_snapshot"}
    assert matrix.metadata["expected_runs"] >= 2


def test_repair_strategy_smoke_matrix_shape() -> None:
    matrix = load_matrix("configs/matrices/repair_strategy_smoke_v1.yaml")
    config = generate_experiment_config(matrix)
    assert len(config.runs) == 24
    strategies = {run.metadata["agent_strategy"] for run in config.runs}
    assert strategies == {"plan_and_execute", "plan_execute_react_repair"}


def test_strategy_registry_lists_and_creates_registered_strategy() -> None:
    assert "plan_execute_react_repair" in STRATEGY_REGISTRY.names()
    strategy = STRATEGY_REGISTRY.create("direct_tool_calling")
    assert strategy.__class__.__name__ == "DirectToolCallingStrategy"


def test_controlled_error_normalizes_snapshot_and_provider_errors() -> None:
    snapshot = normalize_error("scene snapshot does not exist")
    provider = normalize_error("OpenRouter timeout")
    assert snapshot.error_type.value == "SnapshotUnavailable"
    assert snapshot.failure_stage.value == "pre_run_snapshot"
    assert provider.error_type.value == "LlmProviderError"
    assert "UnknownError" not in snapshot.model_dump_json()


def test_unknown_error_is_not_used_for_known_failures() -> None:
    cases = {
        "ReAct strategy reached max_steps": "ReactMaxSteps",
        "No tool call or JSON action returned by LLM": "LlmParseError",
        "Invalid JSON from tool": "InvalidToolResponse",
        "Tool call timed out after 60 seconds": "ToolTimeout",
        "scene reset failed": "ResetSceneFailed",
        "No response from Blender socket": "BlenderSocketNoResponse",
    }
    for message, expected in cases.items():
        error = normalize_error(message)
        assert error.error_type.value == expected
        assert error.error_type.value != "UnknownError"


def test_artifact_manifest_is_written_for_external_snapshot_run(tmp_path: Path) -> None:
    matrix = load_matrix("configs/matrices/mock_report_ready.yaml")
    config = generate_experiment_config(matrix)
    run = config.runs[0].model_copy(update={"output_dir": tmp_path / config.runs[0].run_id})

    result = BatchRunner().runner.run(run)

    manifest_path = Path(result.artifacts_dir) / "artifact_manifest.json"
    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == run.run_id
    assert payload["artifacts"]["run_result"]["exists"] is True
    assert payload["artifacts"]["metrics"]["exists"] is True
    ok, errors = validate_run_artifact_manifest(result.artifacts_dir)
    assert ok, errors


class EarlyFailureBackend:
    mode = ExecutionMode.AGENT_MCP

    def execute(self, config: RunConfig) -> ExecutionResult:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return ExecutionResult(
            ok=False,
            scene_snapshot_path=None,
            artifacts_dir=output_dir,
            error="pre-run scene snapshot could not be collected",
            metadata={
                "strategy": "react",
                "agent_id": "mock_agent",
                "mcp_profile": "minimal",
            },
        )


def test_stub_trace_created_for_pre_run_failure(tmp_path: Path) -> None:
    run = RunConfig(
        run_id="r_stub",
        task_id="geometry_001_basic_primitives",
        execution_mode=ExecutionMode.AGENT_MCP,
        task_path=Path("tasks/geometry/geometry_001_basic_primitives.yaml"),
        artifacts_dir=tmp_path / "r_stub",
        output_dir=tmp_path / "r_stub",
        metadata={
            "agent_id": "mock_agent",
            "agent_strategy": "react",
            "model_id": "mock",
            "mcp_profile": "minimal",
        },
    )

    result = ExperimentRunner(backends={ExecutionMode.AGENT_MCP: EarlyFailureBackend()}).run(run)
    run_dir = Path(result.artifacts_dir)

    trace = json.loads((run_dir / "agent_trace.json").read_text(encoding="utf-8"))
    assert trace["steps"] == []
    assert trace["success"] is False
    assert trace["error"]["error_type"] == "SnapshotUnavailable"
    assert trace["metadata"]["stub_trace"] is True
    assert trace["metadata"]["failure_stage"] == "pre_run_snapshot"
    assert (run_dir / "scene_snapshot_not_available.json").is_file()
    assert (run_dir / "validation_result_not_available.json").is_file()


def test_every_run_has_required_artifacts_or_not_available_marker(tmp_path: Path) -> None:
    run = RunConfig(
        run_id="r_contract",
        task_id="geometry_001_basic_primitives",
        execution_mode=ExecutionMode.AGENT_MCP,
        task_path=Path("tasks/geometry/geometry_001_basic_primitives.yaml"),
        artifacts_dir=tmp_path / "r_contract",
        output_dir=tmp_path / "r_contract",
        metadata={"agent_id": "mock_agent", "agent_strategy": "react"},
    )

    result = ExperimentRunner(backends={ExecutionMode.AGENT_MCP: EarlyFailureBackend()}).run(run)
    run_dir = Path(result.artifacts_dir)

    for name in ("run_result.json", "agent_trace.json", "artifact_manifest.json", "metrics.json"):
        assert (run_dir / name).is_file()
    for name in ("scene_snapshot", "validation_result"):
        assert (run_dir / f"{name}.json").exists() or (run_dir / f"{name}_not_available.json").exists()
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["agent_trace"]["required"] is True
    assert manifest["artifacts"]["agent_trace"]["exists"] is True
    assert manifest["artifacts"]["scene_snapshot"]["not_available_reason"]


class CountingRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, config: RunConfig) -> RunResult:
        self.calls.append(config.run_id)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = RunResult(
            run_id=config.run_id,
            task_id=config.task_id,
            status=RunStatus.PASSED,
            execution_mode=config.execution_mode,
            validation_result_path=output_dir / "validation_result.json",
            scene_snapshot_path=config.snapshot_path,
            artifacts_dir=output_dir,
            total_score=1.0,
            overall_status="passed",
            started_at="2026-05-21T00:00:00Z",
            finished_at="2026-05-21T00:00:01Z",
            duration_sec=1.0,
            summary={},
        )
        (output_dir / "run_result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        (output_dir / "metrics.json").write_text("[]", encoding="utf-8")
        (output_dir / "artifact_manifest.json").write_text(
            json.dumps({
                "run_id": config.run_id,
                "task_id": config.task_id,
                "status": "clean_pass",
                "artifacts": {
                    "run_result": {"path": "run_result.json", "exists": True, "required": True, "not_available_reason": None},
                    "agent_trace": {"path": "agent_trace.json", "exists": False, "required": False, "not_available_reason": None},
                    "scene_snapshot": {"path": "scene_snapshot.json", "exists": False, "required": False, "not_available_reason": None},
                    "validation_result": {"path": "validation_result.json", "exists": False, "required": False, "not_available_reason": None},
                    "metrics": {"path": "metrics.json", "exists": True, "required": True, "not_available_reason": None},
                    "exports": [],
                },
                "files": ["run_result.json", "metrics.json"],
            }),
            encoding="utf-8",
        )
        return result


def test_batch_runner_resume_skips_valid_existing_run(tmp_path: Path) -> None:
    fixture = load_experiment_config(Path("tests/fixtures/runner/experiment_external_snapshots.yaml"))
    run = fixture.runs[0].model_copy(update={"output_dir": tmp_path / fixture.runs[0].run_id})
    config = fixture.model_copy(update={"runs": [run]})
    runner = CountingRunner()

    BatchRunner(runner=runner).run_experiment(config)
    BatchRunner(runner=runner).run_experiment(config, resume=True)

    assert runner.calls == [run.run_id]
    status = json.loads((tmp_path / "resume_status.json").read_text(encoding="utf-8"))
    assert status[run.run_id] == "skipped_existing"
    report = json.loads((tmp_path / "resume_report.json").read_text(encoding="utf-8"))
    assert report["runs"][0]["status"] == "completed_existing"


def test_resume_reruns_incomplete_runs(tmp_path: Path) -> None:
    fixture = load_experiment_config(Path("tests/fixtures/runner/experiment_external_snapshots.yaml"))
    run = fixture.runs[0].model_copy(update={"output_dir": tmp_path / fixture.runs[0].run_id})
    config = fixture.model_copy(update={"runs": [run]})
    runner = CountingRunner()

    BatchRunner(runner=runner).run_experiment(config, resume=True)

    assert runner.calls == [run.run_id]
    report = json.loads((tmp_path / "resume_report.json").read_text(encoding="utf-8"))
    assert report["runs"][0]["status"] in {"rerun_missing", "rerun_failed_again"}


def test_preflight_delegated_checks_do_not_force_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ExperimentConfig(
        experiment_id="preflight_delegated",
        runs=[
            RunConfig(
                run_id="r1",
                task_id="geometry_001_basic_primitives",
                execution_mode=ExecutionMode.AGENT_MCP,
                artifacts_dir=tmp_path / "r1",
                output_dir=tmp_path / "r1",
                mcp_config_path=tmp_path / "mcp.yaml",
                mcp_profile="no_python",
            )
        ],
    )
    (tmp_path / "mcp.yaml").write_text("profile: no_python\nblender_host: localhost\nblender_port: 9876\n", encoding="utf-8")
    monkeypatch.setattr("benchmark.experiments.preflight.find_blender_executable", lambda: Path("/usr/bin/blender"))
    monkeypatch.setattr("benchmark.experiments.preflight._find_blender_process", lambda host, port: {"pid": 1})

    report = build_preflight_report(config, tmp_path)

    statuses = {check["name"]: check["status"] for check in report["checks"]}
    assert statuses["reset_scene"] == "delegated"
    assert statuses["get_scene_snapshot"] == "delegated"
    assert statuses["smoke_tool_call"] == "delegated"
    assert report["status"] == "passed"


def _write_tmp_mock_matrix(tmp_path: Path) -> Path:
    data = yaml.safe_load(Path("configs/matrices/mock_report_ready.yaml").read_text(encoding="utf-8"))
    data["output_root"] = str(tmp_path / "mock_report_ready")
    path = tmp_path / "mock_report_ready.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_validate_report_bundle_passes_for_valid_bundle(tmp_path: Path) -> None:
    matrix_path = _write_tmp_mock_matrix(tmp_path)
    E2EBenchmarkRunner().run_and_report(matrix_path, clean_output=True)
    bundle = tmp_path / "mock_report_ready" / "report_bundle"

    result = validate_report_bundle_result(bundle)

    assert result["status"] == "passed"
    assert (bundle / "bundle_validation_result.json").is_file()
    assert (bundle / "bundle_validation_result.md").is_file()


def test_validate_report_bundle_fails_for_missing_file(tmp_path: Path) -> None:
    matrix_path = _write_tmp_mock_matrix(tmp_path)
    E2EBenchmarkRunner().run_and_report(matrix_path, clean_output=True)
    bundle = tmp_path / "mock_report_ready" / "report_bundle"
    (bundle / "summary.csv").unlink()

    result = validate_report_bundle_result(bundle)

    assert result["status"] == "failed"


def test_analyze_rebuilds_summary_without_llm(tmp_path: Path) -> None:
    matrix_path = _write_tmp_mock_matrix(tmp_path)
    root = tmp_path / "mock_report_ready"
    E2EBenchmarkRunner().run(matrix_path, clean_output=True)
    (root / "summary.csv").unlink(missing_ok=True)

    from benchmark.experiments.e2e_runner import run_analysis

    analysis = run_analysis(root)

    matrix = load_matrix("configs/matrices/mock_report_ready.yaml")
    assert analysis.summary.total_runs == matrix.metadata["expected_runs"]
    assert (root / "summary.csv").is_file()
    assert (root / "experiment_analysis.json").is_file()


def test_build_report_rebuilds_bundle_without_llm(tmp_path: Path) -> None:
    matrix_path = _write_tmp_mock_matrix(tmp_path)
    matrix = load_matrix(matrix_path)
    root = tmp_path / "mock_report_ready"
    E2EBenchmarkRunner().run(matrix_path, clean_output=True)

    from benchmark.experiments.e2e_runner import build_reports, run_analysis

    build_reports(run_analysis(root), matrix)

    assert (root / "report_bundle" / "bundle_validation_result.json").is_file()


def test_mock_report_ready_matrix_creates_valid_bundle(tmp_path: Path) -> None:
    matrix_path = _write_tmp_mock_matrix(tmp_path)
    root = tmp_path / "mock_report_ready"

    E2EBenchmarkRunner().run_and_report(matrix_path, clean_output=True)

    bundle = root / "report_bundle"
    result = validate_report_bundle_result(bundle)
    index = json.loads((bundle / "run_artifact_manifests.json").read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    matrix = load_matrix("configs/matrices/mock_report_ready.yaml")
    assert index["total_runs"] == matrix.metadata["expected_runs"]
    assert index["missing_required_artifacts"] == 0
