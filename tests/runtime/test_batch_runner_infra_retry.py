"""Tests for optional infra run retry in BatchRunner."""

from __future__ import annotations

from pathlib import Path

from benchmark.runner.batch_runner import (
    _annotate_infra_retry,
    _should_retry_run_on_infra_failure,
)
from benchmark.runner.models import ExecutionMode, RunConfig, RunResult, RunStatus


def _run_config() -> RunConfig:
    return RunConfig(
        run_id="run-1",
        task_id="task",
        execution_mode=ExecutionMode.AGENT_MCP,
        artifacts_dir=Path("/tmp/artifacts"),
        output_dir=Path("/tmp/output"),
        mcp_config_path=Path("/tmp/mcp.yaml"),
    )


def _run_result(**updates) -> RunResult:
    payload = {
        "run_id": "run-1",
        "task_id": "task",
        "status": RunStatus.ERROR,
        "execution_mode": ExecutionMode.AGENT_MCP,
        "validation_result_path": None,
        "scene_snapshot_path": None,
        "artifacts_dir": Path("/tmp/artifacts"),
        "overall_status": None,
        "started_at": "2026-05-22T10:00:00Z",
        "finished_at": "2026-05-22T10:00:12Z",
        "duration_sec": 12.0,
        "summary": {},
    }
    payload.update(updates)
    return RunResult(**payload)


def test_should_not_retry_when_disabled() -> None:
    result = _run_result(
        summary={"structured_error": {"is_infra_failure": True, "error_type": "ToolTimeout"}},
    )
    assert _should_retry_run_on_infra_failure(_run_config(), {}, result) is False


def test_should_retry_infra_failure_when_enabled() -> None:
    result = _run_result(
        summary={"structured_error": {"is_infra_failure": True, "error_type": "ToolTimeout"}},
    )
    lifecycle = {"retry_run_on_infra_failure": 1, "retry_run_max_duration_sec": 300}
    assert _should_retry_run_on_infra_failure(_run_config(), lifecycle, result) is True


def test_annotate_infra_retry_records_original_error() -> None:
    original = _run_result(
        summary={"structured_error": {"error_type": "EmptySocketResponse"}},
    )
    retry = _run_result(status=RunStatus.PASSED, summary={})
    annotated = _annotate_infra_retry(retry, original=original)
    assert annotated.summary["infra_retry"]["attempt"] == 2
    assert annotated.summary["infra_retry"]["original_error_type"] == "EmptySocketResponse"
