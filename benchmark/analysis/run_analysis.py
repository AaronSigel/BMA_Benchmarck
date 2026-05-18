from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.agent.models import AgentTrace
from benchmark.analysis.agent_metrics import compute_agent_summary, extract_agent_metrics
from benchmark.analysis.error_taxonomy import aggregate_errors, extract_errors
from benchmark.analysis.models import RunAnalysisResult
from benchmark.analysis.tool_metrics import compute_tool_summary
from benchmark.analysis.trace_reader import RunArtifactBundle
from benchmark.analysis.validation_metrics import (
    compute_validation_summary,
    extract_issues,
)
from benchmark.runner.models import RunResult, RunStatus
from benchmark.validation.models import SceneValidationResult, ValidationStatus


# ---------------------------------------------------------------------------
# Success determination
# ---------------------------------------------------------------------------


def _determine_success(
    run_result: RunResult | None,
    validation_result: SceneValidationResult | None,
    trace: AgentTrace | None,
) -> bool | None:
    """Combine run status and validation status into a single success flag."""
    # When run_result is available its status is authoritative for execution success
    if run_result is not None:
        exec_ok = run_result.status == RunStatus.PASSED
        if validation_result is not None:
            val_ok = validation_result.overall_status in (
                ValidationStatus.PASSED,
                ValidationStatus.WARNING,
            )
            return exec_ok and val_ok
        return exec_ok

    # Fall back to trace.success when no RunResult
    if trace is not None:
        return trace.success

    return None


# ---------------------------------------------------------------------------
# Artifact path collection
# ---------------------------------------------------------------------------


def _collect_artifacts(bundle: RunArtifactBundle) -> list[str]:
    known_files = (
        "agent_trace.json",
        "run_result.json",
        "validation_result.json",
        "scene_snapshot.json",
        "metrics.json",
        "summary.json",
    )
    paths: list[str] = []
    for name in known_files:
        p = bundle.run_dir / name
        if p.exists():
            paths.append(str(p))
    for trace_path in sorted(bundle.run_dir.glob("agent_runs/*/agent_trace.json")):
        paths.append(str(trace_path))
    for export_path in sorted((bundle.run_dir / "exports").glob("*")):
        if export_path.is_file():
            paths.append(str(export_path))
    return paths


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _mcp_profile_from_bundle(bundle: RunArtifactBundle) -> str | None:
    """Try to extract mcp_profile from run_result.summary or bundle.summary."""
    if bundle.run_result is not None:
        profile = bundle.run_result.summary.get("mcp_profile")
        if profile:
            return str(profile)
        execution = bundle.run_result.summary.get("execution")
        if isinstance(execution, dict):
            profile = execution.get("mcp_profile")
            if profile:
                return str(profile)
    if bundle.summary is not None:
        profile = bundle.summary.get("mcp_profile")
        if profile:
            return str(profile)
        execution = bundle.summary.get("execution")
        if isinstance(execution, dict):
            profile = execution.get("mcp_profile")
            if profile:
                return str(profile)
    if bundle.run_result is not None:
        parts = bundle.run_result.run_id.split("__")
        for candidate in ("minimal", "no_python", "inspection_enabled", "python_enabled", "full"):
            if candidate in parts:
                return candidate
    return None


def _agent_id_from_bundle(bundle: RunArtifactBundle) -> str:
    if bundle.agent_trace is not None:
        return bundle.agent_trace.agent_id
    if bundle.run_result is not None:
        return bundle.run_result.summary.get("agent_id", "unknown")
    return "unknown"


def _strategy_from_bundle(bundle: RunArtifactBundle) -> str:
    if bundle.agent_trace is not None:
        return bundle.agent_trace.strategy.value
    if bundle.run_result is not None:
        return bundle.run_result.summary.get("strategy", "unknown")
    return "unknown"


def _model_from_bundle(bundle: RunArtifactBundle) -> str | None:
    if bundle.agent_trace is not None:
        return bundle.agent_trace.model
    if bundle.run_result is not None:
        return bundle.run_result.summary.get("model")
    return None


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


def analyze_run(bundle: RunArtifactBundle) -> RunAnalysisResult:
    """Build a complete RunAnalysisResult from a RunArtifactBundle.

    Tolerates missing agent_trace and missing validation_result — neither
    causes an exception.
    """
    run_result = bundle.run_result
    trace = bundle.agent_trace
    validation = bundle.validation_result

    # Resolve identifiers (prefer run_result as the canonical source)
    if run_result is not None:
        run_id = run_result.run_id
        task_id = run_result.task_id
    elif trace is not None:
        run_id = trace.run_id
        task_id = trace.task_id
    else:
        run_id = bundle.run_dir.name
        task_id = "unknown"

    mcp_profile = _mcp_profile_from_bundle(bundle)
    agent_id = _agent_id_from_bundle(bundle)
    strategy = _strategy_from_bundle(bundle)
    model = _model_from_bundle(bundle)

    # Duration: prefer run_result (wall-clock), fall back to trace
    duration_sec: float | None = None
    if run_result is not None:
        duration_sec = run_result.duration_sec
    if duration_sec is None and trace is not None:
        duration_sec = trace.duration_sec

    # -----------------------------------------------------------------------
    # Agent / tool metrics — gracefully degrade when trace is absent
    # -----------------------------------------------------------------------
    metrics: dict[str, float | str | int | bool] = {}

    if trace is not None:
        agent_summary = compute_agent_summary(trace)
        tool_summary = compute_tool_summary(trace)

        metrics.update({
            # Agent metrics
            "llm_call_count": agent_summary.llm_call_count,
            "planning_step_count": agent_summary.planning_step_count,
            "observation_count": agent_summary.observation_count,
            "final_step_present": agent_summary.final_step_present,
            "retry_count": agent_summary.retry_count,
            "step_limit_reached": agent_summary.step_limit_reached,
            "self_correction_attempts": agent_summary.self_correction_attempts,
            "tool_error_recovery_count": agent_summary.tool_error_recovery_count,
            # Tool metrics
            "tool_call_count": tool_summary.tool_call_count,
            "unique_tool_count": tool_summary.unique_tool_count,
            "invalid_tool_call_count": tool_summary.invalid_tool_call_count,
            "disabled_tool_call_count": tool_summary.disabled_tool_call_count,
            "tool_error_count": tool_summary.tool_error_count,
            "inspection_tool_count": tool_summary.inspection_tool_count,
            "mutation_tool_count": tool_summary.mutation_tool_count,
            "python_tool_call_count": tool_summary.python_tool_call_count,
            "asset_tool_call_count": tool_summary.asset_tool_call_count,
            "tool_repetition_count": tool_summary.tool_repetition_count,
        })
        for key, val in [
            ("average_step_duration_sec", agent_summary.average_step_duration_sec),
            ("max_step_duration_sec", agent_summary.max_step_duration_sec),
            ("average_tool_duration_sec", tool_summary.average_tool_duration_sec),
            ("prompt_tokens", agent_summary.prompt_tokens),
            ("completion_tokens", agent_summary.completion_tokens),
            ("total_tokens", agent_summary.total_tokens),
            ("estimated_cost", agent_summary.estimated_cost),
        ]:
            if val is not None:
                metrics[key] = val

        tool_call_count = tool_summary.tool_call_count
        invalid_tool_call_count = tool_summary.invalid_tool_call_count
        trajectory_length = tool_summary.trajectory_length
        llm_call_count = agent_summary.llm_call_count
        retry_count = agent_summary.retry_count
        error_count = agent_summary.error_count
    else:
        tool_call_count = 0
        invalid_tool_call_count = 0
        trajectory_length = 0
        llm_call_count = 0
        retry_count = 0
        error_count = 0

    # -----------------------------------------------------------------------
    # Validation metrics
    # -----------------------------------------------------------------------
    val_summary = compute_validation_summary(validation)

    metrics["scene_overall_status"] = val_summary.scene_overall_status
    metrics["passed_validator_count"] = val_summary.passed_validator_count
    metrics["failed_validator_count"] = val_summary.failed_validator_count
    metrics["skipped_validator_count"] = val_summary.skipped_validator_count
    metrics["validation_error_count"] = val_summary.validation_error_count
    metrics["validation_warning_count"] = val_summary.validation_warning_count

    for field in (
        "object_score", "transform_score", "material_score",
        "light_score", "camera_score", "export_score",
    ):
        v = getattr(val_summary, field)
        if v is not None:
            metrics[field] = v

    # -----------------------------------------------------------------------
    # Error taxonomy
    # -----------------------------------------------------------------------
    error_counts = aggregate_errors(bundle)
    for cat, count in error_counts.items():
        metrics[f"error.{cat}"] = count

    issues: list[dict[str, Any]] = []
    if validation is not None:
        issues = extract_issues(validation)
        # Add validation errors to total error_count
        error_count += sum(error_counts.get(k, 0) for k in error_counts
                           if k.startswith("scene_"))

    # -----------------------------------------------------------------------
    # RunResult metadata → metrics
    # -----------------------------------------------------------------------
    if run_result is not None:
        if run_result.total_score is not None:
            metrics["run_result_total_score"] = run_result.total_score
        if run_result.execution_mode:
            metrics["execution_mode"] = run_result.execution_mode.value
        for key, val in (run_result.summary or {}).items():
            if isinstance(val, (float, str, int, bool)):
                metrics.setdefault(f"run_summary.{key}", val)

    # -----------------------------------------------------------------------
    # Artifact paths
    # -----------------------------------------------------------------------
    artifacts = _collect_artifacts(bundle)

    # -----------------------------------------------------------------------
    # Success determination
    # -----------------------------------------------------------------------
    success = _determine_success(run_result, validation, trace)

    # Total score: prefer validation result, fall back to run_result
    total_score: float | None = val_summary.scene_total_score
    if total_score is None and run_result is not None:
        total_score = run_result.total_score

    return RunAnalysisResult(
        run_id=run_id,
        task_id=task_id,
        agent_id=agent_id,
        strategy=strategy,
        model=model,
        mcp_profile=mcp_profile,
        total_score=total_score,
        validation_status=val_summary.scene_overall_status,
        tool_call_count=tool_call_count,
        invalid_tool_call_count=invalid_tool_call_count,
        trajectory_length=trajectory_length,
        retry_count=retry_count,
        duration_sec=duration_sec,
        llm_call_count=llm_call_count,
        error_count=error_count,
        success=success,
        metrics=metrics,
        issues=issues,
        artifacts=artifacts,
    )
