from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from benchmark.agent.models import AgentTrace
from benchmark.runner.models import ExperimentResult, RunResult
from benchmark.validation.models import SceneValidationResult

logger = logging.getLogger(__name__)

_AGENT_TRACE_FILENAME = "agent_trace.json"
_RUN_RESULT_FILENAME = "run_result.json"
_VALIDATION_RESULT_FILENAME = "validation_result.json"
_SCENE_SNAPSHOT_FILENAME = "scene_snapshot.json"
_METRICS_FILENAME = "metrics.json"
_SUMMARY_FILENAME = "summary.json"


class TraceReadError(Exception):
    pass


# ---------------------------------------------------------------------------
# RunArtifactBundle
# ---------------------------------------------------------------------------


class RunArtifactBundle(BaseModel):
    """All optional artifacts for one benchmark run directory."""

    run_dir: Path
    agent_trace: AgentTrace | None = None
    run_result: RunResult | None = None
    validation_result: SceneValidationResult | None = None
    scene_snapshot: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Low-level readers — each raises TraceReadError on hard failures
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TraceReadError(f"Cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TraceReadError(f"Invalid JSON in {path}: {exc}") from exc


def read_agent_trace(path: Path | str) -> AgentTrace:
    p = Path(path)
    try:
        return AgentTrace.model_validate_json(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TraceReadError(f"Cannot read agent trace {p}: {exc}") from exc
    except (ValueError, KeyError) as exc:
        raise TraceReadError(f"Invalid agent trace in {p}: {exc}") from exc


def read_run_result(path: Path | str) -> RunResult:
    p = Path(path)
    data = _read_json(p)
    try:
        return RunResult.model_validate(data)
    except (ValueError, KeyError) as exc:
        raise TraceReadError(f"Invalid run_result in {p}: {exc}") from exc


def read_validation_result(path: Path | str) -> SceneValidationResult:
    p = Path(path)
    data = _read_json(p)
    try:
        return SceneValidationResult.model_validate(data)
    except (ValueError, KeyError) as exc:
        raise TraceReadError(f"Invalid validation_result in {p}: {exc}") from exc


def read_experiment_result(path: Path | str) -> ExperimentResult:
    p = Path(path)
    data = _read_json(p)
    try:
        return ExperimentResult.model_validate(data)
    except (ValueError, KeyError) as exc:
        raise TraceReadError(f"Invalid experiment_result in {p}: {exc}") from exc


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def discover_run_artifacts(root: Path | str) -> list[Path]:
    """Return all run directories under *root* that contain at least one known artifact."""
    root_path = Path(root)
    known = {
        _AGENT_TRACE_FILENAME,
        _RUN_RESULT_FILENAME,
        _VALIDATION_RESULT_FILENAME,
    }
    found: list[Path] = []
    for candidate in sorted(root_path.rglob(_AGENT_TRACE_FILENAME)):
        found.append(candidate.parent)
    for candidate in sorted(root_path.rglob(_RUN_RESULT_FILENAME)):
        p = candidate.parent
        if p not in found:
            found.append(p)
    return sorted(set(found))


# ---------------------------------------------------------------------------
# Bundle loader
# ---------------------------------------------------------------------------


def load_run_bundle(run_dir: Path | str) -> RunArtifactBundle:
    """Load all available artifacts from *run_dir* into a RunArtifactBundle.

    Missing files are tolerated; absence of validation_result triggers a warning.
    """
    d = Path(run_dir)

    # agent_trace — optional, no crash if missing
    agent_trace: AgentTrace | None = None
    trace_path = d / _AGENT_TRACE_FILENAME
    if trace_path.exists():
        try:
            agent_trace = read_agent_trace(trace_path)
        except TraceReadError as exc:
            logger.warning("Could not load agent_trace from %s: %s", trace_path, exc)
    else:
        logger.debug("agent_trace not found in %s", d)

    # run_result — optional
    run_result: RunResult | None = None
    rr_path = d / _RUN_RESULT_FILENAME
    if rr_path.exists():
        try:
            run_result = read_run_result(rr_path)
        except TraceReadError as exc:
            logger.warning("Could not load run_result from %s: %s", rr_path, exc)

    # validation_result — optional but emit warning when absent
    validation: SceneValidationResult | None = None
    val_path = d / _VALIDATION_RESULT_FILENAME
    if val_path.exists():
        try:
            validation = read_validation_result(val_path)
        except TraceReadError as exc:
            logger.warning("Could not load validation_result from %s: %s", val_path, exc)
    else:
        warnings.warn(
            f"validation_result.json not found in {d}; analysis will be incomplete.",
            stacklevel=2,
        )
        logger.warning("validation_result not found in %s", d)

    # scene_snapshot — optional raw JSON
    snapshot: dict[str, Any] | None = None
    snap_path = d / _SCENE_SNAPSHOT_FILENAME
    if snap_path.exists():
        try:
            snapshot = _read_json(snap_path)
        except TraceReadError as exc:
            logger.warning("Could not load scene_snapshot from %s: %s", snap_path, exc)

    # metrics — optional raw JSON. Runner writes a list of named metric rows;
    # analysis consumes a mapping, so normalize rows to {name: value}.
    metrics: dict[str, Any] | None = None
    met_path = d / _METRICS_FILENAME
    if met_path.exists():
        try:
            metrics = _normalize_metrics_json(_read_json(met_path))
        except TraceReadError as exc:
            logger.warning("Could not load metrics from %s: %s", met_path, exc)

    # summary — optional raw JSON
    summary: dict[str, Any] | None = None
    sum_path = d / _SUMMARY_FILENAME
    if sum_path.exists():
        try:
            summary = _read_json(sum_path)
        except TraceReadError as exc:
            logger.warning("Could not load summary from %s: %s", sum_path, exc)

    return RunArtifactBundle(
        run_dir=d,
        agent_trace=agent_trace,
        run_result=run_result,
        validation_result=validation,
        scene_snapshot=snapshot,
        metrics=metrics,
        summary=summary,
    )


def _normalize_metrics_json(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        normalized: dict[str, Any] = {}
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                normalized[item["name"]] = item.get("value")
        return normalized
    return {}


# ---------------------------------------------------------------------------
# Backward-compatible aliases used by other analysis modules
# ---------------------------------------------------------------------------

read_trace = read_agent_trace
