from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from benchmark.runner.models import RunResult
from benchmark.runner.models import ExecutionMode
from benchmark.runner.paths import RunArtifactLayout


class ArtifactEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    exists: bool
    required: bool
    not_available_reason: str | None = None


class RunArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    task_id: str
    model: str | None = None
    strategy: str | None = None
    mcp_profile: str | None = None
    status: str
    artifacts: dict[str, ArtifactEntry | list[dict[str, Any]]]
    files: list[str] = Field(default_factory=list)
    structured_error: dict[str, Any] | None = None


def build_run_artifact_manifest(
    result: RunResult,
    layout: RunArtifactLayout,
    *,
    structured_error: dict[str, Any] | None = None,
) -> RunArtifactManifest:
    run_dir = layout.run_dir()
    summary = result.summary if isinstance(result.summary, dict) else {}
    error_payload = structured_error or summary.get("structured_error")
    artifacts: dict[str, ArtifactEntry | list[dict[str, Any]]] = {
        "run_result": _entry(run_dir, "run_result.json", required=True),
        "agent_trace": _entry(run_dir, "agent_trace.json", required=result.execution_mode in {ExecutionMode.AGENT_MCP, ExecutionMode.REMOTE_AGENT}),
        "scene_snapshot": _entry(
            run_dir,
            "scene_snapshot.json",
            required=False,
            not_available_reason=_not_available_reason(run_dir, "scene_snapshot", result, error_payload),
        ),
        "scene_snapshot_not_available": _entry(run_dir, "scene_snapshot_not_available.json", required=False),
        "validation_result": _entry(
            run_dir,
            "validation_result.json",
            required=False,
            not_available_reason=_not_available_reason(run_dir, "validation_result", result, error_payload),
        ),
        "validation_result_not_available": _entry(run_dir, "validation_result_not_available.json", required=False),
        "metrics": _entry(run_dir, "metrics.json", required=True),
        "exports": _exports(run_dir),
        "exports_not_available": _entry(run_dir, "exports_not_available.json", required=False),
    }
    return RunArtifactManifest(
        run_id=result.run_id,
        task_id=result.task_id,
        model=_str_or_none(summary.get("model")),
        strategy=_str_or_none(summary.get("strategy")),
        mcp_profile=_str_or_none(summary.get("mcp_profile")),
        status=_manifest_status(result),
        artifacts=artifacts,
        files=_files(run_dir),
        structured_error=error_payload if isinstance(error_payload, dict) else None,
    )


def write_run_artifact_manifest(
    result: RunResult,
    layout: RunArtifactLayout,
    *,
    structured_error: dict[str, Any] | None = None,
) -> Path:
    manifest = build_run_artifact_manifest(result, layout, structured_error=structured_error)
    path = layout.artifact_manifest_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def validate_run_artifact_manifest(run_dir: Path | str) -> tuple[bool, list[str]]:
    run_path = Path(run_dir)
    manifest_path = run_path / "artifact_manifest.json"
    errors: list[str] = []
    if not manifest_path.exists():
        return False, ["artifact_manifest.json missing"]
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = RunArtifactManifest.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        return False, [f"artifact_manifest.json invalid: {exc}"]
    for name, entry in manifest.artifacts.items():
        if name == "exports" or isinstance(entry, list):
            continue
        if entry.required and not (run_path / entry.path).is_file():
            errors.append(f"required artifact missing: {entry.path}")
    return not errors, errors


def _entry(run_dir: Path, relative: str, *, required: bool, not_available_reason: str | None = None) -> ArtifactEntry:
    exists = (run_dir / relative).is_file()
    return ArtifactEntry(
        path=relative,
        exists=exists,
        required=required,
        not_available_reason=None if exists else not_available_reason,
    )


def _exports(run_dir: Path) -> list[dict[str, Any]]:
    exports_dir = run_dir / "exports"
    if not exports_dir.exists():
        return []
    return [
        {
            "path": str(path.relative_to(run_dir)),
            "exists": path.is_file(),
            "required": False,
            "size_bytes": path.stat().st_size if path.is_file() else 0,
        }
        for path in sorted(exports_dir.rglob("*"))
        if path.is_file()
    ]


def _files(run_dir: Path) -> list[str]:
    if not run_dir.exists():
        return []
    return sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file())


def _manifest_status(result: RunResult) -> str:
    summary = result.summary if isinstance(result.summary, dict) else {}
    validation = summary.get("validation")
    issue_counts = validation.get("issue_counts") if isinstance(validation, dict) else None
    run_status = (result.run_status or result.status).value
    scene_status = result.scene_status.value if result.scene_status else None
    if run_status == "error" or scene_status in {None, "not_available", "skipped"}:
        return "runtime_error"
    if run_status == "passed" and scene_status == "passed":
        return "soft_pass" if isinstance(issue_counts, dict) and issue_counts else "clean_pass"
    if scene_status == "failed":
        return "failed_validation"
    return "runtime_error"


def _missing_reason(result: RunResult, artifact: str, structured_error: object) -> str:
    if isinstance(structured_error, dict) and structured_error.get("message"):
        return str(structured_error["message"])
    if result.error:
        return result.error
    return f"{artifact} was not produced"


def _not_available_reason(run_dir: Path, artifact: str, result: RunResult, structured_error: object) -> str | None:
    artifact_path = run_dir / f"{artifact}.json"
    if artifact_path.exists():
        return None
    marker_path = run_dir / f"{artifact}_not_available.json"
    if marker_path.exists():
        try:
            data = json.loads(marker_path.read_text(encoding="utf-8"))
            reason = data.get("reason") or data.get("message")
            if reason:
                return str(reason)
        except Exception:  # noqa: BLE001
            pass
    return _missing_reason(result, artifact, structured_error)


def _str_or_none(value: object) -> str | None:
    return str(value) if value is not None else None
