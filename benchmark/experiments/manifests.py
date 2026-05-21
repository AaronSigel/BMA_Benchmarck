from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.experiments.models import ExperimentMatrix, GeneratedExperimentManifest
from benchmark.experiments.readiness import check_matrix_readiness
from benchmark.experiments.preflight import tool_contract_hash

_SECRET_KEYS = {
    "api_key",
    "api_key_env",
    "authorization",
    "token",
    "secret",
    "password",
}


def build_manifest(
    matrix: ExperimentMatrix,
    *,
    metadata: dict[str, Any] | None = None,
) -> GeneratedExperimentManifest:
    payload = sanitized_config_payload(matrix)
    readiness = check_matrix_readiness(matrix)
    merged_metadata = {
        "readiness_ok": readiness.ok,
        "readiness_warnings": readiness.warnings,
        "readiness_errors": readiness.errors,
    }
    if metadata:
        merged_metadata.update(_sanitize(metadata))
    return GeneratedExperimentManifest(
        matrix_id=matrix.matrix_id,
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        git_commit=_git_commit(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        task_ids=matrix.tasks.ids,
        agent_ids=matrix.agents.ids,
        mcp_profiles=matrix.mcp_profiles,
        models=matrix.models.ids or matrix.models.providers,
        execution_modes=matrix.execution_modes,
        repetitions=matrix.repetitions,
        env_requirements=readiness.requirements,
        config_hash=stable_config_hash(payload),
        matrix_config_hash=stable_config_hash(payload),
        task_set_hash=_task_set_hash(matrix),
        tool_contract_hash=tool_contract_hash(),
        report_config_hash=_file_hash(matrix.report_config_path) if matrix.report_config_path else None,
        metadata=merged_metadata,
    )


def stable_config_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _task_set_hash(matrix: ExperimentMatrix) -> str:
    root = Path(matrix.metadata.get("tasks_root", "tasks"))
    selected = set(matrix.tasks.ids)
    payload: list[dict[str, str]] = []
    if root.exists():
        for path in sorted(root.glob("**/*")):
            if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8")
            if selected and not any(f"id: {task_id}" in text or f"id: \"{task_id}\"" in text or f"id: '{task_id}'" in text for task_id in selected):
                continue
            payload.append({"path": str(path), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    if not payload:
        payload = [{"task_id": task_id, "sha256": ""} for task_id in sorted(selected)]
    return stable_config_hash(payload)


def write_manifest(manifest: GeneratedExperimentManifest, path: Path | str) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def write_manifest_for_matrix(
    matrix: ExperimentMatrix,
    path: Path | str | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    manifest_path = Path(path) if path is not None else matrix.output_root / "manifest.json"
    write_manifest(build_manifest(matrix, metadata=metadata), manifest_path)
    return manifest_path


def sanitized_config_payload(matrix: ExperimentMatrix) -> dict[str, Any]:
    return _sanitize(matrix.model_dump(mode="json", exclude_none=True))


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            lowered = key.lower()
            if lowered in _SECRET_KEYS or any(secret in lowered for secret in _SECRET_KEYS):
                continue
            clean[key] = _sanitize(item)
        return clean
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value
