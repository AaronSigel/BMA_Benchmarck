from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.experiments.models import ExperimentMatrix

from bma_benchmark.reporting.evidence_pack.sanity import SanitySuiteResult
from bma_benchmark.reporting.scene_examples.models import SceneExample, SceneExampleBundle


def write_run_manifest(
    path: Path,
    *,
    experiment_dir: Path,
    config_path: Path | None,
    matrix: ExperimentMatrix | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "matrix_config": str(config_path) if config_path else None,
        "matrix_id": matrix.matrix_id if matrix else None,
        "experiment_dir": str(experiment_dir),
        "expected_runs": matrix.metadata.get("expected_runs") if matrix else None,
        "tasks": list(matrix.tasks.ids) if matrix else [],
        "models": list(matrix.models.ids) if matrix else [],
        "mcp_profiles": list(matrix.mcp_profiles) if matrix else [],
        "repetitions": matrix.repetitions if matrix else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
