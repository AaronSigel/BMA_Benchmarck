from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactLayout:
    """Stable directory layout for a single agent run."""

    run_dir: Path

    @property
    def agent_config(self) -> Path:
        return self.run_dir / "agent_config.yaml"

    @property
    def task(self) -> Path:
        return self.run_dir / "task.yaml"

    @property
    def agent_trace(self) -> Path:
        return self.run_dir / "agent_trace.json"

    @property
    def tool_results(self) -> Path:
        return self.run_dir / "tool_results.json"

    @property
    def scene_snapshot(self) -> Path:
        return self.run_dir / "scene_snapshot.json"

    @property
    def run_result(self) -> Path:
        return self.run_dir / "run_result.json"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    def create_dirs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def build_artifact_layout(base_dir: Path | str, run_id: str) -> ArtifactLayout:
    """Return an ArtifactLayout rooted at <base_dir>/agent_runs/<run_id>/."""
    return ArtifactLayout(run_dir=Path(base_dir) / "agent_runs" / run_id)
