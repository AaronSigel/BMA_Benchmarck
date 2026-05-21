from pathlib import Path


class RunArtifactLayout:
    def __init__(self, root: Path, run_id: str) -> None:
        self.root = Path(root)
        self.run_id = run_id

    @classmethod
    def from_run_output_dir(cls, output_dir: Path, run_id: str) -> "RunArtifactLayout":
        output_dir = Path(output_dir)
        if output_dir.name == run_id:
            return cls(root=output_dir.parent, run_id=run_id)
        return cls(root=output_dir, run_id=run_id)

    def run_dir(self) -> Path:
        return self.root / self.run_id

    def validation_result_json(self) -> Path:
        return self.run_dir() / "validation_result.json"

    def run_result_json(self) -> Path:
        return self.run_dir() / "run_result.json"

    def scene_snapshot_json(self) -> Path:
        return self.run_dir() / "scene_snapshot.json"

    def metrics_json(self) -> Path:
        return self.run_dir() / "metrics.json"

    def artifact_manifest_json(self) -> Path:
        return self.run_dir() / "artifact_manifest.json"

    def exports_dir(self) -> Path:
        return self.run_dir() / "exports"

    def result_glb(self) -> Path:
        return self.exports_dir() / "result.glb"

    def result_blend(self) -> Path:
        return self.exports_dir() / "result.blend"

    def logs_dir(self) -> Path:
        return self.run_dir() / "logs"

    def ensure(self) -> None:
        self.run_dir().mkdir(parents=True, exist_ok=True)
        self.logs_dir().mkdir(parents=True, exist_ok=True)
        self.exports_dir().mkdir(parents=True, exist_ok=True)
