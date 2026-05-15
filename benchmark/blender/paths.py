from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path
    run_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def run_dir(self) -> Path:
        return self.root / "runs" / self.run_id

    def input_json(self) -> Path:
        return self.run_dir() / "input.json"

    def output_json(self, command: str) -> Path:
        return self.run_dir() / f"{command}.output.json"

    def snapshot_json(self) -> Path:
        return self.run_dir() / "scene_snapshot.json"

    def blend_file(self) -> Path:
        return self.run_dir() / "result.blend"

    def render_png(self) -> Path:
        return self.run_dir() / "render.png"

    def export_file(self, format: str) -> Path:
        suffix = format.lower().lstrip(".")
        return self.run_dir() / "exports" / f"result.{suffix}"

    def stdout_log(self, command: str) -> Path:
        return self.run_dir() / "logs" / f"{command}.stdout.log"

    def stderr_log(self, command: str) -> Path:
        return self.run_dir() / "logs" / f"{command}.stderr.log"

    def ensure(self) -> None:
        self.run_dir().mkdir(parents=True, exist_ok=True)
        (self.run_dir() / "exports").mkdir(parents=True, exist_ok=True)
        (self.run_dir() / "logs").mkdir(parents=True, exist_ok=True)

