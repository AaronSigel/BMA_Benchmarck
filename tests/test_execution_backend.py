import shutil
from pathlib import Path

from benchmark.blender.models import BlenderCommandResult
from benchmark.runner.execution import (
    BlenderSmokeBackend,
    ExternalSnapshotBackend,
    ReplayBackend,
)
from benchmark.runner.models import ExecutionMode, RunConfig


def make_run_config(tmp_path: Path, **overrides: object) -> RunConfig:
    data = {
        "run_id": "geometry_001_replay",
        "task_id": "geometry_001_basic_primitives",
        "execution_mode": ExecutionMode.EXTERNAL_SNAPSHOT,
        "snapshot_path": Path("artifacts/blender_smoke/scene_snapshot.json"),
        "artifacts_dir": Path("artifacts/blender_smoke"),
        "output_dir": tmp_path / "run",
    }
    data.update(overrides)
    return RunConfig(**data)


def copy_snapshot(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2("artifacts/blender_smoke/scene_snapshot.json", destination)
    return destination


def test_external_snapshot_backend_validates_existing_snapshot(tmp_path: Path) -> None:
    snapshot_path = copy_snapshot(tmp_path / "scene_snapshot.json")
    config = make_run_config(tmp_path, snapshot_path=snapshot_path, artifacts_dir=tmp_path)

    result = ExternalSnapshotBackend().execute(config)

    assert result.ok is True
    assert result.scene_snapshot_path == snapshot_path
    assert result.artifacts_dir == tmp_path
    assert result.error is None


def test_external_snapshot_backend_reports_missing_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "missing.json"
    config = make_run_config(tmp_path, snapshot_path=snapshot_path, artifacts_dir=tmp_path)

    result = ExternalSnapshotBackend().execute(config)

    assert result.ok is False
    assert result.scene_snapshot_path == snapshot_path
    assert "does not exist" in str(result.error)


def test_external_snapshot_backend_reports_invalid_snapshot(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "scene_snapshot.json"
    snapshot_path.write_text('{"scene_name": "incomplete"}', encoding="utf-8")
    config = make_run_config(tmp_path, snapshot_path=snapshot_path, artifacts_dir=tmp_path)

    result = ExternalSnapshotBackend().execute(config)

    assert result.ok is False
    assert result.scene_snapshot_path == snapshot_path
    assert "Invalid SceneSnapshot" in str(result.error)


def test_replay_backend_copies_snapshot_and_artifacts(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    copy_snapshot(source_dir / "scene_snapshot.json")
    (source_dir / "exports").mkdir()
    (source_dir / "exports" / "result.glb").write_text("glb", encoding="utf-8")
    output_dir = tmp_path / "output"
    config = make_run_config(
        tmp_path,
        execution_mode=ExecutionMode.REPLAY,
        artifacts_dir=source_dir,
        output_dir=output_dir,
        snapshot_path=None,
    )

    result = ReplayBackend().execute(config)

    assert result.ok is True
    assert result.scene_snapshot_path == output_dir / "scene_snapshot.json"
    assert (output_dir / "scene_snapshot.json").is_file()
    assert (output_dir / "exports" / "result.glb").read_text(encoding="utf-8") == "glb"


def test_blender_smoke_backend_reports_missing_blender(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "benchmark.runner.execution.blender_config.find_blender_executable",
        lambda: None,
    )
    config = make_run_config(
        tmp_path,
        execution_mode=ExecutionMode.BLENDER_SMOKE,
        output_dir=tmp_path / "smoke",
    )

    result = BlenderSmokeBackend().execute(config)

    assert result.ok is False
    assert "Blender executable not found" in str(result.error)


def test_blender_smoke_backend_runs_launcher_with_monkeypatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []

    class FakeLauncher:
        def __init__(self, config) -> None:
            self.config = config

        def run_script(self, script_path, input_json, output_json, extra_args):
            calls.append(
                {
                    "script_path": script_path,
                    "input_json": input_json,
                    "output_json": output_json,
                    "extra_args": extra_args,
                    "blender_bin": self.config.blender_bin,
                }
            )
            copy_snapshot(Path(extra_args[-1]) / "scene_snapshot.json")
            Path(output_json).write_text('{"ok": true}', encoding="utf-8")
            return BlenderCommandResult(
                ok=True,
                command="fake blender smoke",
                output_files=[str(output_json)],
                stdout="",
                stderr="",
                error=None,
                duration_sec=0.1,
            )

    monkeypatch.setattr(
        "benchmark.runner.execution.blender_config.find_blender_executable",
        lambda: "/usr/bin/blender",
    )
    monkeypatch.setattr(
        "benchmark.runner.execution.blender_launcher.BlenderLauncher",
        FakeLauncher,
    )
    output_dir = tmp_path / "smoke"
    config = make_run_config(
        tmp_path,
        execution_mode=ExecutionMode.BLENDER_SMOKE,
        output_dir=output_dir,
    )

    result = BlenderSmokeBackend().execute(config)

    assert result.ok is True
    assert result.scene_snapshot_path == output_dir / "scene_snapshot.json"
    assert result.metadata["command"] == "fake blender smoke"
    assert calls[0]["blender_bin"] == "/usr/bin/blender"
    assert calls[0]["extra_args"] == ["--output-dir", str(output_dir)]
