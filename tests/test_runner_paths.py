from pathlib import Path

from benchmark.runner.paths import RunArtifactLayout


def test_run_artifact_layout_paths_are_stable(tmp_path: Path) -> None:
    layout = RunArtifactLayout(root=tmp_path / "runs", run_id="geometry_001_replay")

    assert layout.run_dir() == tmp_path / "runs" / "geometry_001_replay"
    assert layout.scene_snapshot_json() == layout.run_dir() / "scene_snapshot.json"
    assert layout.validation_result_json() == layout.run_dir() / "validation_result.json"
    assert layout.run_result_json() == layout.run_dir() / "run_result.json"
    assert layout.metrics_json() == layout.run_dir() / "metrics.json"
    assert layout.logs_dir() == layout.run_dir() / "logs"


def test_run_artifact_layout_creates_directories(tmp_path: Path) -> None:
    layout = RunArtifactLayout(root=tmp_path / "runs", run_id="geometry_001_replay")

    layout.ensure()

    assert layout.run_dir().is_dir()
    assert layout.logs_dir().is_dir()


def test_run_artifact_layout_from_run_output_dir_accepts_existing_run_dir(
    tmp_path: Path,
) -> None:
    layout = RunArtifactLayout.from_run_output_dir(
        tmp_path / "runs" / "geometry_001_replay",
        "geometry_001_replay",
    )

    assert layout.root == tmp_path / "runs"
    assert layout.run_dir() == tmp_path / "runs" / "geometry_001_replay"


def test_run_artifact_layout_from_run_output_dir_accepts_root_dir(tmp_path: Path) -> None:
    layout = RunArtifactLayout.from_run_output_dir(tmp_path / "runs", "geometry_001_replay")

    assert layout.root == tmp_path / "runs"
    assert layout.run_dir() == tmp_path / "runs" / "geometry_001_replay"
