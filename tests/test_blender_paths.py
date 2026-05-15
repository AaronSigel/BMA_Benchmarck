from pathlib import Path

from benchmark.blender.paths import ArtifactLayout


def test_artifact_layout_paths_are_stable(tmp_path: Path) -> None:
    layout = ArtifactLayout(root=tmp_path / "artifacts", run_id="run-001")
    run_dir = tmp_path / "artifacts" / "runs" / "run-001"

    assert layout.run_dir() == run_dir
    assert layout.input_json() == run_dir / "input.json"
    assert layout.output_json("render_scene") == run_dir / "render_scene.output.json"
    assert layout.snapshot_json() == run_dir / "scene_snapshot.json"
    assert layout.blend_file() == run_dir / "result.blend"
    assert layout.render_png() == run_dir / "render.png"
    assert layout.export_file("glb") == run_dir / "exports" / "result.glb"
    assert layout.export_file(".fbx") == run_dir / "exports" / "result.fbx"
    assert layout.stdout_log("render_scene") == run_dir / "logs" / "render_scene.stdout.log"
    assert layout.stderr_log("render_scene") == run_dir / "logs" / "render_scene.stderr.log"


def test_artifact_layout_ensure_creates_directories(tmp_path: Path) -> None:
    layout = ArtifactLayout(root=tmp_path / "artifacts", run_id="run-001")

    layout.ensure()

    assert layout.run_dir().is_dir()
    assert (layout.run_dir() / "exports").is_dir()
    assert (layout.run_dir() / "logs").is_dir()


def test_artifact_layout_path_methods_do_not_create_files_or_directories(
    tmp_path: Path,
) -> None:
    layout = ArtifactLayout(root=tmp_path / "artifacts", run_id="run-001")

    paths = [
        layout.input_json(),
        layout.output_json("collect_snapshot"),
        layout.snapshot_json(),
        layout.blend_file(),
        layout.render_png(),
        layout.export_file("glb"),
        layout.stdout_log("collect_snapshot"),
        layout.stderr_log("collect_snapshot"),
    ]

    assert not layout.run_dir().exists()
    assert all(not path.exists() for path in paths)


def test_artifact_layout_accepts_string_root(tmp_path: Path) -> None:
    layout = ArtifactLayout(root=str(tmp_path / "artifacts"), run_id="run-001")

    assert layout.root == tmp_path / "artifacts"

