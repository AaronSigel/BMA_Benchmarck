import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.blender.config import find_blender_executable
from benchmark.blender.models import SceneSnapshot


pytestmark = [
    pytest.mark.blender,
    pytest.mark.integration,
]


def require_blender() -> str:
    blender_bin = find_blender_executable()
    if blender_bin is None:
        pytest.skip("Blender executable not found")
    return blender_bin


def test_blender_smoke_creates_real_artifacts(tmp_path: Path) -> None:
    require_blender()
    output_dir = tmp_path / "blender_smoke"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmark.blender.cli",
            "smoke",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr

    snapshot_path = output_dir / "scene_snapshot.json"
    blend_path = output_dir / "result.blend"
    render_path = output_dir / "render.png"
    export_path = output_dir / "exports" / "result.glb"
    smoke_output_path = output_dir / "smoke_output.json"

    for path in [snapshot_path, blend_path, render_path, export_path, smoke_output_path]:
        assert path.exists(), f"missing artifact: {path}"
        assert path.stat().st_size > 0, f"empty artifact: {path}"

    snapshot = SceneSnapshot.model_validate_json(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot.objects
    assert snapshot.materials
    assert snapshot.lights
    assert snapshot.cameras

    smoke_output = json.loads(smoke_output_path.read_text(encoding="utf-8"))
    assert smoke_output["ok"] is True
    assert list(smoke_output["results"]) == [
        "reset_scene",
        "create_fixture_scene",
        "collect_snapshot",
        "save_scene",
        "render_scene",
        "export_scene",
    ]

