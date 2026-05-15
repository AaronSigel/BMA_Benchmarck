from pathlib import Path

import benchmark.blender.cli as blender_cli
from benchmark.blender.errors import BlenderNotFoundError
from benchmark.blender.models import BlenderCommandResult


class FakeLauncher:
    calls = []

    def __init__(self) -> None:
        pass

    def run_module_command(self, command: str, payload: dict, output_dir: Path):
        self.calls.append((command, payload, output_dir))
        _create_payload_artifacts(payload)
        return BlenderCommandResult(
            ok=True,
            command=command,
            output_files=[str(output_dir / "output.json")],
            stdout="",
            stderr="",
            error=None,
            duration_sec=0.01,
        )

    def run_script(
        self,
        script_path: Path,
        input_json: Path | None = None,
        output_json: Path | None = None,
        timeout_sec: int | None = None,
        extra_args: list[str] | None = None,
    ):
        output_dir = Path(extra_args[extra_args.index("--output-dir") + 1])
        self.calls.append(("smoke_script", script_path, input_json, output_json, extra_args))
        for path in [
            output_dir / "scene_snapshot.json",
            output_dir / "result.blend",
            output_dir / "render.png",
            output_dir / "exports" / "result.glb",
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"artifact")
        if output_json is not None:
            output_json.write_bytes(b"{}")
        return BlenderCommandResult(
            ok=True,
            command="smoke",
            output_files=[str(output_json)] if output_json is not None else [],
            stdout="",
            stderr="",
            error=None,
            duration_sec=0.01,
        )


class MissingBlenderLauncher:
    def __init__(self) -> None:
        raise BlenderNotFoundError("Blender executable not found")


def _create_payload_artifacts(payload: dict) -> None:
    for key in ("save_path", "path", "output_path"):
        value = payload.get(key)
        if not value:
            continue

        path = Path(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")


def test_check_reports_found_blender(monkeypatch, capsys) -> None:
    monkeypatch.setattr(blender_cli, "find_blender_executable", lambda: "/usr/bin/blender")

    exit_code = blender_cli.main(["check"])

    assert exit_code == 0
    assert "/usr/bin/blender" in capsys.readouterr().out


def test_check_reports_missing_blender(monkeypatch, capsys) -> None:
    monkeypatch.setattr(blender_cli, "find_blender_executable", lambda: None)

    exit_code = blender_cli.main(["check"])

    assert exit_code == 1
    assert "Blender executable not found" in capsys.readouterr().out


def test_fixture_command_runs_create_fixture_scene(monkeypatch, tmp_path: Path) -> None:
    FakeLauncher.calls = []
    monkeypatch.setattr(blender_cli, "BlenderLauncher", FakeLauncher)
    output_dir = tmp_path / "blender_smoke"

    exit_code = blender_cli.main(["fixture", "--output-dir", str(output_dir)])

    assert exit_code == 0
    assert FakeLauncher.calls == [
        (
            "create_fixture_scene",
            {
                "scene_name": "BMA Fixture Scene",
                "save_path": str(output_dir / "result.blend"),
            },
            output_dir,
        )
    ]
    assert (output_dir / "result.blend").exists()


def test_snapshot_render_and_export_commands(monkeypatch, tmp_path: Path) -> None:
    FakeLauncher.calls = []
    monkeypatch.setattr(blender_cli, "BlenderLauncher", FakeLauncher)
    output_dir = tmp_path / "blender_smoke"

    assert blender_cli.main(["snapshot", "--output-dir", str(output_dir)]) == 0
    assert blender_cli.main(["render", "--output-dir", str(output_dir)]) == 0
    assert blender_cli.main(["export", "--format", "glb", "--output-dir", str(output_dir)]) == 0

    assert [call[0] for call in FakeLauncher.calls] == [
        "collect_snapshot",
        "render_scene",
        "export_scene",
    ]
    assert (output_dir / "scene_snapshot.json").exists()
    assert (output_dir / "render.png").exists()
    assert (output_dir / "exports" / "result.glb").exists()


def test_smoke_command_runs_all_commands_and_creates_expected_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    FakeLauncher.calls = []
    monkeypatch.setattr(blender_cli, "BlenderLauncher", FakeLauncher)
    output_dir = tmp_path / "blender_smoke"

    exit_code = blender_cli.main(["smoke", "--output-dir", str(output_dir)])

    assert exit_code == 0
    assert [call[0] for call in FakeLauncher.calls] == ["smoke_script"]
    assert (output_dir / "scene_snapshot.json").exists()
    assert (output_dir / "result.blend").exists()
    assert (output_dir / "render.png").exists()
    assert (output_dir / "exports" / "result.glb").exists()


def test_command_reports_missing_blender(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(blender_cli, "BlenderLauncher", MissingBlenderLauncher)

    exit_code = blender_cli.main(["snapshot", "--output-dir", str(tmp_path)])

    assert exit_code == 1
    assert "Blender executable not found" in capsys.readouterr().out
