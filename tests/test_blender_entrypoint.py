import json
import subprocess
import sys
from pathlib import Path


ENTRYPOINT = Path("benchmark/blender/scripts/blender_entrypoint.py")


def run_entrypoint(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_entrypoint_is_not_imported_by_regular_tests() -> None:
    assert "benchmark.blender.scripts.blender_entrypoint" not in sys.modules


def test_entrypoint_writes_json_error_for_unknown_command(tmp_path: Path) -> None:
    output_json = tmp_path / "result.json"

    completed = run_entrypoint("--command", "unknown", "--output", str(output_json))

    assert completed.returncode == 1
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["ok"] is False
    assert data["command"] == "unknown"
    assert "unsupported command" in data["error"]


def test_entrypoint_parses_arguments_after_double_dash(tmp_path: Path) -> None:
    output_json = tmp_path / "result.json"

    completed = run_entrypoint(
        "--background",
        "--python",
        str(ENTRYPOINT),
        "--",
        "--command",
        "unknown",
        "--output",
        str(output_json),
    )

    assert completed.returncode == 1
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["command"] == "unknown"


def test_entrypoint_reads_command_and_payload_from_input_json(tmp_path: Path) -> None:
    input_json = tmp_path / "input.json"
    output_json = tmp_path / "result.json"
    input_json.write_text(
        json.dumps({"command": "reset_scene", "payload": {"scene_name": "Smoke"}}),
        encoding="utf-8",
    )

    completed = run_entrypoint("--input", str(input_json), "--output", str(output_json))

    assert completed.returncode == 1
    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["ok"] is False
    assert data["command"] == "reset_scene"
    assert "bpy" in data["error"]


def test_entrypoint_uses_output_dir_when_output_is_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "entrypoint"

    completed = run_entrypoint(
        "--command",
        "unknown",
        "--output-dir",
        str(output_dir),
    )

    assert completed.returncode == 1
    assert (output_dir / "output.json").exists()


def test_entrypoint_registers_all_stage_2_commands() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")

    for command in [
        "reset_scene",
        "create_fixture_scene",
        "collect_snapshot",
        "save_scene",
        "render_scene",
        "export_scene",
    ]:
        assert f'"{command}"' in source
