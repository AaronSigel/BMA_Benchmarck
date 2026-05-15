import json
import subprocess
from pathlib import Path

import pytest

from benchmark.blender.config import BlenderConfig
from benchmark.blender.errors import BlenderProcessError, BlenderTimeoutError
import benchmark.blender.launcher as launcher_module
from benchmark.blender.launcher import BlenderLauncher


def test_run_script_builds_expected_blender_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []

    def fake_run(command, capture_output, text, timeout, check):
        calls.append(
            {
                "command": command,
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
                "check": check,
            }
        )
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    launcher = BlenderLauncher(
        BlenderConfig(blender_bin="/usr/bin/blender", default_timeout_sec=42)
    )
    script = tmp_path / "script.py"
    input_json = tmp_path / "input.json"
    output_json = tmp_path / "nested" / "output.json"

    result = launcher.run_script(
        script_path=script,
        input_json=input_json,
        output_json=output_json,
        extra_args=["--mode", "smoke"],
    )

    assert calls == [
        {
            "command": [
                "/usr/bin/blender",
                "--background",
                "--python",
                str(script),
                "--",
                "--input",
                str(input_json),
                "--output",
                str(output_json),
                "--mode",
                "smoke",
            ],
            "capture_output": True,
            "text": True,
            "timeout": 42,
            "check": False,
        }
    ]
    assert result.ok is True
    assert result.stdout == "done"
    assert result.stderr == ""
    assert result.output_files == [str(output_json)]
    assert output_json.parent.exists()


def test_run_script_respects_non_headless_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_command = None

    def fake_run(command, capture_output, text, timeout, check):
        nonlocal captured_command
        captured_command = command
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    launcher = BlenderLauncher(BlenderConfig(blender_bin="blender", headless=False))

    launcher.run_script(script_path=tmp_path / "script.py")

    assert captured_command == ["blender", "--python", str(tmp_path / "script.py"), "--"]


def test_run_script_timeout_becomes_blender_timeout_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(command, capture_output, text, timeout, check):
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr("subprocess.run", fake_run)
    launcher = BlenderLauncher(BlenderConfig(blender_bin="blender"))

    with pytest.raises(BlenderTimeoutError):
        launcher.run_script(script_path=tmp_path / "script.py", timeout_sec=1)


def test_run_script_nonzero_exit_becomes_blender_process_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(command, capture_output, text, timeout, check):
        return subprocess.CompletedProcess(command, 7, stdout="out", stderr="err")

    monkeypatch.setattr("subprocess.run", fake_run)
    launcher = BlenderLauncher(BlenderConfig(blender_bin="blender"))

    with pytest.raises(BlenderProcessError) as exc_info:
        launcher.run_script(script_path=tmp_path / "script.py")

    message = str(exc_info.value)
    assert "Blender exited with code 7" in message
    assert "stdout" in message
    assert "out" in message
    assert "stderr" in message
    assert "err" in message


def test_run_module_command_writes_payload_and_creates_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_command = None

    def fake_run(command, capture_output, text, timeout, check):
        nonlocal captured_command
        captured_command = command
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    launcher = BlenderLauncher(BlenderConfig(blender_bin="blender"))
    output_dir = tmp_path / "artifacts" / "collect"

    result = launcher.run_module_command(
        command="collect_snapshot",
        payload={"scene": "Scene"},
        output_dir=output_dir,
        timeout_sec=3,
    )

    assert json.loads((output_dir / "input.json").read_text(encoding="utf-8")) == {
        "command": "collect_snapshot",
        "payload": {"scene": "Scene"},
    }
    assert result.output_files == [str(output_dir / "output.json")]
    assert captured_command is not None
    assert captured_command[:5] == [
        "blender",
        "--background",
        "--python",
        str(Path(launcher_module.__file__).parent / "scripts" / "blender_entrypoint.py"),
        "--",
    ]
    assert "--command" in captured_command
    assert "collect_snapshot" in captured_command
    assert "--output-dir" in captured_command
    assert str(output_dir) in captured_command


def test_save_current_scene_runs_save_scene_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []
    launcher = BlenderLauncher(BlenderConfig(blender_bin="blender"))
    output_path = tmp_path / "scene.blend"

    def fake_run_module_command(command, payload, output_dir, timeout_sec=None):
        calls.append(
            {
                "command": command,
                "payload": payload,
                "output_dir": output_dir,
                "timeout_sec": timeout_sec,
            }
        )
        return "result"

    monkeypatch.setattr(launcher, "run_module_command", fake_run_module_command)

    result = launcher.save_current_scene(output_path)

    assert result == "result"
    assert calls == [
        {
            "command": "save_scene",
            "payload": {"path": str(output_path)},
            "output_dir": output_path.parent,
            "timeout_sec": None,
        }
    ]
