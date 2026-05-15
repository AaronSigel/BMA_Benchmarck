import json
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmark.blender.config import BlenderConfig, get_blender_config
from benchmark.blender.errors import BlenderProcessError, BlenderTimeoutError
from benchmark.blender.models import BlenderCommandResult


def _tail(value: str | None, limit: int = 4000) -> str:
    if not value:
        return ""
    return value[-limit:]


class BlenderLauncher:
    def __init__(self, config: BlenderConfig | None = None) -> None:
        self.config = config or get_blender_config()

    def run_script(
        self,
        script_path: Path,
        input_json: Path | None = None,
        output_json: Path | None = None,
        timeout_sec: int | None = None,
        extra_args: list[str] | None = None,
    ) -> BlenderCommandResult:
        script_path = Path(script_path)
        if input_json is not None:
            input_json = Path(input_json)
        if output_json is not None:
            output_json = Path(output_json)
            output_json.parent.mkdir(parents=True, exist_ok=True)

        command = self._build_script_command(
            script_path=script_path,
            input_json=input_json,
            output_json=output_json,
            extra_args=extra_args,
        )
        effective_timeout = timeout_sec or self.config.default_timeout_sec

        started_at = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration_sec = time.perf_counter() - started_at
            raise BlenderTimeoutError(
                f"Blender command timed out after {effective_timeout} seconds "
                f"(duration: {duration_sec:.3f}s): {' '.join(command)}"
            ) from exc

        duration_sec = time.perf_counter() - started_at
        error = None
        if completed.returncode != 0:
            error = self._format_process_error(completed, output_json)

        result = BlenderCommandResult(
            ok=completed.returncode == 0,
            command=" ".join(command),
            output_files=[str(output_json)] if output_json is not None else [],
            stdout=completed.stdout,
            stderr=completed.stderr,
            error=error,
            duration_sec=duration_sec,
        )

        if completed.returncode != 0:
            raise BlenderProcessError(result.error)

        return result

    def run_module_command(
        self,
        command: str,
        payload: dict[str, Any],
        output_dir: Path,
        timeout_sec: int | None = None,
    ) -> BlenderCommandResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_json = output_dir / "input.json"
        output_json = output_dir / "output.json"
        input_json.write_text(
            json.dumps({"command": command, "payload": payload}, indent=2),
            encoding="utf-8",
        )

        entrypoint = Path(__file__).parent / "scripts" / "blender_entrypoint.py"
        return self.run_script(
            script_path=entrypoint,
            input_json=input_json,
            output_json=output_json,
            timeout_sec=timeout_sec,
            extra_args=["--command", command, "--output-dir", str(output_dir)],
        )

    def save_current_scene(self, output_path: Path) -> BlenderCommandResult:
        output_path = Path(output_path)
        output_dir = output_path.parent
        return self.run_module_command(
            command="save_scene",
            payload={"path": str(output_path)},
            output_dir=output_dir,
        )

    def _build_script_command(
        self,
        script_path: Path,
        input_json: Path | None,
        output_json: Path | None,
        extra_args: list[str] | None,
    ) -> list[str]:
        command = [self.config.blender_bin]
        if self.config.headless:
            command.append("--background")

        command.extend(["--python", str(script_path), "--"])

        if input_json is not None:
            command.extend(["--input", str(input_json)])
        if output_json is not None:
            command.extend(["--output", str(output_json)])
        if extra_args:
            command.extend(extra_args)

        return command

    def _format_process_error(
        self,
        completed: subprocess.CompletedProcess[str],
        output_json: Path | None,
    ) -> str:
        parts = [f"Blender exited with code {completed.returncode}"]
        if output_json is not None and output_json.exists():
            parts.append(f"output_json: {output_json.read_text(encoding='utf-8')}")
        if completed.stderr:
            parts.append(f"stderr:\n{_tail(completed.stderr)}")
        if completed.stdout:
            parts.append(f"stdout:\n{_tail(completed.stdout)}")
        return "\n".join(parts)
