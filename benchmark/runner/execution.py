import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

import benchmark.blender.config as blender_config
import benchmark.blender.launcher as blender_launcher
from benchmark.blender.errors import BlenderError
from benchmark.blender.models import SceneSnapshot
from benchmark.runner.models import ExecutionMode, RunConfig


class ExecutionResult(BaseModel):
    ok: bool
    scene_snapshot_path: Path | None
    artifacts_dir: Path
    output_files: list[Path] = Field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionBackend(ABC):
    mode: ExecutionMode

    @abstractmethod
    def execute(self, config: RunConfig) -> ExecutionResult:
        """Produce or locate a SceneSnapshot for a run."""


class ExternalSnapshotBackend(ExecutionBackend):
    mode = ExecutionMode.EXTERNAL_SNAPSHOT

    def execute(self, config: RunConfig) -> ExecutionResult:
        if config.snapshot_path is None:
            return _error_result(config.artifacts_dir, "snapshot_path is required")

        snapshot_path = Path(config.snapshot_path)
        validation_error = _validate_snapshot_file(snapshot_path)
        if validation_error is not None:
            return _error_result(config.artifacts_dir, validation_error, snapshot_path)

        return ExecutionResult(
            ok=True,
            scene_snapshot_path=snapshot_path,
            artifacts_dir=config.artifacts_dir,
            output_files=[snapshot_path],
            metadata={"mode": self.mode.value},
        )

    def run(self, config: RunConfig) -> Path:
        result = self.execute(config)
        if not result.ok or result.scene_snapshot_path is None:
            raise ValueError(result.error or "external snapshot execution failed")
        return result.scene_snapshot_path


class ReplayBackend(ExecutionBackend):
    mode = ExecutionMode.REPLAY

    def execute(self, config: RunConfig) -> ExecutionResult:
        source_dir = Path(config.artifacts_dir)
        output_dir = Path(config.output_dir)
        if not source_dir.exists():
            return _error_result(output_dir, f"artifacts_dir does not exist: {source_dir}")
        if not source_dir.is_dir():
            return _error_result(output_dir, f"artifacts_dir is not a directory: {source_dir}")

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            _copy_artifacts(source_dir, output_dir)
        except OSError as error:
            return _error_result(output_dir, f"Failed to copy replay artifacts: {error}")

        snapshot_path = output_dir / "scene_snapshot.json"
        validation_error = _validate_snapshot_file(snapshot_path)
        if validation_error is not None:
            return _error_result(output_dir, validation_error, snapshot_path)

        return ExecutionResult(
            ok=True,
            scene_snapshot_path=snapshot_path,
            artifacts_dir=output_dir,
            output_files=_list_files(output_dir),
            metadata={"mode": self.mode.value, "source_artifacts_dir": str(source_dir)},
        )


class BlenderSmokeBackend(ExecutionBackend):
    mode = ExecutionMode.BLENDER_SMOKE

    def execute(self, config: RunConfig) -> ExecutionResult:
        blender_bin = blender_config.find_blender_executable()
        output_dir = Path(config.output_dir)
        if blender_bin is None:
            return _error_result(
                output_dir,
                "Blender executable not found. Set BMA_BLENDER_BIN or add blender to PATH.",
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        input_json = output_dir / "smoke_input.json"
        output_json = output_dir / "smoke_output.json"
        input_json.write_text(
            json.dumps({"scene_name": "BMA Smoke Scene"}, indent=2),
            encoding="utf-8",
        )

        try:
            launcher = blender_launcher.BlenderLauncher(
                blender_config.BlenderConfig(blender_bin=blender_bin)
            )
            command_result = launcher.run_script(
                script_path=Path(blender_launcher.__file__).parent / "scripts" / "smoke_scene.py",
                input_json=input_json,
                output_json=output_json,
                extra_args=["--output-dir", str(output_dir)],
            )
        except BlenderError as error:
            return _error_result(output_dir, str(error))
        except OSError as error:
            return _error_result(output_dir, f"Failed to run Blender smoke backend: {error}")

        snapshot_path = output_dir / "scene_snapshot.json"
        validation_error = _validate_snapshot_file(snapshot_path)
        if validation_error is not None:
            return _error_result(output_dir, validation_error, snapshot_path)

        result_blend = output_dir / "result.blend"
        final_blend = output_dir / "final_scene.blend"
        if result_blend.is_file() and not final_blend.exists():
            shutil.copy2(result_blend, final_blend)

        output_files = [Path(path) for path in command_result.output_files]
        for path in [input_json, output_json, snapshot_path]:
            if path not in output_files:
                output_files.append(path)

        return ExecutionResult(
            ok=True,
            scene_snapshot_path=snapshot_path,
            artifacts_dir=output_dir,
            output_files=output_files,
            metadata={
                "mode": self.mode.value,
                "command": command_result.command,
                "duration_sec": command_result.duration_sec,
            },
        )


def _copy_artifacts(source_dir: Path, output_dir: Path) -> None:
    if source_dir.resolve() == output_dir.resolve():
        return

    for path in source_dir.iterdir():
        destination = output_dir / path.name
        if path.resolve() == output_dir.resolve():
            continue
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(path, destination)


def _validate_snapshot_file(path: Path) -> str | None:
    if not path.exists():
        return f"scene snapshot does not exist: {path}"
    if not path.is_file():
        return f"scene snapshot is not a file: {path}"

    try:
        SceneSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        return f"Failed to read scene snapshot {path}: {error}"
    except ValidationError as error:
        return f"Invalid SceneSnapshot in {path}: {error}"

    return None


def _list_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _error_result(
    artifacts_dir: Path,
    error: str,
    scene_snapshot_path: Path | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        ok=False,
        scene_snapshot_path=scene_snapshot_path,
        artifacts_dir=artifacts_dir,
        error=error,
    )
