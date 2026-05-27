from __future__ import annotations

import json
from pathlib import Path

from benchmark.blender.config import BlenderConfig
from benchmark.blender.errors import BlenderError
from benchmark.blender.launcher import BlenderLauncher

from bma_benchmark.reporting.rendering.render_plan import RenderedSceneArtifacts, resolve_source_scene


def render_scene_artifacts(
    run_dir: Path,
    *,
    blender_bin: str | Path = "blender",
    width: int = 1280,
    height: int = 720,
    mode: str = "viewport",
    timeout_sec: int = 120,
    task_id: str | None = None,
) -> RenderedSceneArtifacts:
    run_dir = Path(run_dir)
    run_id = run_dir.name
    viewport_path = run_dir / "viewport.png"
    final_render_path = run_dir / "final_render.png"
    marker_path = run_dir / "render_not_available.json"

    if mode == "viewport" and viewport_path.is_file() and viewport_path.stat().st_size > 0:
        return RenderedSceneArtifacts(
            run_id=run_id,
            run_dir=run_dir,
            viewport_path=viewport_path,
            status="skipped",
            reason="viewport.png already exists",
        )
    if mode == "render" and final_render_path.is_file() and final_render_path.stat().st_size > 0:
        return RenderedSceneArtifacts(
            run_id=run_id,
            run_dir=run_dir,
            final_render_path=final_render_path,
            status="skipped",
            reason="final_render.png already exists",
        )
    if mode == "both" and viewport_path.is_file() and final_render_path.is_file():
        return RenderedSceneArtifacts(
            run_id=run_id,
            run_dir=run_dir,
            viewport_path=viewport_path,
            final_render_path=final_render_path,
            status="skipped",
            reason="render outputs already exist",
        )

    if task_id is None:
        task_id = _task_id_from_run(run_dir)

    source_path, reason = resolve_source_scene(run_dir, task_id=task_id)
    if source_path is None:
        _write_render_not_available(run_dir, reason or "no source scene")
        return RenderedSceneArtifacts(
            run_id=run_id,
            run_dir=run_dir,
            status="failed",
            reason=reason,
        )

    launcher = BlenderLauncher(BlenderConfig(blender_bin=str(blender_bin)))
    payload = {
        "source_path": str(source_path.resolve()),
        "output_dir": str(run_dir.resolve()),
        "mode": mode,
        "width": width,
        "height": height,
    }

    try:
        result = launcher.run_module_command(
            command="render_report_scene",
            payload=payload,
            output_dir=run_dir,
            timeout_sec=timeout_sec,
        )
    except BlenderError as exc:
        message = str(exc)
        _write_render_not_available(run_dir, message, source=str(source_path))
        return RenderedSceneArtifacts(
            run_id=run_id,
            run_dir=run_dir,
            source_scene_path=source_path,
            status="failed",
            reason=message,
        )

    output_json = run_dir / "output.json"
    render_result = _read_render_result(output_json, result.stdout if hasattr(result, "stdout") else None)
    if not render_result.get("ok"):
        message = str(render_result.get("error") or "render_report_scene failed")
        _write_render_not_available(run_dir, message, source=str(source_path))
        return RenderedSceneArtifacts(
            run_id=run_id,
            run_dir=run_dir,
            source_scene_path=source_path,
            status="failed",
            reason=message,
        )

    vp = Path(render_result["viewport_path"]) if render_result.get("viewport_path") else None
    fr = Path(render_result["final_render_path"]) if render_result.get("final_render_path") else None
    if vp is None and viewport_path.is_file():
        vp = viewport_path
    if fr is None and final_render_path.is_file():
        fr = final_render_path

    if marker_path.exists():
        marker_path.unlink()

    if vp is None and fr is None:
        reason = "render produced no PNG outputs"
        _write_render_not_available(run_dir, reason, source=str(source_path))
        return RenderedSceneArtifacts(
            run_id=run_id,
            run_dir=run_dir,
            source_scene_path=source_path,
            status="failed",
            reason=reason,
        )

    return RenderedSceneArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        source_scene_path=source_path,
        viewport_path=vp if vp and vp.is_file() else None,
        final_render_path=fr if fr and fr.is_file() else None,
        status="rendered",
    )


def _read_render_result(output_json: Path, stdout: str | None) -> dict:
    if output_json.is_file():
        try:
            data = json.loads(output_json.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                inner = data.get("result")
                if isinstance(inner, dict):
                    return inner
                return data
        except (OSError, json.JSONDecodeError):
            pass
    if stdout:
        try:
            data = json.loads(stdout.strip().splitlines()[-1])
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError, IndexError):
            pass
    return {"ok": False, "error": "missing render output json"}


def _write_render_not_available(run_dir: Path, reason: str, *, source: str | None = None) -> None:
    payload = {
        "reason": reason,
        "stage": "render_report_scene",
    }
    if source:
        payload["source"] = source
    path = run_dir / "render_not_available.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _task_id_from_run(run_dir: Path) -> str | None:
    run_result = run_dir / "run_result.json"
    if not run_result.is_file():
        return None
    try:
        data = json.loads(run_result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    task_id = data.get("task_id")
    return str(task_id) if task_id else None
