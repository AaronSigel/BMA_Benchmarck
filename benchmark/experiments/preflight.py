from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.mcp.config import load_mcp_config
from benchmark.mcp.server_adapter import ExternalBlenderMcpServerAdapter
from benchmark.mcp.tool_contract import TOOL_CONTRACT_MAP
from benchmark.runner.models import ExecutionMode, ExperimentConfig, RunConfig


class PreflightError(RuntimeError):
    """Raised when an experiment cannot safely start."""


def prepare_output_root(output_root: Path, *, clean_output: bool) -> dict[str, Any]:
    """Ensure output root freshness policy before readiness creates directories."""
    existed_before = output_root.exists()
    existing_entries = sorted(p.name for p in output_root.iterdir()) if existed_before else []
    if existed_before and existing_entries and not clean_output:
        raise PreflightError(
            f"Output root already contains artifacts: {output_root}. "
            "Re-run with --clean-output to remove stale pilot artifacts first."
        )
    if existed_before and clean_output:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        "output_root": str(output_root),
        "clean_output": clean_output,
        "existed_before": existed_before,
        "removed_existing_output": existed_before and clean_output,
        "existing_entry_count": len(existing_entries),
        "created_at": _utc_now(),
    }


def run_contract_smoke_for_experiment(config: ExperimentConfig, output_root: Path) -> dict[str, Any]:
    """Validate the live Blender/MCP process supports the benchmark BMA contract."""
    mcp_run = _first_mcp_run(config)
    if mcp_run is None:
        return {
            "required": False,
            "ok": True,
            "reason": "experiment has no MCP-backed runs",
        }

    if mcp_run.mcp_config_path is None:
        raise PreflightError("MCP-backed run has no mcp_config_path; cannot run contract smoke")

    mcp_cfg = load_mcp_config(mcp_run.mcp_config_path)
    if mcp_run.mcp_profile:
        mcp_cfg = mcp_cfg.model_copy(update={"profile": mcp_run.mcp_profile})

    adapter = ExternalBlenderMcpServerAdapter(mcp_cfg)
    snapshot_path = output_root / "preflight_contract_smoke_snapshot.json"
    metadata: dict[str, Any] = {
        "required": True,
        "ok": False,
        "profile": mcp_cfg.profile,
        "socket": {"host": mcp_cfg.blender_host, "port": mcp_cfg.blender_port},
        "config_path": str(mcp_run.mcp_config_path),
        "snapshot_path": str(snapshot_path),
        "tool_contract_hash": tool_contract_hash(),
        "started_at": _utc_now(),
    }

    try:
        adapter.reset_scene()
        create_result = adapter.call_tool(
            "bma_create_object",
            {
                "type": "MESH_CUBE",
                "name": "ProbeCube",
                "dimensions": [2.0, 2.0, 2.0],
            },
        )
        material_result = adapter.call_tool(
            "bma_set_material",
            {
                "object_name": "ProbeCube",
                "color": [1.0, 0.0, 0.0, 1.0],
                "material_name": "ProbeRed",
                "roughness": 0.5,
                "metallic": 0.0,
            },
        )
        adapter.collect_scene_snapshot(snapshot_path)
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        _assert_contract_snapshot(snapshot)
    except Exception as exc:  # noqa: BLE001
        metadata["finished_at"] = _utc_now()
        metadata["error"] = str(exc)
        try:
            adapter.reset_scene()
        except Exception:  # noqa: BLE001
            pass
        raise PreflightError(
            "Socket is reachable, but running server does not support expected BMA contract: "
            f"{exc}"
        ) from exc
    finally:
        try:
            adapter.reset_scene()
        except Exception:  # noqa: BLE001
            pass

    metadata.update(
        {
            "ok": True,
            "finished_at": _utc_now(),
            "create_result": _json_safe(create_result),
            "material_result": _json_safe(material_result),
        }
    )
    return metadata


def collect_runtime_metadata(config: ExperimentConfig, output_root: Path) -> dict[str, Any]:
    mcp_run = _first_mcp_run(config)
    mcp_config_path = mcp_run.mcp_config_path if mcp_run is not None else None
    mcp_profile = mcp_run.mcp_profile if mcp_run is not None else None
    runtime: dict[str, Any] = {
        "created_at": _utc_now(),
        "output_root": str(output_root),
        "git_dirty": _git_dirty(),
        "tool_contract_hash": tool_contract_hash(),
        "mcp_profile": mcp_profile,
        "mcp_config_path": str(mcp_config_path) if mcp_config_path is not None else None,
        "latest_run_file_mtime": latest_artifact_mtime(output_root),
    }
    if mcp_config_path is not None:
        try:
            cfg = load_mcp_config(mcp_config_path)
            if mcp_profile:
                cfg = cfg.model_copy(update={"profile": mcp_profile})
            runtime["mcp_socket"] = {"host": cfg.blender_host, "port": cfg.blender_port}
            runtime["blender_process"] = _find_blender_process(cfg.blender_host, cfg.blender_port)
        except Exception as exc:  # noqa: BLE001
            runtime["mcp_runtime_error"] = str(exc)
    return runtime


def latest_artifact_mtime(output_root: Path) -> str | None:
    latest: float | None = None
    if not output_root.exists():
        return None
    for path in output_root.rglob("*"):
        if path.is_file():
            mtime = path.stat().st_mtime
            latest = mtime if latest is None else max(latest, mtime)
    if latest is None:
        return None
    return datetime.fromtimestamp(latest, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def tool_contract_hash() -> str:
    names = ["bma_create_object", "bma_set_transform", "bma_set_material"]
    payload = {
        name: TOOL_CONTRACT_MAP[name].model_dump(mode="json", exclude_none=True)
        for name in names
        if name in TOOL_CONTRACT_MAP
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _first_mcp_run(config: ExperimentConfig) -> RunConfig | None:
    for run in config.runs:
        if run.execution_mode in {
            ExecutionMode.MCP_SMOKE,
            ExecutionMode.MCP_EXTERNAL,
            ExecutionMode.AGENT_MCP,
            ExecutionMode.REMOTE_AGENT,
        }:
            return run
    return None


def _assert_contract_snapshot(snapshot: dict[str, Any]) -> None:
    objects = {obj.get("name"): obj for obj in snapshot.get("objects", []) if isinstance(obj, dict)}
    probe = objects.get("ProbeCube")
    if probe is None:
        raise AssertionError("ProbeCube missing from contract smoke snapshot")
    dims = _vector_values(probe.get("dimensions"))
    if not _vec_close(dims, [2.0, 2.0, 2.0]):
        raise AssertionError(f"ProbeCube dimensions mismatch: {dims}")
    if probe.get("primitive_hint") != "cube":
        raise AssertionError(f"ProbeCube primitive_hint mismatch: {probe.get('primitive_hint')!r}")
    slots = probe.get("material_slots") or []
    if "ProbeRed" not in slots:
        raise AssertionError(f"ProbeCube material_slots missing ProbeRed: {slots}")
    materials = {mat.get("name") for mat in snapshot.get("materials", []) if isinstance(mat, dict)}
    if "ProbeRed" not in materials:
        raise AssertionError(f"ProbeRed material missing from snapshot: {sorted(materials)}")


def _vector_values(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        try:
            return [float(value["x"]), float(value["y"]), float(value["z"])]
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, list) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
    return None


def _vec_close(actual: list[float] | None, expected: list[float], tolerance: float = 1e-4) -> bool:
    if actual is None or len(actual) != len(expected):
        return False
    return all(abs(a - e) <= tolerance for a, e in zip(actual, expected))


def _find_blender_process(host: str, port: int) -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    port_text = str(port)
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if "blender" not in stripped or port_text not in stripped:
            continue
        pid_text, _, args = stripped.partition(" ")
        addon_path = _arg_after(args.split(), "--addon")
        return {
            "pid": int(pid_text) if pid_text.isdigit() else pid_text,
            "host": host,
            "port": port,
            "addon_path": addon_path,
            "argv": args,
        }
    return None


def _arg_after(argv: list[str], flag: str) -> str | None:
    try:
        idx = argv.index(flag)
    except ValueError:
        return None
    if idx + 1 >= len(argv):
        return None
    return argv[idx + 1]


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
