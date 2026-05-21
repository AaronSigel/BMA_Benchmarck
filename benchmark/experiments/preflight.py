from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.mcp.config import load_mcp_config
from benchmark.mcp.profiles import McpProfile
from benchmark.mcp.server_adapter import ExternalBlenderMcpServerAdapter
from benchmark.mcp.tool_contract import TOOL_CONTRACT_MAP
from benchmark.mcp.tool_registry import McpToolRegistry
from benchmark.runner.models import ExecutionMode, ExperimentConfig, RunConfig
from benchmark.blender.config import find_blender_executable


class PreflightError(RuntimeError):
    """Raised when an experiment cannot safely start."""


def write_preflight_report(config: ExperimentConfig, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    report = build_preflight_report(config, output_root)
    (output_root / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_root / "preflight_report.md").write_text(_preflight_markdown(report), encoding="utf-8")
    return report


def build_preflight_report(config: ExperimentConfig, output_root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, started: float, details: dict[str, Any] | None = None) -> None:
        checks.append({
            "name": name,
            "status": status,
            "duration_sec": round(time.perf_counter() - started, 6),
            "details": details or {},
        })

    started = time.perf_counter()
    add("python_environment", "passed", started, {"python": sys.version.split()[0], "executable": sys.executable})

    started = time.perf_counter()
    blender = find_blender_executable()
    needs_blender = any(run.execution_mode in {ExecutionMode.BLENDER_SMOKE, ExecutionMode.AGENT_MCP, ExecutionMode.REMOTE_AGENT} for run in config.runs)
    add(
        "blender_available",
        "passed" if blender else ("warning" if needs_blender else "skipped"),
        started,
        {"path": str(blender) if blender else None, "required": needs_blender},
    )

    started = time.perf_counter()
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".preflight_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        add("report_output_directory_writable", "passed", started, {"output_root": str(output_root)})
    except OSError as exc:
        add("report_output_directory_writable", "failed", started, {"error": str(exc), "output_root": str(output_root)})

    started = time.perf_counter()
    profile_preflight = run_profile_preflight_for_experiment(config)
    add("mcp_profiles", "passed" if profile_preflight.get("ok") else "failed", started, profile_preflight)

    mcp_run = _first_mcp_run(config)
    started = time.perf_counter()
    if mcp_run is None or mcp_run.mcp_config_path is None:
        add("mcp_socket", "skipped", started, {"required": False})
    else:
        try:
            cfg = load_mcp_config(mcp_run.mcp_config_path)
            if mcp_run.mcp_profile:
                cfg = cfg.model_copy(update={"profile": mcp_run.mcp_profile})
            proc = _find_blender_process(cfg.blender_host, cfg.blender_port)
            add("mcp_socket", "passed" if proc else "warning", started, {"host": cfg.blender_host, "port": cfg.blender_port, "process": proc})
        except Exception as exc:  # noqa: BLE001
            add("mcp_socket", "warning", started, {"error": str(exc)})

    started = time.perf_counter()
    models = sorted({str(run.metadata.get("model_id")) for run in config.runs if run.metadata.get("model_id")})
    needs_openrouter = any("openrouter" in str(run.agent_config_path or "") for run in config.runs)
    api_available = bool(os.environ.get("OPENROUTER_API_KEY"))
    add("openrouter_api", "passed" if (api_available or not needs_openrouter) else "warning", started, {"required": needs_openrouter, "env_var": "OPENROUTER_API_KEY", "available": api_available})

    started = time.perf_counter()
    add("model_config", "passed", started, {"models": models or ["default"]})

    for name in ("reset_scene", "get_scene_snapshot", "smoke_tool_call"):
        started = time.perf_counter()
        if mcp_run is None:
            add(name, "skipped", started, {"required": False})
        else:
            add(
                name,
                "delegated",
                started,
                {
                    "required": True,
                    "delegated_to_live_mcp_smoke": True,
                    "reason": "live MCP smoke is handled by run_contract_smoke_for_experiment when enabled",
                },
            )

    status = _overall_status(checks)
    return {"status": status, "checks": checks}


def _overall_status(checks: list[dict[str, Any]]) -> str:
    statuses = {str(check.get("status")) for check in checks}
    if "failed" in statuses:
        return "failed"
    if "warning" in statuses:
        return "warning"
    return "passed"


def _preflight_markdown(report: dict[str, Any]) -> str:
    lines = ["# Preflight Report", "", f"status: {report.get('status', 'unknown')}", "", "| Check | Status | Duration (s) |", "| --- | --- | --- |"]
    for check in report.get("checks", []):
        if isinstance(check, dict):
            lines.append(f"| {check.get('name')} | {check.get('status')} | {check.get('duration_sec')} |")
    return "\n".join(lines) + "\n"


def prepare_output_root(output_root: Path, *, clean_output: bool, allow_existing: bool = False) -> dict[str, Any]:
    """Ensure output root freshness policy before readiness creates directories."""
    existed_before = output_root.exists()
    existing_entries = sorted(p.name for p in output_root.iterdir()) if existed_before else []
    if existed_before and existing_entries and not clean_output and not allow_existing:
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
        "allow_existing": allow_existing,
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
        reset_result = adapter.reset_scene()
        if isinstance(reset_result, dict) and reset_result.get("warning"):
            raise RuntimeError(str(reset_result["warning"]))
        create_result = adapter.call_tool(
            "bma_create_object",
            {
                "type": "MESH_CUBE",
                "name": "ProbeCube",
                "dimensions": [2.0, 2.0, 2.0],
            },
        )
        if create_result.get("ok") is False:
            _err = create_result.get("error") or {}
            _msg = _err.get("message") if isinstance(_err, dict) else str(_err)
            raise RuntimeError(f"bma_create_object probe failed: {_msg}")
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
        if material_result.get("ok") is False:
            _err = material_result.get("error") or {}
            _msg = _err.get("message") if isinstance(_err, dict) else str(_err)
            raise RuntimeError(f"bma_set_material probe failed: {_msg}")
        adapter.collect_scene_snapshot(snapshot_path)
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        _assert_contract_snapshot(snapshot)
        export_results = _run_export_smoke(adapter, output_root)
    except AssertionError as exc:
        raise PreflightError(f"Socket is reachable, but running server does not support expected BMA contract: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        metadata["finished_at"] = _utc_now()
        metadata["error"] = str(exc)
        return metadata  # ok remains False; caller decides whether to abort
    finally:
        try:
            reset_result = adapter.reset_scene()
            if isinstance(reset_result, dict) and reset_result.get("warning"):
                metadata["cleanup_warning"] = str(reset_result["warning"])
        except Exception:  # noqa: BLE001
            pass

    metadata.update(
        {
            "ok": True,
            "finished_at": _utc_now(),
            "create_result": _json_safe(create_result),
            "material_result": _json_safe(material_result),
            "export_results": _json_safe(export_results),
        }
    )
    return metadata


def _run_export_smoke(adapter: ExternalBlenderMcpServerAdapter, output_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for export_format, relative_path in (("blend", "result.blend"), ("glb", "exports/result.glb")):
        reset_result = adapter.reset_scene()
        if isinstance(reset_result, dict) and reset_result.get("warning"):
            raise RuntimeError(str(reset_result["warning"]))
        create_result = adapter.call_tool(
            "bma_create_object",
            {
                "type": "MESH_CUBE",
                "name": "ExportProbeCube",
                "dimensions": [1.0, 1.0, 1.0],
            },
        )
        if create_result.get("ok") is False:
            _err = create_result.get("error") or {}
            _msg = _err.get("message") if isinstance(_err, dict) else str(_err)
            raise RuntimeError(f"export smoke create cube failed: {_msg}")
        filepath = output_root / "preflight_exports" / relative_path
        export_result = adapter.call_tool(
            "bma_export_scene",
            {
                "format": export_format,
                "filename": relative_path,
                "filepath": str(filepath),
            },
        )
        _assert_export_smoke_result(export_format, filepath, export_result)
        results.append(export_result)
    return results


def _assert_export_smoke_result(export_format: str, filepath: Path, result: dict[str, Any]) -> None:
    if result.get("ok") is False:
        err = result.get("error") or {}
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise RuntimeError(f"bma_export_scene {export_format} smoke failed: {msg}")
    payload = result.get("result") if isinstance(result.get("result"), dict) else {}
    exists = bool(payload.get("exists")) or filepath.exists()
    size = payload.get("file_size_bytes")
    if not isinstance(size, int):
        size = filepath.stat().st_size if filepath.exists() else 0
    if not exists:
        raise RuntimeError(f"bma_export_scene {export_format} smoke did not create {filepath}")
    if size <= 0:
        raise RuntimeError(f"bma_export_scene {export_format} smoke created empty file {filepath}")


EXPECTED_BMA_CONTRACT = frozenset({
    "bma_create_object",
    "bma_set_transform",
    "bma_assign_material",
    "bma_create_light",
    "bma_create_camera_look_at",
    "bma_export_scene",
    "bma_get_scene_snapshot",
    "bma_get_object_info",
})
DISALLOWED_AGENT_TOOLS = frozenset({
    "create_object",
    "assign_material",
    "export_scene",
})


def run_profile_preflight_for_experiment(config: ExperimentConfig) -> dict[str, Any]:
    profiles = sorted({run.mcp_profile for run in config.runs if run.mcp_profile})
    registry = McpToolRegistry()
    results = []
    for profile_name in profiles:
        try:
            profile = McpProfile(profile_name)
            advertised = {contract.name for contract in registry.list_for_profile(profile)}
            missing = sorted(EXPECTED_BMA_CONTRACT - advertised)
            disallowed = sorted(DISALLOWED_AGENT_TOOLS & advertised)
            disabled_for_safe_profiles = []
            if profile in {McpProfile.MINIMAL, McpProfile.NO_PYTHON, McpProfile.INSPECTION_ENABLED}:
                disabled_for_safe_profiles = sorted({"execute_blender_code"} & advertised)
            ok = not missing and not disallowed and not disabled_for_safe_profiles
            reason = None
            if missing:
                reason = f"Missing expected contract tools: {', '.join(missing)}"
            elif disallowed:
                reason = f"Unknown tool in advertised contract: {', '.join(disallowed)}"
            elif disabled_for_safe_profiles:
                reason = f"Disabled tools advertised for safe profile: {', '.join(disabled_for_safe_profiles)}"
            results.append({
                "profile": profile_name,
                "preflight_status": "passed" if ok else "failed",
                "reason": reason,
                "advertised_tools": sorted(advertised),
                "expected_contract": sorted(EXPECTED_BMA_CONTRACT),
                "checks": {
                    "create_object_tool": "bma_create_object" in advertised,
                    "material_tool": bool({"bma_assign_material", "bma_set_material"} & advertised),
                    "camera_tool": "bma_create_camera_look_at" in advertised,
                    "export_tool": "bma_export_scene" in advertised,
                    "inspection_tools": {"bma_get_scene_snapshot", "bma_get_object_info"}.issubset(advertised),
                    "disabled_tools_blocked": not disabled_for_safe_profiles,
                },
            })
        except Exception as exc:  # noqa: BLE001
            results.append({
                "profile": profile_name,
                "preflight_status": "failed",
                "reason": str(exc),
            })
    return {
        "enabled": True,
        "ok": all(item.get("preflight_status") == "passed" for item in results),
        "profiles": results,
    }


def filter_config_by_profile_preflight(config: ExperimentConfig, preflight: dict[str, Any]) -> ExperimentConfig:
    failed = {
        item["profile"]
        for item in preflight.get("profiles", [])
        if item.get("preflight_status") != "passed"
    }
    if not failed:
        return config
    runs = [run for run in config.runs if run.mcp_profile not in failed]
    metadata = {
        **config.metadata,
        "profile_preflight": preflight,
        "skipped_by_preflight": len(config.runs) - len(runs),
        "excluded_profiles": sorted(failed),
    }
    return config.model_copy(update={"runs": runs, "metadata": metadata})


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
