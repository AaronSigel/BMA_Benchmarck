"""CLI for MCP layer: check, list-tools, start-server, start-headless-blender, smoke."""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from benchmark.mcp.config import McpServerConfig, config_from_env, load_mcp_config
from benchmark.mcp.connection_check import check_mcp_server_config, is_blender_socket_available
from benchmark.mcp.errors import McpLayerError, McpServerStartError
from benchmark.mcp.profiles import McpProfile
from benchmark.mcp.server_adapter import ExternalBlenderMcpServerAdapter
from benchmark.mcp.tool_registry import McpToolRegistry


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bma-mcp",
        description="BMA MCP utilities — check, list-tools, start-server, start-headless-blender, smoke",
    )
    parser.add_argument("--config", metavar="PATH", help="Path to MCP config YAML")
    parser.add_argument("--host", default=None, help="Blender socket host override")
    parser.add_argument("--port", type=int, default=None, help="Blender socket port override")
    parser.add_argument(
        "--profile",
        choices=[p.value for p in McpProfile],
        default=None,
        help="Tool-gating profile override",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # check
    sub.add_parser("check", help="Check Blender socket connectivity and config validity")

    # list-tools
    sub.add_parser("list-tools", help="List tools available for the active profile")

    # start-server
    ss = sub.add_parser("start-server", help="Start the blender-mcp MCP server process")
    ss.add_argument(
        "--distribution",
        choices=["upstream", "fork", "local"],
        default=None,
        help="Server distribution override",
    )
    ss.add_argument("--package-source", default=None, help="package_source override for fork/local")
    ss.add_argument(
        "--wait",
        action="store_true",
        help="Block until SIGINT/SIGTERM, then stop the server",
    )

    # start-headless-blender
    shb = sub.add_parser(
        "start-headless-blender",
        help="Launch Blender in --background with the MCP add-on (headless mode)",
    )
    shb.add_argument("--addon", default=None, help="Path to addon.py (overrides auto-discovery)")
    shb.add_argument(
        "--wait",
        action="store_true",
        help="Block until SIGINT/SIGTERM, then stop Blender",
    )

    # smoke
    smoke_p = sub.add_parser("smoke", help="Run a minimal tool smoke-check (no LLM)")
    smoke_p.add_argument(
        "--tools",
        nargs="+",
        default=["get_scene_info"],
        help="Tools to call during smoke (default: get_scene_info)",
    )
    smoke_p.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Write smoke result JSON to this file",
    )

    return parser


def _resolve_config(args: argparse.Namespace) -> McpServerConfig:
    if getattr(args, "config", None):
        cfg = load_mcp_config(Path(args.config))
    else:
        cfg = config_from_env()

    overrides: dict = {}
    if getattr(args, "host", None) is not None:
        overrides["blender_host"] = args.host
    if getattr(args, "port", None) is not None:
        overrides["blender_port"] = args.port
    if getattr(args, "profile", None) is not None:
        overrides["profile"] = args.profile
    if getattr(args, "distribution", None) is not None:
        overrides["server_distribution"] = args.distribution
    if getattr(args, "package_source", None) is not None:
        overrides["package_source"] = args.package_source

    if overrides:
        cfg = cfg.model_copy(update=overrides)
    return cfg


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_check(cfg: McpServerConfig) -> int:
    result = check_mcp_server_config(cfg)
    print(json.dumps({
        "status": "OK" if result.ok else "FAIL",
        "blender_socket_available": result.blender_socket_available,
        "host": result.host,
        "port": result.port,
        "profile": result.profile,
        "telemetry_disabled": result.telemetry_disabled,
        "issues": result.issues,
    }))
    return 0 if result.ok else 1


def cmd_list_tools(cfg: McpServerConfig) -> int:
    registry = McpToolRegistry()
    try:
        profile = McpProfile(cfg.profile)
    except ValueError:
        profile = McpProfile.FULL
    tools = [c.name for c in registry.list_for_profile(profile)]
    print(json.dumps({"profile": cfg.profile, "count": len(tools), "tools": tools}))
    return 0


def cmd_start_server(cfg: McpServerConfig, wait: bool = False) -> int:
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    try:
        cmd = adapter.build_command()
    except McpServerStartError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}))
        return 1

    print(json.dumps({"status": "starting", "command": cmd}))
    try:
        process = adapter.start()
    except McpServerStartError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}))
        return 1

    print(json.dumps({"status": "started", "pid": process.pid}))

    if not wait:
        return 0

    return _wait_for_process(lambda: adapter.is_running(process), lambda: adapter.stop(process))


def cmd_start_headless_blender(
    cfg: McpServerConfig,
    addon_path: str | None = None,
    wait: bool = False,
) -> int:
    from benchmark.mcp.headless.launcher import HeadlessBlenderMcpLauncher

    addon = Path(addon_path) if addon_path else None
    launcher = HeadlessBlenderMcpLauncher(cfg, addon_path=addon)

    try:
        cmd = launcher.build_command()
    except McpServerStartError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}))
        return 1

    print(json.dumps({"status": "starting", "command": cmd}))
    try:
        launcher.start()
    except McpServerStartError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}))
        return 1

    print(json.dumps({"status": "started"}))

    if not wait:
        return 0

    return _wait_for_process(lambda: launcher.is_running, launcher.stop)


def cmd_smoke(
    cfg: McpServerConfig,
    tools: list[str],
    output: str | None = None,
) -> int:
    """Run a comprehensive MCP smoke-check and optionally write results to a JSON file.

    Checks:
      1. Config loaded              — always true if we reach here
      2. Telemetry disabled         — cfg.disable_telemetry
      3. Profile active             — cfg.profile is valid
      4. Blender socket available   — TCP probe
      5. MCP server available       — same socket probe (blender-mcp listens on it)
      6. get_scene_info             — always included
      7. get_bma_profile_info       — included when server_distribution in (fork, local)
    """
    import datetime

    report: dict = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": cfg.blender_host + ":" + str(cfg.blender_port),
        "profile": cfg.profile,
        "checks": {},
        "tool_results": {},
        "status": "PASS",
    }

    def _fail(reason: str) -> int:
        report["status"] = "FAIL"
        report["error"] = reason
        _emit(report, output)
        return 1

    # 1. Config loaded — trivially true
    report["checks"]["config_loaded"] = True

    # 2. Telemetry disabled
    report["checks"]["telemetry_disabled"] = cfg.disable_telemetry
    if not cfg.disable_telemetry:
        return _fail("Telemetry is not disabled (set disable_telemetry: true in config)")

    # 3. Profile valid
    try:
        McpProfile(cfg.profile)
        report["checks"]["profile_valid"] = True
    except ValueError:
        report["checks"]["profile_valid"] = False
        return _fail(f"Unknown profile: '{cfg.profile}'")

    # 4 + 5. Blender socket / MCP server reachable
    socket_ok = is_blender_socket_available(cfg)
    report["checks"]["blender_socket_available"] = socket_ok
    report["checks"]["mcp_server_available"] = socket_ok
    if not socket_ok:
        return _fail(
            f"Blender socket not reachable at {cfg.blender_host}:{cfg.blender_port}. "
            "Ensure Blender is running with the blender-mcp add-on active."
        )

    # Build tool list: always include get_scene_info; add get_bma_profile_info for fork/local
    smoke_tools = list(tools)
    is_fork = cfg.server_distribution in ("fork", "local")
    if is_fork and "get_bma_profile_info" not in smoke_tools:
        smoke_tools = ["get_bma_profile_info"] + smoke_tools

    # 6/7. Call tools
    adapter = ExternalBlenderMcpServerAdapter(cfg)
    failed = False
    for tool in smoke_tools:
        try:
            result = adapter.call_tool(tool)
            report["tool_results"][tool] = {"status": "ok", "result": result}
        except McpLayerError as exc:
            report["tool_results"][tool] = {"status": "error", "error": str(exc)}
            failed = True

    if failed:
        report["status"] = "FAIL"

    _emit(report, output)
    return 0 if not failed else 1


def _emit(report: dict, output: str | None) -> None:
    """Print report to stdout and optionally write to a file."""
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"[bma-mcp smoke] Result written to {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = _resolve_config(args)
    except McpLayerError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.command == "check":
        sys.exit(cmd_check(cfg))
    elif args.command == "list-tools":
        sys.exit(cmd_list_tools(cfg))
    elif args.command == "start-server":
        sys.exit(cmd_start_server(cfg, wait=getattr(args, "wait", False)))
    elif args.command == "start-headless-blender":
        sys.exit(cmd_start_headless_blender(
            cfg,
            addon_path=getattr(args, "addon", None),
            wait=getattr(args, "wait", False),
        ))
    elif args.command == "smoke":
        sys.exit(cmd_smoke(cfg, args.tools, output=getattr(args, "output", None)))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _wait_for_process(is_running_fn, stop_fn) -> int:
    """Block until SIGINT/SIGTERM, then call stop_fn."""
    _stop_requested = False

    def _handler(signum, frame):  # noqa: ARG001
        nonlocal _stop_requested
        print(f"\n[bma-mcp] Signal {signum} received, stopping…", file=sys.stderr)
        _stop_requested = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    try:
        while not _stop_requested and is_running_fn():
            time.sleep(0.5)
    finally:
        stop_fn()

    return 0
