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


def cmd_smoke(cfg: McpServerConfig, tools: list[str]) -> int:
    if not is_blender_socket_available(cfg):
        print(json.dumps({
            "status": "FAIL",
            "error": f"Blender socket not reachable at {cfg.blender_host}:{cfg.blender_port}",
        }))
        return 1

    adapter = ExternalBlenderMcpServerAdapter(cfg)
    results: dict = {}
    failed = False
    for tool in tools:
        try:
            results[tool] = {"status": "ok", "result": adapter.call_tool(tool)}
        except McpLayerError as exc:
            results[tool] = {"status": "error", "error": str(exc)}
            failed = True

    print(json.dumps({"profile": cfg.profile, "tools": results}, default=str))
    return 1 if failed else 0


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
        sys.exit(cmd_smoke(cfg, args.tools))


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
