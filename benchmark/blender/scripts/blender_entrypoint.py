import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VIRTUAL_ENV = os.environ.get("VIRTUAL_ENV")
if VIRTUAL_ENV:
    venv_site_packages = (
        Path(VIRTUAL_ENV)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if venv_site_packages.exists() and str(venv_site_packages) not in sys.path:
        sys.path.insert(0, str(venv_site_packages))


COMMANDS = {
    "reset_scene": ("benchmark.blender.scripts.reset_scene", "reset_scene"),
    "create_fixture_scene": (
        "benchmark.blender.scripts.create_fixture_scene",
        "create_fixture_scene",
    ),
    "collect_snapshot": (
        "benchmark.blender.scripts.collect_snapshot",
        "collect_snapshot",
    ),
    "save_scene": ("benchmark.blender.scripts.save_scene", "save_scene"),
    "render_scene": ("benchmark.blender.scripts.render_scene", "render_scene"),
    "render_report_scene": (
        "benchmark.blender.scripts.render_report_scene",
        "render_report_scene",
    ),
    "export_scene": ("benchmark.blender.scripts.export_scene", "export_scene"),
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    blender_args = argv
    if "--" in argv:
        blender_args = argv[argv.index("--") + 1 :]

    parser = argparse.ArgumentParser()
    parser.add_argument("--command")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(blender_args)


def read_input(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def resolve_output_path(output: Path | None, output_dir: Path | None) -> Path | None:
    if output is not None:
        return output
    if output_dir is not None:
        return output_dir / "output.json"
    return None


def write_output(path: Path | None, data: dict[str, Any]) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_command(command: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    module_name, function_name = COMMANDS[command]
    module = importlib.import_module(module_name)
    func = getattr(module, function_name)
    return func


def make_payload(input_data: dict[str, Any]) -> dict[str, Any]:
    payload = input_data.get("payload", input_data)
    if not isinstance(payload, dict):
        raise ValueError("input payload must be a JSON object")
    return payload


def run(argv: list[str]) -> int:
    args = parse_args(argv)
    output_path = resolve_output_path(args.output, args.output_dir)
    command = args.command

    try:
        input_data = read_input(args.input)
        command = command or input_data.get("command")
        if not command:
            raise ValueError("command is required")
        if command not in COMMANDS:
            raise ValueError(f"unsupported command: {command}")

        payload = make_payload(input_data)
        handler = load_command(command)
        result = handler(payload)
        response = {
            "ok": True,
            "command": command,
            "result": result,
            "error": None,
        }
        write_output(output_path, response)
        return 0
    except Exception as exc:
        response = {
            "ok": False,
            "command": command,
            "result": None,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_output(output_path, response)
        return 1


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
