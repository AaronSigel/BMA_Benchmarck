import argparse
import json
from pathlib import Path

from benchmark.blender.config import find_blender_executable
from benchmark.blender.errors import BlenderError
from benchmark.blender.launcher import BlenderLauncher


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        return _check()

    if args.command == "fixture":
        return _fixture(args.output_dir)
    if args.command == "snapshot":
        return _snapshot(args.output_dir)
    if args.command == "render":
        return _render(args.output_dir)
    if args.command == "export":
        return _export(args.output_dir, args.format)
    if args.command == "smoke":
        return _smoke(args.output_dir)

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Blender automation commands.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("check", help="Check whether Blender executable is available.")

    fixture_parser = subparsers.add_parser("fixture", help="Create fixture scene.")
    fixture_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/blender_smoke"))

    snapshot_parser = subparsers.add_parser("snapshot", help="Collect scene snapshot.")
    snapshot_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/blender_smoke"))

    render_parser = subparsers.add_parser("render", help="Render current scene.")
    render_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/blender_smoke"))

    export_parser = subparsers.add_parser("export", help="Export current scene.")
    export_parser.add_argument("--format", default="glb")
    export_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/blender_smoke"))

    smoke_parser = subparsers.add_parser("smoke", help="Run Blender automation smoke flow.")
    smoke_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/blender_smoke"))

    return parser


def _check() -> int:
    blender_bin = find_blender_executable()
    if blender_bin is None:
        print("ERROR: Blender executable not found. Set BMA_BLENDER_BIN or add blender to PATH.")
        return 1

    print(f"Blender executable: {blender_bin}")
    return 0


def _fixture(output_dir: Path) -> int:
    output_dir = Path(output_dir)
    return _run_blender_command(
        "create_fixture_scene",
        {
            "scene_name": "BMA Fixture Scene",
            "save_path": str(output_dir / "result.blend"),
        },
        output_dir,
    )


def _snapshot(output_dir: Path) -> int:
    output_dir = Path(output_dir)
    return _run_blender_command(
        "collect_snapshot",
        {"output_path": str(output_dir / "scene_snapshot.json")},
        output_dir,
    )


def _render(output_dir: Path) -> int:
    output_dir = Path(output_dir)
    return _run_blender_command(
        "render_scene",
        {"output_path": str(output_dir / "render.png")},
        output_dir,
    )


def _export(output_dir: Path, export_format: str) -> int:
    output_dir = Path(output_dir)
    return _run_blender_command(
        "export_scene",
        {
            "output_path": str(output_dir / "exports" / f"result.{export_format.lower().lstrip('.')}"),
            "format": export_format,
        },
        output_dir,
    )


def _smoke(output_dir: Path) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_json = output_dir / "smoke_input.json"
    output_json = output_dir / "smoke_output.json"
    input_json.write_text(json.dumps({"scene_name": "BMA Smoke Scene"}, indent=2), encoding="utf-8")

    try:
        launcher = BlenderLauncher()
        result = launcher.run_script(
            script_path=Path(__file__).parent / "scripts" / "smoke_scene.py",
            input_json=input_json,
            output_json=output_json,
            extra_args=["--output-dir", str(output_dir)],
        )
    except BlenderError as error:
        print(f"ERROR: {error}")
        return 1

    print("smoke: ok")
    for output_file in result.output_files:
        print(f"  {output_file}")
    print(f"Smoke artifacts written to {output_dir}")
    return 0


def _run_blender_command(command: str, payload: dict, output_dir: Path) -> int:
    try:
        launcher = BlenderLauncher()
        result = launcher.run_module_command(command=command, payload=payload, output_dir=output_dir)
    except BlenderError as error:
        print(f"ERROR: {error}")
        return 1

    print(f"{command}: ok")
    for output_file in result.output_files:
        print(f"  {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
