import argparse
import json
import os
import sys
import traceback
from pathlib import Path

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

from benchmark.blender.scripts.collect_snapshot import collect_snapshot
from benchmark.blender.scripts.create_fixture_scene import create_fixture_scene
from benchmark.blender.scripts.export_scene import export_scene
from benchmark.blender.scripts.render_scene import render_scene
from benchmark.blender.scripts.reset_scene import reset_scene
from benchmark.blender.scripts.save_scene import save_scene


def parse_args(argv: list[str]) -> argparse.Namespace:
    blender_args = argv
    if "--" in argv:
        blender_args = argv[argv.index("--") + 1 :]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(blender_args)


def read_input(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_output(path: Path | None, data: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_smoke(output_dir: Path, payload: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    exports_dir = output_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    scene_name = payload.get("scene_name", "BMA Smoke Scene")
    results = {
        "reset_scene": reset_scene({"scene_name": scene_name}),
        "create_fixture_scene": create_fixture_scene({"scene_name": scene_name}),
        "collect_snapshot": collect_snapshot({"output_path": str(output_dir / "scene_snapshot.json")}),
        "save_scene": save_scene({"path": str(output_dir / "result.blend")}),
        "render_scene": render_scene({"output_path": str(output_dir / "render.png")}),
        "export_scene": export_scene(
            {
                "output_path": str(exports_dir / "result.glb"),
                "format": "glb",
            }
        ),
    }
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        payload = read_input(args.input)
        results = run_smoke(args.output_dir, payload)
        write_output(args.output, {"ok": True, "results": results, "error": None})
        return 0
    except Exception as exc:
        write_output(
            args.output,
            {
                "ok": False,
                "results": None,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
