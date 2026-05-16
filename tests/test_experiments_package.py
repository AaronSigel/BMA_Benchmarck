import importlib
from pathlib import Path


def test_experiments_package_modules_importable() -> None:
    modules = [
        "benchmark.experiments",
        "benchmark.experiments.models",
        "benchmark.experiments.matrix",
        "benchmark.experiments.generator",
        "benchmark.experiments.readiness",
        "benchmark.experiments.manifests",
        "benchmark.experiments.e2e_runner",
        "benchmark.experiments.cli",
    ]

    for module in modules:
        importlib.import_module(module)


def test_experiments_package_does_not_import_bpy() -> None:
    package_dir = Path(__file__).resolve().parents[1] / "benchmark" / "experiments"

    for path in package_dir.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import bpy" not in source
        assert "from bpy" not in source
