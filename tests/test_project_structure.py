from pathlib import Path


def test_benchmark_package_importable() -> None:
    import benchmark

    assert benchmark.__doc__


def test_expected_project_directories_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_directories = [
        "benchmark",
        "benchmark/tasks",
        "benchmark/schemas",
        "tasks",
        "tasks/geometry",
        "tasks/materials",
        "tasks/lighting",
        "tasks/camera",
        "tasks/export",
        "tests",
    ]

    for directory in expected_directories:
        assert (root / directory).is_dir()

