import subprocess
import sys


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "benchmark.tasks.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_list_outputs_tasks() -> None:
    result = run_cli("list", "--tasks-dir", "tasks")

    assert result.returncode == 0
    assert "geometry_001_basic_primitives" in result.stdout
    assert "category" in result.stdout
    assert "difficulty" in result.stdout


def test_cli_show_outputs_task_details() -> None:
    result = run_cli("show", "geometry_001_basic_primitives", "--tasks-dir", "tasks")

    assert result.returncode == 0
    assert "id: geometry_001_basic_primitives" in result.stdout
    assert "prompt:" in result.stdout
    assert "allowed_tools:" in result.stdout
    assert "expected_scene:" in result.stdout
    assert "success_criteria:" in result.stdout


def test_cli_show_unknown_task_outputs_clear_error() -> None:
    result = run_cli("show", "missing_task", "--tasks-dir", "tasks")

    assert result.returncode == 1
    assert "unknown task id 'missing_task'" in result.stdout


def test_cli_validate_checks_task_set() -> None:
    result = run_cli("validate", "--tasks-dir", "tasks")

    assert result.returncode == 0
    assert "Found 18 task(s)" in result.stdout
    assert "No warnings" in result.stdout

