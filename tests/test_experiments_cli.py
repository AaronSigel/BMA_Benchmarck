from pathlib import Path

import yaml

from benchmark.experiments import cli


def write_smoke_matrix(tmp_path: Path) -> Path:
    matrix_path = tmp_path / "smoke_matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "matrix_id": "cli_smoke_matrix",
                "title": "CLI Smoke Matrix",
                "tasks": {"ids": ["geometry_001_basic_primitives"]},
                "agents": {"ids": ["mock_agent"]},
                "mcp_profiles": ["minimal"],
                "execution_modes": ["external_snapshot"],
                "repetitions": 1,
                "output_root": str(tmp_path / "out"),
                "metadata": {
                    "snapshot_path": "tests/fixtures/validation/valid_geometry_snapshot.json",
                    "artifacts_dir": "tests/fixtures/validation",
                },
            }
        ),
        encoding="utf-8",
    )
    return matrix_path


def test_generate_creates_experiment_config(tmp_path: Path, capsys) -> None:
    matrix_path = write_smoke_matrix(tmp_path)
    output_path = tmp_path / "experiment.yaml"

    exit_code = cli.main(["generate", "--matrix", str(matrix_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    data = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "wrote" in captured.out
    assert data["experiment_id"] == "cli_smoke_matrix"
    assert len(data["runs"]) == 1


def test_readiness_prints_pass_and_writes_json(tmp_path: Path, capsys) -> None:
    matrix_path = write_smoke_matrix(tmp_path)
    output_path = tmp_path / "readiness.json"

    exit_code = cli.main(["readiness", "--matrix", str(matrix_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "status: pass" in captured.out
    assert output_path.is_file()


def test_readiness_prints_warnings_for_api_matrix(capsys, monkeypatch) -> None:
    monkeypatch.setenv("BMA_DISABLE_DOTENV", "1")
    for key in ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        monkeypatch.delenv(key, raising=False)

    exit_code = cli.main(["readiness", "--matrix", "configs/matrices/api_models_matrix.yaml"])

    captured = capsys.readouterr()
    assert exit_code in {0, 1}
    assert "status:" in captured.out
    assert "WARNING:" in captured.out
    assert "OPENROUTER_API_KEY" in captured.out


def test_run_and_report_creates_outputs(tmp_path: Path, capsys) -> None:
    matrix_path = write_smoke_matrix(tmp_path)

    exit_code = cli.main(["run-and-report", "--matrix", str(matrix_path)])

    captured = capsys.readouterr()
    output_root = tmp_path / "out"
    assert exit_code == 0
    assert f"report: {output_root / 'report.md'}" in captured.out
    assert (output_root / "experiment_result.json").is_file()
    assert (output_root / "experiment_analysis.json").is_file()
    assert (output_root / "report.md").is_file()
    assert (output_root / "report.html").is_file()


def test_run_and_report_clean_output_flag(tmp_path: Path, capsys) -> None:
    matrix_path = write_smoke_matrix(tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "stale.json").write_text("{}", encoding="utf-8")

    exit_code = cli.main(["run-and-report", "--matrix", str(matrix_path), "--clean-output"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"report: {output_root / 'report.md'}" in captured.out
    assert not (output_root / "stale.json").exists()


def test_run_and_report_existing_output_returns_controlled_error(tmp_path: Path, capsys) -> None:
    matrix_path = write_smoke_matrix(tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "stale.json").write_text("{}", encoding="utf-8")

    exit_code = cli.main(["run-and-report", "--matrix", str(matrix_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "ERROR:" in captured.out
    assert "--clean-output" in captured.out


def test_run_launches_smoke_matrix(tmp_path: Path, capsys) -> None:
    matrix_path = write_smoke_matrix(tmp_path)

    exit_code = cli.main(["run", "--matrix", str(matrix_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"experiment_id": "cli_smoke_matrix"' in captured.out
    assert '"status": "passed"' in captured.out
    assert (tmp_path / "out" / "experiment_result.json").is_file()


def test_run_and_analyze_creates_analysis(tmp_path: Path, capsys) -> None:
    matrix_path = write_smoke_matrix(tmp_path)

    exit_code = cli.main(["run-and-analyze", "--matrix", str(matrix_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"experiment_id": "out"' in captured.out
    assert (tmp_path / "out" / "experiment_analysis.json").is_file()


def test_list_matrices_prints_available_matrix_files(tmp_path: Path, capsys) -> None:
    matrix_path = write_smoke_matrix(tmp_path)

    exit_code = cli.main(["list-matrices", "--directory", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "cli_smoke_matrix" in captured.out
    assert str(matrix_path) in captured.out
