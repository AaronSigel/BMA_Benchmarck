from pathlib import Path

import pytest
import yaml

from benchmark.analysis.models import ExperimentAnalysisResult, ExperimentSummary
from benchmark.experiments import e2e_runner as e2e_module
from benchmark.experiments.e2e_runner import E2EBenchmarkRunner
from benchmark.runner.models import ExperimentResult

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "e2e"


class RecordingBatchRunner:
    def __init__(self, output_root: Path | None = None) -> None:
        self.output_root = output_root
        self.called = False
        self.run_ids: list[str] = []

    def run_experiment(self, config):
        self.called = True
        self.run_ids = [run.run_id for run in config.runs]
        if self.output_root is not None:
            assert (self.output_root / "manifest.json").is_file()
        return ExperimentResult(experiment_id=config.experiment_id, runs=[], summary={"total_runs": 0})


def write_smoke_matrix(tmp_path: Path) -> Path:
    matrix_path = tmp_path / "smoke_matrix.yaml"
    data = yaml.safe_load((FIXTURE_DIR / "smoke_matrix.yaml").read_text(encoding="utf-8"))
    data["output_root"] = str(tmp_path / "out")
    matrix_path.write_text(
        yaml.safe_dump(data),
        encoding="utf-8",
    )
    return matrix_path


def test_e2e_smoke_fixture_files_exist_and_are_service_free() -> None:
    expected = {
        "smoke_matrix.yaml",
        "mock_agent.yaml",
        "geometry_task.yaml",
        "scene_snapshot.json",
        "expected_report_excerpt.md",
    }
    assert expected == {path.name for path in FIXTURE_DIR.iterdir() if path.is_file()}

    matrix = yaml.safe_load((FIXTURE_DIR / "smoke_matrix.yaml").read_text(encoding="utf-8"))
    agent = yaml.safe_load((FIXTURE_DIR / "mock_agent.yaml").read_text(encoding="utf-8"))
    assert matrix["execution_modes"] == ["external_snapshot"]
    assert matrix["mcp_profiles"] == ["minimal"]
    assert agent["llm"]["provider"] == "mock"
    assert "api_key_env" not in agent["llm"]


def test_prepare_writes_manifest_before_batch_run(tmp_path: Path) -> None:
    matrix_path = write_smoke_matrix(tmp_path)
    output_root = tmp_path / "out"
    runner = E2EBenchmarkRunner(batch_runner=RecordingBatchRunner(output_root))

    result = runner.run(matrix_path)

    assert result.experiment_id == "e2e_smoke_matrix"
    assert (output_root / "manifest.json").is_file()
    assert runner.batch_runner.called is True
    assert runner.batch_runner.run_ids == [
        "e2e_smoke_matrix__e2e_geometry_task__e2e_mock_agent__minimal__r1"
    ]


def test_prepare_creates_experiment_config(tmp_path: Path) -> None:
    matrix_path = write_smoke_matrix(tmp_path)
    output_root = tmp_path / "out"

    config = E2EBenchmarkRunner().prepare(matrix_path)

    assert config.experiment_id == "e2e_smoke_matrix"
    assert len(config.runs) == 1
    assert config.runs[0].run_id == "e2e_smoke_matrix__e2e_geometry_task__e2e_mock_agent__minimal__r1"
    assert config.runs[0].output_dir == output_root / config.runs[0].run_id
    assert (output_root / "manifest.json").is_file()


def test_run_launches_mock_experiment(tmp_path: Path) -> None:
    matrix_path = write_smoke_matrix(tmp_path)

    result = E2EBenchmarkRunner().run(matrix_path)

    assert result.experiment_id == "e2e_smoke_matrix"
    assert result.summary["total_runs"] == 1
    assert result.summary["passed_runs"] == 1


def test_run_and_analyze_creates_analysis(tmp_path: Path) -> None:
    matrix_path = write_smoke_matrix(tmp_path)
    output_root = tmp_path / "out"

    analysis = E2EBenchmarkRunner().run_and_analyze(matrix_path)

    assert analysis.experiment_id == "out"
    assert analysis.summary.total_runs == 1
    assert (output_root / "experiment_analysis.json").is_file()


def test_smoke_matrix_run_and_report_passes_without_external_services(tmp_path: Path) -> None:
    matrix_path = write_smoke_matrix(tmp_path)

    report_path = E2EBenchmarkRunner().run_and_report(matrix_path)

    output_root = tmp_path / "out"
    assert report_path == output_root / "report.md"
    assert report_path.is_file()
    assert (output_root / "report.html").is_file()
    assert (output_root / "experiment_analysis.json").is_file()
    assert (output_root / "experiment_result.json").is_file()
    assert (output_root / "manifest.json").is_file()
    report_text = report_path.read_text(encoding="utf-8")
    for line in (FIXTURE_DIR / "expected_report_excerpt.md").read_text(encoding="utf-8").splitlines():
        if line.strip():
            assert line in report_text


def test_run_and_report_requires_clean_output_for_existing_artifacts(tmp_path: Path) -> None:
    matrix_path = write_smoke_matrix(tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "old_run.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="--clean-output"):
        E2EBenchmarkRunner().run_and_report(matrix_path)


def test_run_and_report_clean_output_removes_existing_artifacts(tmp_path: Path) -> None:
    matrix_path = write_smoke_matrix(tmp_path)
    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "old_run.json").write_text("{}", encoding="utf-8")

    report_path = E2EBenchmarkRunner().run_and_report(matrix_path, clean_output=True)

    assert report_path.is_file()
    assert not (output_root / "old_run.json").exists()


def test_readiness_error_prevents_batch_start(tmp_path: Path) -> None:
    matrix_path = tmp_path / "bad_matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "matrix_id": "bad",
                "tasks": {"ids": ["missing_task"]},
                "agents": {"ids": ["mock_agent"]},
                "mcp_profiles": ["minimal"],
                "execution_modes": ["external_snapshot"],
                "output_root": str(tmp_path / "out"),
            }
        ),
        encoding="utf-8",
    )
    batch = RecordingBatchRunner()

    with pytest.raises(RuntimeError, match="missing_task"):
        E2EBenchmarkRunner(batch_runner=batch).run(matrix_path)

    assert batch.called is False


def test_run_and_report_orders_batch_analysis_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    matrix_path = write_smoke_matrix(tmp_path)
    calls: list[str] = []

    class OrderedBatchRunner(RecordingBatchRunner):
        def run_experiment(self, config):
            calls.append("batch")
            return super().run_experiment(config)

    def fake_analysis(output_root: Path):
        calls.append("analysis")
        return ExperimentAnalysisResult(
            experiment_id=output_root.name,
            runs=[],
            summary=ExperimentSummary(),
        )

    def fake_report(analysis, matrix):
        calls.append("report")
        path = matrix.output_root / "report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("report", encoding="utf-8")
        return path

    monkeypatch.setattr(e2e_module, "_run_analysis", fake_analysis)
    monkeypatch.setattr(e2e_module, "_build_reports", fake_report)

    report_path = E2EBenchmarkRunner(batch_runner=OrderedBatchRunner()).run_and_report(matrix_path)

    assert calls == ["batch", "analysis", "report"]
    assert report_path == tmp_path / "out" / "report.md"
