"""Dynamic report_text_ru.md generation tests."""

from __future__ import annotations

from pathlib import Path

from benchmark.analysis.models import ExperimentAnalysisResult, ExperimentSummary, RunAnalysisResult
from benchmark.analysis.report_bundle import write_report_text_ru


def _analysis(
    runs=None,
    metadata=None,
    **summary_kwargs,
) -> ExperimentAnalysisResult:
    summary = ExperimentSummary(total_runs=4, **summary_kwargs)
    return ExperimentAnalysisResult(
        experiment_id="diagnostic_repeat_gemini_v5",
        runs=runs or [],
        summary=summary,
        metadata=metadata or {"repetitions": 2},
    )


def test_report_text_uses_actual_repetitions(tmp_path: Path) -> None:
    analysis = _analysis(metadata={"repetitions": 2})
    out = tmp_path / "report_text_ru.md"
    write_report_text_ru(analysis, out)
    text = out.read_text(encoding="utf-8")
    assert "2 повторност" in text
    assert "одна повторность" not in text


def test_report_text_does_not_call_react_low_success_when_success_high(tmp_path: Path) -> None:
    runs = [
        RunAnalysisResult(
            run_id=f"r{i}",
            task_id="geometry_001",
            agent_id="react_openrouter",
            strategy="react",
            success=True,
            pass_type="clean_pass",
        )
        for i in range(9)
    ] + [
        RunAnalysisResult(
            run_id="r9",
            task_id="geometry_001",
            agent_id="react_openrouter",
            strategy="react",
            success=False,
            pass_type="runtime_error",
        ),
    ]
    analysis = _analysis(
        reported_success_rate=0.9,
        runs=runs,
        metadata={"repetitions": 2},
    )
    out = tmp_path / "report_text_ru.md"
    write_report_text_ru(analysis, out)
    text = out.read_text(encoding="utf-8")
    assert "validator-guided repair" in text
    assert "низкий success rate ReAct" not in text


def test_report_text_mentions_infra_model_validation_split(tmp_path: Path) -> None:
    analysis = _analysis(
        infra_error_rate=0.05,
        model_failure_rate=0.04,
        validation_failure_rate=0.03,
        tool_runtime_failure_rate=0.02,
        soft_success_diagnostic_rate=0.01,
        metadata={"repetitions": 2},
    )
    out = tmp_path / "report_text_ru.md"
    write_report_text_ru(analysis, out)
    text = out.read_text(encoding="utf-8")
    assert "инфраструктурные ошибки" in text
    assert "Model failure rate" in text
    assert "validation failure rate" in text
