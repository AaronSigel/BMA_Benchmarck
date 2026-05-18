from pathlib import Path

from benchmark.metrics.aggregate import (
    aggregate_run_results,
    group_by_execution_mode,
    group_by_task,
    rank_runs_by_score,
)
from benchmark.runner.models import ExecutionMode, RunResult, RunStatus


def make_run_result(
    run_id: str,
    task_id: str,
    status: RunStatus,
    score: float | None,
    mode: ExecutionMode = ExecutionMode.EXTERNAL_SNAPSHOT,
) -> RunResult:
    return RunResult(
        run_id=run_id,
        task_id=task_id,
        status=status,
        execution_mode=mode,
        validation_result_path=Path("validation_result.json") if score is not None else None,
        scene_snapshot_path=Path("scene_snapshot.json") if score is not None else None,
        artifacts_dir=Path("artifacts"),
        total_score=score,
        overall_status=status.value if status is not RunStatus.ERROR else None,
        started_at="2026-05-15T10:00:00Z",
        finished_at="2026-05-15T10:00:01Z",
        duration_sec=1.0,
        error="boom" if status is RunStatus.ERROR else None,
        summary={},
    )


def test_aggregate_run_results_ignores_missing_scores_for_average() -> None:
    results = [
        make_run_result("run_high", "task_a", RunStatus.PASSED, 1.0),
        make_run_result("run_low", "task_a", RunStatus.FAILED, 0.25),
        make_run_result("run_error", "task_b", RunStatus.ERROR, None),
    ]

    summary = aggregate_run_results(results)

    assert summary.total_runs == 3
    assert summary.attempted_runs == 3
    assert summary.completed_runs == 2
    assert summary.validated_runs == 2
    assert summary.passed_runs == 1
    assert summary.failed_runs == 1
    assert summary.error_runs == 1
    assert summary.average_score == 0.625
    assert summary.average_score_on_validated_runs == 0.625
    assert summary.min_score == 0.25
    assert summary.max_score == 1.0
    assert summary.success_rate_on_all_attempted_runs == 1 / 3
    assert [metric.name for metric in summary.metrics].count("total_score") == 2


def test_aggregate_run_results_returns_none_scores_when_no_scores_exist() -> None:
    summary = aggregate_run_results(
        [make_run_result("run_error", "task_a", RunStatus.ERROR, None)]
    )

    assert summary.average_score is None
    assert summary.min_score is None
    assert summary.max_score is None
    assert summary.error_runs == 1
    assert summary.validated_runs == 0
    assert summary.success_rate_on_all_attempted_runs == 0.0


def test_group_by_task() -> None:
    results = [
        make_run_result("run_1", "task_a", RunStatus.PASSED, 1.0),
        make_run_result("run_2", "task_a", RunStatus.FAILED, 0.0),
        make_run_result("run_3", "task_b", RunStatus.ERROR, None),
    ]

    grouped = group_by_task(results)

    assert set(grouped) == {"task_a", "task_b"}
    assert grouped["task_a"].total_runs == 2
    assert grouped["task_a"].average_score == 0.5
    assert grouped["task_b"].error_runs == 1


def test_group_by_execution_mode() -> None:
    results = [
        make_run_result(
            "run_external",
            "task_a",
            RunStatus.PASSED,
            1.0,
            ExecutionMode.EXTERNAL_SNAPSHOT,
        ),
        make_run_result("run_replay", "task_a", RunStatus.FAILED, 0.5, ExecutionMode.REPLAY),
    ]

    grouped = group_by_execution_mode(results)

    assert set(grouped) == {"external_snapshot", "replay"}
    assert grouped["external_snapshot"].passed_runs == 1
    assert grouped["replay"].failed_runs == 1


def test_rank_runs_by_score_sorts_descending_and_puts_missing_scores_last() -> None:
    results = [
        make_run_result("run_none", "task_a", RunStatus.ERROR, None),
        make_run_result("run_mid", "task_a", RunStatus.FAILED, 0.5),
        make_run_result("run_high", "task_a", RunStatus.PASSED, 1.0),
    ]

    ranked = rank_runs_by_score(results)

    assert [result.run_id for result in ranked] == ["run_high", "run_mid", "run_none"]
