from __future__ import annotations

from benchmark.agent.strategies.react import _classify_no_progress


def test_no_progress_scene_already_passed() -> None:
    assert _classify_no_progress(
        tool_result_ok=True,
        snapshot_available=True,
        validation_available=True,
        scene_already_passed=True,
        score_before=1.0,
        score_after=1.0,
        issue_count_before=0,
        issue_count_after=0,
        repeated_action=False,
    ) == "scene_already_passed"


def test_no_progress_tool_failed() -> None:
    assert _classify_no_progress(
        tool_result_ok=False,
        snapshot_available=True,
        validation_available=True,
        scene_already_passed=False,
        score_before=0.5,
        score_after=0.5,
        issue_count_before=2,
        issue_count_after=2,
        repeated_action=False,
    ) == "tool_failed"


def test_no_progress_repeated_action_model_failure() -> None:
    assert _classify_no_progress(
        tool_result_ok=True,
        snapshot_available=True,
        validation_available=True,
        scene_already_passed=False,
        score_before=0.5,
        score_after=0.5,
        issue_count_before=2,
        issue_count_after=2,
        repeated_action=True,
    ) == "repeated_action"
