from __future__ import annotations

from benchmark.runner.error_classification import ErrorClass, classify_failure


def test_empty_socket_not_model_failure() -> None:
    result = classify_failure(error_type="EmptySocketResponse", failure_stage="socket_response")
    assert result.is_infra_failure is True
    assert result.is_model_failure is False
    assert result.error_class == ErrorClass.INFRA_ERROR


def test_reset_failed_not_model_failure() -> None:
    result = classify_failure(error_type="ResetSceneFailed", failure_stage="reset_scene")
    assert result.is_infra_failure is True
    assert result.is_model_failure is False


def test_react_invalid_action_after_passed_scene_is_soft_diagnostic() -> None:
    result = classify_failure(
        error_type="ReactInvalidAction",
        scene_status="passed",
        scene_passed_but_agent_error=True,
    )
    assert result.error_class == ErrorClass.SOFT_SUCCESS_DIAGNOSTIC
    assert result.is_model_failure is False
    assert result.diagnostic_only is True


def test_react_no_progress_with_tool_failure_not_model_failure() -> None:
    result = classify_failure(
        error_type="ReactNoProgress",
        no_progress_reason="tool_failed",
    )
    assert result.error_class == ErrorClass.TOOL_RUNTIME_ERROR
    assert result.is_model_failure is False
    assert result.is_tool_runtime_failure is True
