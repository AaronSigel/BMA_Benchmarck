from __future__ import annotations

from benchmark.runner.error_classification import (
    ErrorClass,
    classify_failure,
    is_hard_model_failure,
    is_soft_success_diagnostic,
)


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


def test_hard_model_failure_excludes_soft_diagnostic() -> None:
    assert is_hard_model_failure(
        is_model_failure=True,
        error_class=ErrorClass.SOFT_SUCCESS_DIAGNOSTIC.value,
        diagnostic_only=True,
    ) is False
    assert is_hard_model_failure(
        is_model_failure=True,
        pass_type="soft_pass",
        scene_status="passed",
        error_type="ReactMaxSteps",
    ) is False
    assert is_hard_model_failure(
        is_model_failure=True,
        error_type="ReactNoProgress",
        scene_status="failed",
    ) is True


def test_soft_success_diagnostic_helpers() -> None:
    assert is_soft_success_diagnostic(
        error_class=ErrorClass.SOFT_SUCCESS_DIAGNOSTIC.value,
    ) is True
    assert is_soft_success_diagnostic(diagnostic_only=True) is True
    assert is_soft_success_diagnostic(error_class=ErrorClass.AGENT_ERROR.value) is False


def test_classification_merge_overwrites_false_flags() -> None:
    metrics = {"is_model_failure": True, "structured_error_type": "ReactMaxSteps"}
    classification = classify_failure(
        error_type="ReactMaxSteps",
        scene_status="passed",
        scene_passed_but_agent_error=True,
    )
    always_apply = frozenset({
        "is_model_failure",
        "is_agent_failure",
        "is_infra_failure",
        "is_validation_failure",
        "is_tool_runtime_failure",
        "is_scene_available",
        "scene_passed_before_error",
        "diagnostic_only",
    })
    for key, value in classification.model_dump(mode="json").items():
        if key in always_apply:
            metrics[key] = value
    assert metrics["is_model_failure"] is False
    assert metrics["diagnostic_only"] is True
