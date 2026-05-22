from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorClass(str, Enum):
    INFRA_ERROR = "INFRA_ERROR"
    TOOL_RUNTIME_ERROR = "TOOL_RUNTIME_ERROR"
    AGENT_ERROR = "AGENT_ERROR"
    VALIDATION_FAILURE = "VALIDATION_FAILURE"
    SOFT_SUCCESS_DIAGNOSTIC = "SOFT_SUCCESS_DIAGNOSTIC"


_INFRA_ERROR_TYPES = frozenset({
    "EmptySocketResponse",
    "BlenderSocketNoResponse",
    "BlenderSocketUnavailable",
    "BlenderWorkerUnhealthy",
    "ResetSceneFailed",
    "SnapshotUnavailable",
    "ToolTimeout",
    "SocketTimeout",
    "SocketError",
    "PreflightCheckFailed",
})

_TRANSIENT_INFRA_ERROR_TYPES = frozenset({
    "EmptySocketResponse",
    "BlenderSocketNoResponse",
    "BlenderWorkerUnhealthy",
    "ResetSceneFailed",
    "SnapshotUnavailable",
    "ToolTimeout",
    "SocketTimeout",
    "SocketError",
})

_INFRA_PARSE_STAGES = frozenset({"socket_response", "tool_response_parse"})

_TOOL_RUNTIME_ERROR_TYPES = frozenset({
    "BlenderRuntimeError",
    "ToolExecutionFailed",
    "InvalidToolResponse",
    "InvalidJsonResponse",
    "EmptyToolResponse",
    "ToolError",
    "ExportUnavailable",
    "ExportFailed",
    "CameraLookAtFailed",
    "ValidationUnavailable",
})

_AGENT_ERROR_TYPES = frozenset({
    "ReactInvalidAction",
    "ReactNoProgress",
    "ReactMaxSteps",
    "ReactNonStrictResponse",
    "ReactBlockedExport",
    "DirectNoAction",
    "LlmParseError",
    "InvalidToolCall",
    "InvalidArguments",
    "LlmProviderError",
})

_SOFT_SUCCESS_TYPES = frozenset({
    "ScenePassedButAgentError",
    "ScenePassedWithWarning",
    "ScenePassedAfterRepair",
})


class FailureClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_class: ErrorClass | None = None
    error_type: str | None = None
    failure_stage: str | None = None
    is_model_failure: bool = False
    is_agent_failure: bool = False
    is_infra_failure: bool = False
    is_validation_failure: bool = False
    is_tool_runtime_failure: bool = False
    is_scene_available: bool = False
    scene_passed_before_error: bool = False
    diagnostic_only: bool = False
    no_progress_reason: str | None = None


def classify_failure(
    *,
    error_type: str | None = None,
    failure_stage: str | None = None,
    scene_status: str | None = None,
    run_status: str | None = None,
    scene_passed_but_agent_error: bool = False,
    early_stop_reason: str | None = None,
    validation_issues: list[Any] | None = None,
    no_progress_reason: str | None = None,
    runtime_healthy: bool = True,
) -> FailureClassification:
    """Классифицирует ошибку run по оси model/infra/validation/agent."""
    normalized_type = _normalize_error_type(error_type)
    scene_available = scene_status not in {None, "", "not_available", "skipped"}
    scene_passed = scene_status in {"passed", "warning"} or scene_passed_but_agent_error
    scene_passed_before = scene_passed or bool(early_stop_reason)

    if scene_passed_but_agent_error or (
        scene_passed_before and normalized_type in _AGENT_ERROR_TYPES
    ):
        return FailureClassification(
            error_class=ErrorClass.SOFT_SUCCESS_DIAGNOSTIC,
            error_type=normalized_type or "ScenePassedButAgentError",
            failure_stage=failure_stage,
            is_model_failure=False,
            is_agent_failure=True,
            is_infra_failure=False,
            is_validation_failure=False,
            is_tool_runtime_failure=False,
            is_scene_available=scene_available,
            scene_passed_before_error=True,
            diagnostic_only=True,
            no_progress_reason=no_progress_reason,
        )

    if no_progress_reason == "scene_already_passed":
        return FailureClassification(
            error_class=ErrorClass.SOFT_SUCCESS_DIAGNOSTIC,
            error_type=normalized_type or "ReactNoProgress",
            failure_stage=failure_stage,
            is_model_failure=False,
            is_agent_failure=True,
            is_scene_available=scene_available,
            scene_passed_before_error=True,
            diagnostic_only=True,
            no_progress_reason=no_progress_reason,
        )

    if no_progress_reason in {"tool_failed"}:
        if is_infra_error_type(normalized_type) or is_infra_parse_error(normalized_type, failure_stage):
            return FailureClassification(
                error_class=ErrorClass.INFRA_ERROR,
                error_type=normalized_type or "ToolExecutionFailed",
                failure_stage=failure_stage,
                is_infra_failure=True,
                is_model_failure=False,
                is_scene_available=scene_available,
                no_progress_reason=no_progress_reason,
            )
        return FailureClassification(
            error_class=ErrorClass.TOOL_RUNTIME_ERROR,
            error_type=normalized_type or "ToolExecutionFailed",
            failure_stage=failure_stage,
            is_tool_runtime_failure=True,
            is_model_failure=False,
            is_infra_failure=False,
            is_scene_available=scene_available,
            no_progress_reason=no_progress_reason,
        )

    if no_progress_reason == "snapshot_failed":
        return FailureClassification(
            error_class=ErrorClass.INFRA_ERROR,
            error_type=normalized_type or "SnapshotUnavailable",
            failure_stage=failure_stage or "post_run_snapshot",
            is_infra_failure=True,
            is_model_failure=False,
            is_scene_available=scene_available,
            no_progress_reason=no_progress_reason,
        )

    if normalized_type in _SOFT_SUCCESS_TYPES:
        return FailureClassification(
            error_class=ErrorClass.SOFT_SUCCESS_DIAGNOSTIC,
            error_type=normalized_type,
            failure_stage=failure_stage,
            is_model_failure=False,
            is_agent_failure=True,
            is_scene_available=scene_available,
            scene_passed_before_error=True,
            diagnostic_only=True,
            no_progress_reason=no_progress_reason,
        )

    if is_infra_parse_error(normalized_type, failure_stage):
        return FailureClassification(
            error_class=ErrorClass.INFRA_ERROR,
            error_type=normalized_type,
            failure_stage=failure_stage,
            is_infra_failure=True,
            is_model_failure=False,
            is_scene_available=scene_available,
            scene_passed_before_error=scene_passed_before,
            no_progress_reason=no_progress_reason,
        )

    if normalized_type in _INFRA_ERROR_TYPES:
        return FailureClassification(
            error_class=ErrorClass.INFRA_ERROR,
            error_type=normalized_type,
            failure_stage=failure_stage,
            is_infra_failure=True,
            is_model_failure=False,
            is_scene_available=scene_available,
            scene_passed_before_error=scene_passed_before,
            no_progress_reason=no_progress_reason,
        )

    if normalized_type in _TOOL_RUNTIME_ERROR_TYPES:
        return FailureClassification(
            error_class=ErrorClass.TOOL_RUNTIME_ERROR,
            error_type=normalized_type,
            failure_stage=failure_stage,
            is_tool_runtime_failure=True,
            is_model_failure=False,
            is_infra_failure=False,
            is_scene_available=scene_available,
            scene_passed_before_error=scene_passed_before,
            no_progress_reason=no_progress_reason,
        )

    if normalized_type in _AGENT_ERROR_TYPES:
        return FailureClassification(
            error_class=ErrorClass.AGENT_ERROR,
            error_type=normalized_type,
            failure_stage=failure_stage,
            is_agent_failure=True,
            is_model_failure=True,
            is_scene_available=scene_available,
            scene_passed_before_error=scene_passed_before,
            no_progress_reason=no_progress_reason,
        )

    if validation_issues and run_status in {"failed", "error", "passed"} and scene_status == "failed":
        return FailureClassification(
            error_class=ErrorClass.VALIDATION_FAILURE,
            error_type=normalized_type or "validation_failed",
            failure_stage=failure_stage or "validation",
            is_validation_failure=True,
            is_model_failure=runtime_healthy,
            is_scene_available=scene_available,
            scene_passed_before_error=False,
            no_progress_reason=no_progress_reason,
        )

    if scene_status == "failed" and not normalized_type:
        return FailureClassification(
            error_class=ErrorClass.VALIDATION_FAILURE,
            error_type="validation_failed",
            failure_stage=failure_stage or "validation",
            is_validation_failure=True,
            is_model_failure=runtime_healthy,
            is_scene_available=True,
            no_progress_reason=no_progress_reason,
        )

    if normalized_type:
        return FailureClassification(
            error_class=ErrorClass.AGENT_ERROR if normalized_type == "UnclassifiedError" else None,
            error_type=normalized_type,
            failure_stage=failure_stage,
            is_model_failure=normalized_type not in _INFRA_ERROR_TYPES | _TOOL_RUNTIME_ERROR_TYPES,
            is_scene_available=scene_available,
            scene_passed_before_error=scene_passed_before,
            no_progress_reason=no_progress_reason,
        )

    return FailureClassification(
        is_scene_available=scene_available,
        scene_passed_before_error=scene_passed_before,
        no_progress_reason=no_progress_reason,
    )


def enrich_structured_error(
    structured_error: dict[str, Any] | None,
    *,
    scene_status: str | None = None,
    run_status: str | None = None,
    scene_passed_but_agent_error: bool = False,
    early_stop_reason: str | None = None,
    validation_issues: list[Any] | None = None,
    no_progress_reason: str | None = None,
    runtime_healthy: bool = True,
) -> dict[str, Any] | None:
    """Добавляет error_class и is_* флаги к structured_error."""
    if not structured_error:
        if scene_passed_but_agent_error:
            classification = classify_failure(
                error_type="ScenePassedButAgentError",
                scene_status=scene_status,
                run_status=run_status,
                scene_passed_but_agent_error=True,
                early_stop_reason=early_stop_reason,
                no_progress_reason=no_progress_reason,
            )
            return classification.model_dump(mode="json")
        return None

    error_type = str(structured_error.get("error_type") or "")
    failure_stage = structured_error.get("failure_stage")
    if isinstance(failure_stage, dict):
        failure_stage = failure_stage.get("value") or str(failure_stage)
    failure_stage = str(failure_stage) if failure_stage else None

    classification = classify_failure(
        error_type=error_type,
        failure_stage=failure_stage,
        scene_status=scene_status,
        run_status=run_status,
        scene_passed_but_agent_error=scene_passed_but_agent_error,
        early_stop_reason=early_stop_reason,
        validation_issues=validation_issues,
        no_progress_reason=no_progress_reason,
        runtime_healthy=runtime_healthy,
    )
    enriched = dict(structured_error)
    enriched.update(classification.model_dump(mode="json", exclude_none=True))
    return enriched


def error_class_for_type(error_type: str | None) -> ErrorClass | None:
    """Возвращает error_class для известного error_type."""
    normalized = _normalize_error_type(error_type)
    if not normalized:
        return None
    if normalized in _INFRA_ERROR_TYPES:
        return ErrorClass.INFRA_ERROR
    if normalized in _TOOL_RUNTIME_ERROR_TYPES:
        return ErrorClass.TOOL_RUNTIME_ERROR
    if normalized in _AGENT_ERROR_TYPES:
        return ErrorClass.AGENT_ERROR
    if normalized in _SOFT_SUCCESS_TYPES:
        return ErrorClass.SOFT_SUCCESS_DIAGNOSTIC
    return None


_POST_PASS_AGENT_ERRORS = frozenset({
    "ReactInvalidAction",
    "ReactNoProgress",
    "ReactMaxSteps",
})


def is_soft_success_diagnostic(
    *,
    error_class: str | None = None,
    diagnostic_only: bool = False,
) -> bool:
    """True, если run относится к post-validation soft diagnostic."""
    if diagnostic_only:
        return True
    return str(error_class or "").strip() == ErrorClass.SOFT_SUCCESS_DIAGNOSTIC.value


def is_hard_model_failure(
    *,
    is_model_failure: bool,
    is_infra_failure: bool = False,
    error_class: str | None = None,
    diagnostic_only: bool = False,
    pass_type: str | None = None,
    scene_status: str | None = None,
    error_type: str | None = None,
) -> bool:
    """True только для жёстких model/agent failures, без soft diagnostic."""
    if not is_model_failure or is_infra_failure:
        return False
    if is_soft_success_diagnostic(error_class=error_class, diagnostic_only=diagnostic_only):
        return False
    if (
        str(pass_type or "").strip() == "soft_pass"
        and str(scene_status or "").strip() == "passed"
        and str(error_type or "").strip() in _POST_PASS_AGENT_ERRORS
    ):
        return False
    return True


def is_infra_error_type(error_type: str | None) -> bool:
    normalized = _normalize_error_type(error_type)
    return normalized in _INFRA_ERROR_TYPES if normalized else False


def is_transient_infra_error_type(error_type: str | None) -> bool:
    normalized = _normalize_error_type(error_type)
    return normalized in _TRANSIENT_INFRA_ERROR_TYPES if normalized else False


def is_infra_parse_error(error_type: str | None, failure_stage: str | None) -> bool:
    normalized = _normalize_error_type(error_type)
    if normalized != "InvalidJsonResponse":
        return False
    stage = str(failure_stage or "").strip().lower()
    return stage in _INFRA_PARSE_STAGES


def _normalize_error_type(error_type: str | None) -> str | None:
    if not error_type:
        return None
    value = str(error_type).strip()
    if not value or value.lower() in {"null", "none"}:
        return None
    if value.startswith("ControlledErrorType."):
        value = value.split(".", 1)[1]
    aliases = {
        "BLENDER_SOCKET_UNAVAILABLE": "BlenderSocketUnavailable",
        "TOOL_TIMEOUT": "ToolTimeout",
        "RESET_SCENE_FAILED": "ResetSceneFailed",
        "SNAPSHOT_UNAVAILABLE": "SnapshotUnavailable",
        "INVALID_TOOL_RESPONSE": "InvalidToolResponse",
        "BLENDER_RUNTIME_ERROR": "BlenderRuntimeError",
    }
    return aliases.get(value, value)
