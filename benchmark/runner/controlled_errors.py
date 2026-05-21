from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ControlledErrorType(str, Enum):
    REACT_MAX_STEPS = "ReactMaxSteps"
    LLM_PARSE_ERROR = "LlmParseError"
    INVALID_TOOL_CALL = "InvalidToolCall"
    BLENDER_SOCKET_UNAVAILABLE = "BlenderSocketUnavailable"
    REPORT_BUILD_ERROR = "ReportBuildError"
    PREFLIGHT_CHECK_FAILED = "PreflightCheckFailed"
    TOOL_TIMEOUT = "ToolTimeout"
    INVALID_TOOL_RESPONSE = "InvalidToolResponse"
    BLENDER_RUNTIME_ERROR = "BlenderRuntimeError"
    LLM_PROVIDER_ERROR = "LlmProviderError"
    SNAPSHOT_UNAVAILABLE = "SnapshotUnavailable"
    VALIDATION_UNAVAILABLE = "ValidationUnavailable"
    RESET_SCENE_FAILED = "ResetSceneFailed"
    EXPORT_UNAVAILABLE = "ExportUnavailable"
    UNCLASSIFIED = "UnclassifiedError"


class ControlledErrorSource(str, Enum):
    AGENT = "agent"
    TOOL = "tool"
    BLENDER = "blender"
    MCP = "mcp"
    PROVIDER = "provider"
    VALIDATOR = "validator"
    REPORTING = "reporting"
    PREFLIGHT = "preflight"


class ControlledFailureStage(str, Enum):
    PREFLIGHT = "preflight"
    RESET_SCENE = "reset_scene"
    PRE_RUN_SNAPSHOT = "pre_run_snapshot"
    AGENT_EXECUTION = "agent_execution"
    TOOL_CALL = "tool_call"
    POST_RUN_SNAPSHOT = "post_run_snapshot"
    VALIDATION = "validation"
    EXPORT = "export"
    ANALYSIS = "analysis"
    REPORT_BUILD = "report_build"


class ControlledError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_type: ControlledErrorType
    message: str
    source: ControlledErrorSource
    recoverable: bool = True
    failure_stage: ControlledFailureStage
    raw_error: str


def normalize_error(
    error: BaseException | str,
    *,
    source: str | ControlledErrorSource | None = None,
    failure_stage: str | ControlledFailureStage | None = None,
) -> ControlledError:
    raw = str(error)
    text = raw.lower()
    explicit_source = _source(source)

    error_type = ControlledErrorType.UNCLASSIFIED
    inferred_source = explicit_source or ControlledErrorSource.AGENT
    inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    recoverable = True

    if "react strategy reached max_steps" in text or "max_steps" in text or "step limit" in text:
        error_type = ControlledErrorType.REACT_MAX_STEPS
        inferred_source = explicit_source or ControlledErrorSource.AGENT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "no tool call or json action returned by llm" in text or "failed to parse" in text:
        error_type = ControlledErrorType.LLM_PARSE_ERROR
        inferred_source = explicit_source or ControlledErrorSource.AGENT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "reset" in text and "scene" in text:
        error_type = ControlledErrorType.RESET_SCENE_FAILED
        inferred_source = explicit_source or ControlledErrorSource.BLENDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.RESET_SCENE
    elif "pre-run scene snapshot could not be collected" in text or "snapshot" in text:
        error_type = ControlledErrorType.SNAPSHOT_UNAVAILABLE
        inferred_source = explicit_source or ControlledErrorSource.BLENDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.PRE_RUN_SNAPSHOT
    elif "no response from blender socket" in text or ("socket" in text and ("unavailable" in text or "no response" in text or "refused" in text)):
        error_type = ControlledErrorType.BLENDER_SOCKET_UNAVAILABLE
        inferred_source = explicit_source or ControlledErrorSource.BLENDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_CALL
    elif "validation" in text or "validator" in text:
        error_type = ControlledErrorType.VALIDATION_UNAVAILABLE
        inferred_source = explicit_source or ControlledErrorSource.VALIDATOR
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.VALIDATION
    elif "export" in text or ".glb" in text or ".blend" in text:
        error_type = ControlledErrorType.EXPORT_UNAVAILABLE
        inferred_source = explicit_source or ControlledErrorSource.TOOL
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.EXPORT
    elif "invalid json" in text or "jsondecode" in text:
        error_type = ControlledErrorType.INVALID_TOOL_RESPONSE
        inferred_source = explicit_source or ControlledErrorSource.TOOL
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_CALL
    elif "timeout" in text or "timed out" in text:
        if "openrouter" in text or "provider" in text or "llm" in text:
            error_type = ControlledErrorType.LLM_PROVIDER_ERROR
            inferred_source = explicit_source or ControlledErrorSource.PROVIDER
            inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
        else:
            error_type = ControlledErrorType.TOOL_TIMEOUT
            inferred_source = explicit_source or ControlledErrorSource.TOOL
            inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_CALL
    elif "openrouter" in text or "provider" in text or "api key" in text or "llm" in text:
        error_type = ControlledErrorType.LLM_PROVIDER_ERROR
        inferred_source = explicit_source or ControlledErrorSource.PROVIDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "invalid tool call" in text or "tool is not allowed" in text or "unknown tool" in text:
        error_type = ControlledErrorType.INVALID_TOOL_CALL
        inferred_source = explicit_source or ControlledErrorSource.TOOL
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_CALL
    elif "report" in text:
        error_type = ControlledErrorType.REPORT_BUILD_ERROR
        inferred_source = explicit_source or ControlledErrorSource.REPORTING
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.REPORT_BUILD
    elif "preflight" in text:
        error_type = ControlledErrorType.PREFLIGHT_CHECK_FAILED
        inferred_source = explicit_source or ControlledErrorSource.PREFLIGHT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.PREFLIGHT
    elif "blender" in text or "crash" in text:
        error_type = ControlledErrorType.BLENDER_RUNTIME_ERROR
        inferred_source = explicit_source or ControlledErrorSource.BLENDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_CALL
        recoverable = False
    elif "tool" in text or "unknown" in text:
        error_type = ControlledErrorType.INVALID_TOOL_RESPONSE
        inferred_source = explicit_source or ControlledErrorSource.TOOL
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_CALL

    return ControlledError(
        error_type=error_type,
        message=raw,
        recoverable=recoverable,
        source=inferred_source,
        failure_stage=inferred_stage,
        raw_error=raw,
    )


def controlled_error_payload(
    error: BaseException | str,
    *,
    source: str | ControlledErrorSource | None = None,
    failure_stage: str | ControlledFailureStage | None = None,
) -> dict[str, Any]:
    return normalize_error(error, source=source, failure_stage=failure_stage).model_dump(mode="json")


def _source(value: str | ControlledErrorSource | None) -> ControlledErrorSource | None:
    if value is None:
        return None
    if isinstance(value, ControlledErrorSource):
        return value
    try:
        return ControlledErrorSource(value)
    except ValueError:
        return None


def _stage(value: str | ControlledFailureStage | None) -> ControlledFailureStage | None:
    if value is None:
        return None
    if isinstance(value, ControlledFailureStage):
        return value
    try:
        return ControlledFailureStage(value)
    except ValueError:
        return None
