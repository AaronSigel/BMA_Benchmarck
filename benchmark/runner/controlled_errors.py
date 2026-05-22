from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ControlledErrorType(str, Enum):
    REACT_MAX_STEPS = "ReactMaxSteps"
    REACT_NO_PROGRESS = "ReactNoProgress"
    REACT_INVALID_ACTION = "ReactInvalidAction"
    REACT_NON_STRICT_RESPONSE = "ReactNonStrictResponse"
    REACT_BLOCKED_EXPORT = "ReactBlockedExport"
    DIRECT_NO_ACTION = "DirectNoAction"
    PLAN_PARSE_ERROR = "PlanParseError"
    PLAN_SCHEMA_ERROR = "PlanSchemaError"
    LLM_PARSE_ERROR = "LlmParseError"
    INVALID_TOOL_CALL = "InvalidToolCall"
    INVALID_ARGUMENTS = "InvalidArguments"
    BLENDER_SOCKET_UNAVAILABLE = "BlenderSocketUnavailable"
    BLENDER_SOCKET_NO_RESPONSE = "BlenderSocketNoResponse"
    EMPTY_SOCKET_RESPONSE = "EmptySocketResponse"
    BLENDER_WORKER_UNHEALTHY = "BlenderWorkerUnhealthy"
    REPORT_BUILD_ERROR = "ReportBuildError"
    PREFLIGHT_CHECK_FAILED = "PreflightCheckFailed"
    TOOL_TIMEOUT = "ToolTimeout"
    SOCKET_TIMEOUT = "SocketTimeout"
    SOCKET_ERROR = "SocketError"
    INVALID_TOOL_RESPONSE = "InvalidToolResponse"
    INVALID_JSON_RESPONSE = "InvalidJsonResponse"
    EMPTY_TOOL_RESPONSE = "EmptyToolResponse"
    TOOL_EXECUTION_FAILED = "ToolExecutionFailed"
    BLENDER_RUNTIME_ERROR = "BlenderRuntimeError"
    LLM_PROVIDER_ERROR = "LlmProviderError"
    SNAPSHOT_UNAVAILABLE = "SnapshotUnavailable"
    VALIDATION_UNAVAILABLE = "ValidationUnavailable"
    RESET_SCENE_FAILED = "ResetSceneFailed"
    EXPORT_UNAVAILABLE = "ExportUnavailable"
    SCENE_PASSED_BUT_AGENT_ERROR = "ScenePassedButAgentError"
    SCENE_PASSED_WITH_WARNING = "ScenePassedWithWarning"
    SCENE_PASSED_AFTER_REPAIR = "ScenePassedAfterRepair"
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
    SOCKET_RESPONSE = "socket_response"
    TOOL_RESPONSE_PARSE = "tool_response_parse"
    SNAPSHOT_COLLECTION = "snapshot_collection"
    SNAPSHOT_NORMALIZATION = "snapshot_normalization"
    EXECUTE_CODE_FALLBACK = "execute_code_fallback"
    BLENDER_PYTHON_EXECUTION = "blender_python_execution"


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

    if "reactblockedexport" in text or "export_blocked" in text:
        error_type = ControlledErrorType.REACT_BLOCKED_EXPORT
        inferred_source = explicit_source or ControlledErrorSource.AGENT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "reactinvalidaction" in text or "invalid react action" in text:
        error_type = ControlledErrorType.REACT_INVALID_ACTION
        inferred_source = explicit_source or ControlledErrorSource.AGENT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "reactnoprogress" in text or "no_progress_detected" in text or "no progress detected" in text:
        error_type = ControlledErrorType.REACT_NO_PROGRESS
        inferred_source = explicit_source or ControlledErrorSource.AGENT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "reactnonstrictresponse" in text:
        error_type = ControlledErrorType.REACT_NON_STRICT_RESPONSE
        inferred_source = explicit_source or ControlledErrorSource.AGENT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "react strategy reached max_steps" in text or "reactmaxsteps" in text or "reached max_steps" in text or "max_steps" in text or "step limit" in text or "agentsteplimit" in text:
        error_type = ControlledErrorType.REACT_MAX_STEPS
        inferred_source = explicit_source or ControlledErrorSource.AGENT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "directnoaction" in text or "direct no action" in text:
        error_type = ControlledErrorType.DIRECT_NO_ACTION
        inferred_source = explicit_source or ControlledErrorSource.AGENT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "plan-and-execute response must be a json object with a plan list" in text:
        error_type = ControlledErrorType.PLAN_PARSE_ERROR
        inferred_source = explicit_source or ControlledErrorSource.AGENT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "plan-and-execute response must contain a non-empty plan list" in text:
        error_type = ControlledErrorType.PLAN_SCHEMA_ERROR
        inferred_source = explicit_source or ControlledErrorSource.AGENT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "plan[" in text and (
        ".step must be" in text
        or ".tool must be" in text
        or ".arguments must be" in text
        or ".description must be" in text
        or "must be an object" in text
    ):
        error_type = ControlledErrorType.PLAN_SCHEMA_ERROR
        inferred_source = explicit_source or ControlledErrorSource.AGENT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "no tool call or json action returned by llm" in text or "failed to parse" in text or "did not include action" in text or "no action found" in text or "llmresponseparseerror" in text or "llmparseerror" in text or "reactnonstrictresponse" in text:
        error_type = ControlledErrorType.LLM_PARSE_ERROR
        inferred_source = explicit_source or ControlledErrorSource.AGENT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "repeated the same action" in text or "repeated action" in text:
        error_type = ControlledErrorType.REACT_INVALID_ACTION
        inferred_source = explicit_source or ControlledErrorSource.AGENT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "duplicate object" in text:
        error_type = ControlledErrorType.INVALID_TOOL_CALL
        inferred_source = explicit_source or ControlledErrorSource.TOOL
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_CALL
    elif "reset" in text and "scene" in text:
        error_type = ControlledErrorType.RESET_SCENE_FAILED
        inferred_source = explicit_source or ControlledErrorSource.BLENDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.RESET_SCENE
    elif "emptysocketresponse" in text or "empty response from blender socket" in text:
        error_type = ControlledErrorType.EMPTY_SOCKET_RESPONSE
        inferred_source = explicit_source or ControlledErrorSource.BLENDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.SOCKET_RESPONSE
    elif "emptytoolresponse" in text or "empty tool response" in text:
        error_type = ControlledErrorType.EMPTY_TOOL_RESPONSE
        inferred_source = explicit_source or ControlledErrorSource.TOOL
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_RESPONSE_PARSE
    elif "invalidjsonresponse" in text or "invalid json response" in text:
        error_type = ControlledErrorType.INVALID_JSON_RESPONSE
        inferred_source = explicit_source or ControlledErrorSource.TOOL
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_RESPONSE_PARSE
    elif "blenderworkerunhealthy" in text or "worker unhealthy" in text:
        error_type = ControlledErrorType.BLENDER_WORKER_UNHEALTHY
        inferred_source = explicit_source or ControlledErrorSource.BLENDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.PREFLIGHT
    elif "pre-run scene snapshot could not be collected" in text or ("snapshot" in text and "not" in text):
        error_type = ControlledErrorType.SNAPSHOT_UNAVAILABLE
        inferred_source = explicit_source or ControlledErrorSource.BLENDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.PRE_RUN_SNAPSHOT
    elif "no response from blender socket" in text or "blendersocketunavailable" in text or "blendersocketnoresponse" in text or ("socket" in text and ("unavailable" in text or "no response" in text or "refused" in text or "error" in text)):
        if "no response" in text:
            error_type = ControlledErrorType.BLENDER_SOCKET_NO_RESPONSE
        else:
            error_type = ControlledErrorType.BLENDER_SOCKET_UNAVAILABLE
        inferred_source = explicit_source or ControlledErrorSource.BLENDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_CALL
    elif "timeout" in text or "timed out" in text or "timed-out" in text:
        if "openrouter" in text or "provider" in text or "llm" in text:
            error_type = ControlledErrorType.LLM_PROVIDER_ERROR
            inferred_source = explicit_source or ControlledErrorSource.PROVIDER
            inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
        else:
            error_type = ControlledErrorType.TOOL_TIMEOUT
            inferred_source = explicit_source or ControlledErrorSource.TOOL
            inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_CALL
    elif "openrouter" in text or "api key" in text or "llm provider" in text or "llmprovider" in text or "401" in text or "403" in text or "rate limit" in text:
        error_type = ControlledErrorType.LLM_PROVIDER_ERROR
        inferred_source = explicit_source or ControlledErrorSource.PROVIDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "invalid json" in text or "jsondecode" in text or "json decode" in text or "invalid json from" in text:
        error_type = ControlledErrorType.INVALID_TOOL_RESPONSE
        inferred_source = explicit_source or ControlledErrorSource.TOOL
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_CALL
    elif "invalid tool call" in text or "tool is not allowed" in text or "unknown tool" in text or "tool not found" in text or "not allowed in this profile" in text or "tooldisablederror" in text:
        error_type = ControlledErrorType.INVALID_TOOL_CALL
        inferred_source = explicit_source or ControlledErrorSource.AGENT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_CALL
    elif "export" in text or ".glb" in text or ".blend" in text:
        error_type = ControlledErrorType.EXPORT_UNAVAILABLE
        inferred_source = explicit_source or ControlledErrorSource.TOOL
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.EXPORT
    elif "validation" in text or "validator" in text:
        error_type = ControlledErrorType.VALIDATION_UNAVAILABLE
        inferred_source = explicit_source or ControlledErrorSource.VALIDATOR
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.VALIDATION
    elif "report" in text:
        error_type = ControlledErrorType.REPORT_BUILD_ERROR
        inferred_source = explicit_source or ControlledErrorSource.REPORTING
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.REPORT_BUILD
    elif "preflight" in text:
        error_type = ControlledErrorType.PREFLIGHT_CHECK_FAILED
        inferred_source = explicit_source or ControlledErrorSource.PREFLIGHT
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.PREFLIGHT
    elif "blender" in text or "crash" in text or "blenderruntimeerror" in text or "bpy_struct" in text:
        error_type = ControlledErrorType.BLENDER_RUNTIME_ERROR
        inferred_source = explicit_source or ControlledErrorSource.BLENDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_CALL
        recoverable = False
    elif "provider" in text or "llm" in text:
        error_type = ControlledErrorType.LLM_PROVIDER_ERROR
        inferred_source = explicit_source or ControlledErrorSource.PROVIDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.AGENT_EXECUTION
    elif "tool" in text or "unknown" in text:
        error_type = ControlledErrorType.INVALID_TOOL_RESPONSE
        inferred_source = explicit_source or ControlledErrorSource.TOOL
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_CALL
    elif "snapshot" in text:
        error_type = ControlledErrorType.SNAPSHOT_UNAVAILABLE
        inferred_source = explicit_source or ControlledErrorSource.BLENDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.PRE_RUN_SNAPSHOT
    elif "connection" in text or "socket" in text or "refused" in text:
        error_type = ControlledErrorType.BLENDER_SOCKET_UNAVAILABLE
        inferred_source = explicit_source or ControlledErrorSource.BLENDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_CALL
    elif "execution" in text or "runtime" in text or "script" in text:
        error_type = ControlledErrorType.BLENDER_RUNTIME_ERROR
        inferred_source = explicit_source or ControlledErrorSource.BLENDER
        inferred_stage = _stage(failure_stage) or ControlledFailureStage.TOOL_CALL
        recoverable = False

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
    enrich: bool = True,
    **classification_kwargs: Any,
) -> dict[str, Any]:
    payload = normalize_error(error, source=source, failure_stage=failure_stage).model_dump(mode="json")
    if enrich:
        from benchmark.runner.error_classification import enrich_structured_error
        return enrich_structured_error(payload, **classification_kwargs) or payload
    return payload


def controlled_error_from_envelope(
    envelope: dict[str, Any],
    *,
    source: str | ControlledErrorSource | None = None,
    failure_stage: str | ControlledFailureStage | None = None,
    **classification_kwargs: Any,
) -> dict[str, Any]:
    """Строит structured_error из MCP tool envelope."""
    error = envelope.get("error") if isinstance(envelope.get("error"), dict) else {}
    error_type = str(error.get("type") or "UnclassifiedError")
    message = str(error.get("message") or envelope.get("error") or "tool failed")
    stage = error.get("stage") or error.get("failure_stage") or failure_stage
    inner = envelope.get("result") if isinstance(envelope.get("result"), dict) else {}
    if not stage and isinstance(inner, dict):
        stage = inner.get("failure_stage") or stage
    payload = normalize_error(
        message,
        source=source or ControlledErrorSource.TOOL,
        failure_stage=stage or failure_stage,
    ).model_dump(mode="json")
    # Переопределяем error_type из envelope, если он известен
    try:
        payload["error_type"] = ControlledErrorType(error_type).value
    except ValueError:
        mapped = _map_envelope_error_type(error_type)
        payload["error_type"] = mapped.value if mapped else error_type
    payload["message"] = message
    payload["raw_error"] = message
    from benchmark.runner.error_classification import enrich_structured_error
    return enrich_structured_error(payload, **classification_kwargs) or payload


def _map_envelope_error_type(error_type: str) -> ControlledErrorType | None:
    mapping = {
        "EmptySocketResponse": ControlledErrorType.EMPTY_SOCKET_RESPONSE,
        "InvalidJsonResponse": ControlledErrorType.INVALID_JSON_RESPONSE,
        "ToolTimeout": ControlledErrorType.TOOL_TIMEOUT,
        "SocketTimeout": ControlledErrorType.SOCKET_TIMEOUT,
        "SocketError": ControlledErrorType.SOCKET_ERROR,
        "ToolError": ControlledErrorType.TOOL_EXECUTION_FAILED,
        "BlenderRuntimeError": ControlledErrorType.BLENDER_RUNTIME_ERROR,
    }
    return mapping.get(error_type)


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
