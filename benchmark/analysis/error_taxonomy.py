from __future__ import annotations

import re
from collections import Counter

from benchmark.agent.models import AgentStep, AgentStepType, AgentTrace
from benchmark.analysis.models import ErrorCategory, ErrorRecord
from benchmark.analysis.trace_reader import RunArtifactBundle
from benchmark.validation.models import ValidationIssue

# ---------------------------------------------------------------------------
# Trace-error classification: ordered from most-specific to least-specific
# ---------------------------------------------------------------------------

_TRACE_RULES: list[tuple[ErrorCategory, re.Pattern[str]]] = [
    # tool_disabled: "not allowed", "disabled", ToolDisabledError message
    (ErrorCategory.TOOL_DISABLED,
     re.compile(r"not allowed|is disabled|tool.*disabled|disabled.*tool|ToolDisabledError", re.I)),
    # tool_unknown: tool not found / unknown
    (ErrorCategory.TOOL_UNKNOWN,
     re.compile(r"(tool|command).*(not found|unknown|unrecognised|unrecognized|unavailable)|UnknownToolError", re.I)),
    # agent_step_limit: max_steps reached
    (ErrorCategory.AGENT_STEP_LIMIT,
     re.compile(r"max.?steps|step.?limit.?reached|AgentStepLimitError", re.I)),
    # llm_timeout: LLM or agent timeout
    (ErrorCategory.LLM_TIMEOUT,
     re.compile(r"timeout|timed.?out|AgentTimeoutError|LlmTimeout", re.I)),
    # llm_parse_error: failed to parse LLM response
    (ErrorCategory.LLM_PARSE_ERROR,
     re.compile(r"(parse|parsing|json.*decode|failed to parse|LlmResponseParseError|invalid.*json|unexpected.*token)", re.I)),
    # mcp_connection_error: socket / connection failures
    (ErrorCategory.MCP_CONNECTION_ERROR,
     re.compile(r"(connection|socket|mcp).*(error|fail|refused|unavailable)|BlenderSocketUnavailable|McpServerStart", re.I)),
    # remote_agent_error
    (ErrorCategory.REMOTE_AGENT_ERROR,
     re.compile(r"remote.?agent|RemoteAgentError|RemoteAgentTimeout", re.I)),
    # tool_invalid_arguments: bad arguments / missing fields
    (ErrorCategory.TOOL_INVALID_ARGUMENTS,
     re.compile(r"(argument|parameter|field|key).*(invalid|missing|required|unexpected)|invalid.*(argument|param|input)|missing.*(field|key|param)", re.I)),
    # tool_runtime_error: execution / blender / script errors
    (ErrorCategory.TOOL_RUNTIME_ERROR,
     re.compile(r"(execution|blender|script|runtime).*(error|fail|exception)|traceback|ToolInvocationError", re.I)),
]

# ---------------------------------------------------------------------------
# Validation-issue classification: keyed by issue code prefix/substring
# ---------------------------------------------------------------------------

_ISSUE_CODE_RULES: list[tuple[ErrorCategory, re.Pattern[str]]] = [
    (ErrorCategory.SCENE_OBJECT_MISSING,
     re.compile(r"^object_missing$|^object_type_mismatch", re.I)),
    (ErrorCategory.SCENE_TRANSFORM_MISMATCH,
     re.compile(r"primitive_mismatch|object_missing_for_transform|location_mismatch|rotation_mismatch|scale_mismatch|dimensions_mismatch|transform", re.I)),
    (ErrorCategory.SCENE_MATERIAL_MISMATCH,
     re.compile(r"material_|object_missing_for_material|object_material_", re.I)),
    (ErrorCategory.SCENE_LIGHT_MISMATCH,
     re.compile(r"light_", re.I)),
    (ErrorCategory.SCENE_CAMERA_MISMATCH,
     re.compile(r"camera_|active_camera_", re.I)),
    (ErrorCategory.SCENE_EXPORT_MISSING,
     re.compile(r"export_", re.I)),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_trace_error(step: AgentStep) -> ErrorCategory:
    """Classify a step's error message into an ErrorCategory."""
    message = step.error or ""
    for category, pattern in _TRACE_RULES:
        if pattern.search(message):
            return category
    return ErrorCategory.UNKNOWN_ERROR


def classify_validation_issue(issue: ValidationIssue) -> ErrorCategory:
    """Classify a ValidationIssue by its code into an ErrorCategory."""
    code = issue.code or ""
    for category, pattern in _ISSUE_CODE_RULES:
        if pattern.search(code):
            return category
    # Fallback: classify by message
    message = issue.message or ""
    for category, pattern in _ISSUE_CODE_RULES:
        if pattern.search(message):
            return category
    return ErrorCategory.UNKNOWN_ERROR


def extract_errors(trace: AgentTrace) -> list[ErrorRecord]:
    """Extract and classify all error steps from a trace."""
    records: list[ErrorRecord] = []
    for step in trace.steps:
        if step.error:
            records.append(
                ErrorRecord(
                    run_id=trace.run_id,
                    task_id=trace.task_id,
                    step_index=step.step_index,
                    category=classify_trace_error(step),
                    message=step.error,
                    tool_name=step.tool_name if step.step_type == AgentStepType.TOOL_CALL else None,
                )
            )
    return records


def aggregate_errors(run_bundle: RunArtifactBundle) -> dict[str, int]:
    """Aggregate error counts from both the agent trace and validation result.

    Returns a dict mapping ErrorCategory value → count.
    """
    counts: Counter[str] = Counter()

    # Trace errors
    if run_bundle.agent_trace is not None:
        for record in extract_errors(run_bundle.agent_trace):
            counts[record.category.value] += 1

    # Validation issues
    if run_bundle.validation_result is not None:
        val = run_bundle.validation_result
        seen: set[tuple[str, str | None, str | None]] = set()
        for issue in val.issues:
            key = _issue_key(issue)
            if key in seen:
                continue
            seen.add(key)
            counts[classify_validation_issue(issue).value] += 1
        for validator in val.validators:
            for issue in validator.issues:
                key = _issue_key(issue)
                if key in seen:
                    continue
                seen.add(key)
                counts[classify_validation_issue(issue).value] += 1

    return dict(counts)


def _issue_key(issue: ValidationIssue) -> tuple[str, str | None, str | None]:
    return (issue.code, issue.expected_path, issue.actual_path)


def summarize_errors(records: list[ErrorRecord]) -> dict[str, int]:
    """Return a category → count mapping from a list of ErrorRecord."""
    return dict(Counter(r.category.value for r in records))
