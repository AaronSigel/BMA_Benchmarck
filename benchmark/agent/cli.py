from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from benchmark.env import load_project_dotenv
from benchmark.agent.config_loader import load_agent_config
from benchmark.agent.errors import AgentError
from benchmark.agent.llm import LlmResponse, LlmToolCall, MockLlmClient
from benchmark.agent.models import (
    AgentConfig,
    AgentStrategyName,
    AgentStepType,
    LlmProvider,
    RemoteAgentProvider,
)
from benchmark.agent.runtime import run_task
from benchmark.agent.tool_executor import MockToolExecutor
from benchmark.agent.trace import read_agent_trace, summarize_trace
from benchmark.tasks.loader import TaskLoadError, load_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent runtime utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run an agent on one benchmark task.")
    run.add_argument("--task", type=Path, required=True, help="Path to a benchmark task YAML file.")
    run.add_argument("--agent-config", type=Path, required=True, help="Path to an agent config YAML file.")
    run.add_argument("--output-dir", type=Path, required=True, help="Directory for agent artifacts.")

    trace_summary = subparsers.add_parser("trace-summary", help="Print a compact agent trace summary.")
    trace_summary.add_argument("--trace", type=Path, required=True, help="Path to agent_trace.json.")

    subparsers.add_parser("list-strategies", help="List supported agent strategies.")
    subparsers.add_parser("list-providers", help="List supported LLM and remote-agent providers.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_project_dotenv()
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args.task, args.agent_config, args.output_dir)
    if args.command == "trace-summary":
        return _trace_summary(args.trace)
    if args.command == "list-strategies":
        print("\n".join(strategy.value for strategy in AgentStrategyName))
        return 0
    if args.command == "list-providers":
        print(_format_providers())
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def _run(task_path: Path, agent_config_path: Path, output_dir: Path) -> int:
    try:
        task = load_task(task_path)
        config = load_agent_config(agent_config_path)
        result = run_task(
            task,
            config,
            MockToolExecutor(),
            output_dir,
            llm_client=_default_mock_llm_client(config),
        )
    except (AgentError, OSError, TaskLoadError, ValidationError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(_format_run_result(result.trace_path, result.ok, result.error))
    return 0 if result.ok else 1


def _trace_summary(trace_path: Path) -> int:
    try:
        trace = read_agent_trace(trace_path)
    except (AgentError, OSError, ValidationError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    summary = summarize_trace(trace)
    lines = [
        f"run_id: {summary['run_id']}",
        f"task_id: {summary['task_id']}",
        f"agent_id: {summary['agent_id']}",
        f"strategy: {summary['strategy']}",
        f"success: {summary['success']}",
        f"steps: {summary['steps_count']}",
        f"tools: {summary['tool_calls_count']}",
        f"errors: {summary['errors_count']}",
    ]
    tool_names = [
        step.tool_name
        for step in trace.steps
        if step.step_type == AgentStepType.TOOL_CALL and step.tool_name
    ]
    if tool_names:
        lines.append("tool_names: " + ", ".join(tool_names))
    errors = [step.error for step in trace.steps if step.error]
    if trace.error and trace.error not in errors:
        errors.append(trace.error)
    if errors:
        lines.extend(f"error: {error}" for error in errors)
    print("\n".join(lines))
    return 0


def _format_providers() -> str:
    lines = ["llm:"]
    lines.extend(f"  {provider.value}" for provider in LlmProvider)
    lines.append("remote_agent:")
    lines.extend(f"  {provider.value}" for provider in RemoteAgentProvider)
    return "\n".join(lines)


def _format_run_result(trace_path: Path | None, ok: bool, error: str | None) -> str:
    lines = [
        f"ok: {str(ok).lower()}",
        f"trace_path: {trace_path or ''}",
    ]
    if error:
        lines.append(f"error: {error}")
    return "\n".join(lines)


def _default_mock_llm_client(config: AgentConfig) -> MockLlmClient | None:
    if config.llm is None or config.llm.provider != LlmProvider.MOCK:
        return None
    if config.strategy == AgentStrategyName.PLAN_AND_EXECUTE:
        return MockLlmClient(
            [
                LlmResponse(
                    content=json.dumps(
                        {
                            "plan": [
                                {
                                    "step": 1,
                                    "description": "Inspect scene for CLI dry run.",
                                    "tool": "get_scene_info",
                                    "arguments": {},
                                }
                            ]
                        }
                    )
                )
            ]
        )
    if config.strategy == AgentStrategyName.DIRECT_TOOL_CALLING:
        return MockLlmClient(
            [
                LlmResponse(
                    tool_calls=[
                        LlmToolCall(name="get_scene_info", arguments={}),
                    ]
                )
            ]
        )
    return MockLlmClient([LlmResponse(content="Mock agent run completed.")])


if __name__ == "__main__":
    raise SystemExit(main())
