from pathlib import Path

from benchmark.agent.cli import main


def _parse_trace_path(stdout: str) -> Path:
    """Extract the trace_path value printed by the CLI run command."""
    for line in stdout.splitlines():
        if line.startswith("trace_path:"):
            value = line.split(":", 1)[1].strip()
            return Path(value)
    raise AssertionError(f"trace_path not found in CLI output:\n{stdout}")


def test_agent_cli_run_creates_trace(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "agent_run"

    exit_code = main(
        [
            "run",
            "--task",
            "tasks/geometry/geometry_001_basic_primitives.yaml",
            "--agent-config",
            "configs/agents/mock_agent.yaml",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "trace_path:" in captured.out
    trace_path = _parse_trace_path(captured.out)
    assert trace_path.exists()
    assert trace_path.name == "agent_trace.json"
    assert str(output_dir) in str(trace_path)


def test_agent_cli_trace_summary_prints_steps_tools_errors(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "agent_run"
    assert (
        main(
            [
                "run",
                "--task",
                "tasks/geometry/geometry_001_basic_primitives.yaml",
                "--agent-config",
                "configs/agents/mock_agent.yaml",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    trace_path = _parse_trace_path(capsys.readouterr().out)

    exit_code = main(["trace-summary", "--trace", str(trace_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "steps:" in captured.out
    assert "tools:" in captured.out
    assert "errors:" in captured.out


def test_agent_cli_list_strategies(capsys) -> None:
    assert main(["list-strategies"]) == 0

    captured = capsys.readouterr()
    assert "direct_tool_calling" in captured.out
    assert "react" in captured.out
    assert "plan_and_execute" in captured.out
    assert "remote_agent" in captured.out


def test_agent_cli_list_providers_excludes_ollama(capsys) -> None:
    assert main(["list-providers"]) == 0

    captured = capsys.readouterr()
    assert "openrouter" in captured.out
    assert "openai_compatible" in captured.out
    assert "anthropic" in captured.out
    assert "ollama" not in captured.out
