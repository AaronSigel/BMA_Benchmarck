from benchmark.agent.models import AgentConfig, AgentStrategyName, LlmConfig
from benchmark.agent.prompts import (
    PromptBuilder,
    build_plan_prompt,
    build_react_prompt_context,
    build_system_prompt,
    build_task_prompt,
    build_tool_result_message,
)


def make_agent_config(**kwargs: object) -> AgentConfig:
    data = {
        "agent_id": "agent-1",
        "strategy": AgentStrategyName.REACT,
        "llm": LlmConfig(provider="mock", model="mock"),
    }
    data.update(kwargs)
    return AgentConfig(**data)


def test_system_prompt_includes_allowed_tools_and_rules() -> None:
    prompt = build_system_prompt(
        make_agent_config(),
        "minimal",
        [
            {"name": "create_object", "description": "Create object"},
            {"name": "get_scene_info", "description": "Inspect scene"},
        ],
    )

    assert "create_object" in prompt
    assert "get_scene_info" in prompt
    assert "Use only the MCP tools listed below" in prompt
    assert "Return tool_calls" in prompt
    assert "JSON action" in prompt
    assert "inspection tools" in prompt
    assert "BenchmarkTask.prompt" in prompt


def test_no_python_prompt_filters_and_forbids_execute_blender_code() -> None:
    prompt = build_system_prompt(
        make_agent_config(),
        "no_python",
        [
            {"name": "create_object"},
            {"name": "execute_blender_code"},
        ],
    )

    assert "Do not use execute_blender_code" in prompt
    assert '"name": "execute_blender_code"' not in prompt
    assert "create_object" in prompt


def test_prompt_filters_external_asset_tools_when_profile_disallows_them() -> None:
    prompt = build_system_prompt(
        make_agent_config(),
        "minimal",
        [
            {"name": "create_object"},
            {"name": "download_asset"},
        ],
    )

    assert "download_asset" not in prompt
    assert "Do not use external asset tools" in prompt


def test_task_prompt_includes_task_prompt() -> None:
    prompt = build_task_prompt({"id": "task-1", "prompt": "Create a red cube"})

    assert "Task ID: task-1" in prompt
    assert "BenchmarkTask.prompt: Create a red cube" in prompt


def test_prompt_builder_does_not_include_secrets() -> None:
    prompt = PromptBuilder().build_system_prompt(
        make_agent_config(system_prompt_template="api_key=secret-value\nUse tools."),
        "minimal",
        [
            {"name": "create_object", "api_key_env": "SECRET_ENV"},
            {"name": "inspect_scene", "metadata": {"token": "secret-token"}},
        ],
    )
    task_prompt = build_task_prompt(
        {"id": "task-1", "prompt": "Do work", "api_key": "secret-value"}
    )

    assert "secret-value" not in prompt
    assert "SECRET_ENV" not in prompt
    assert "secret-token" not in prompt
    assert "secret-value" not in task_prompt
    assert "[redacted]" in prompt
    assert "[redacted]" in task_prompt


def test_react_plan_and_tool_result_prompts() -> None:
    react_prompt = build_react_prompt_context(
        {"id": "task-1", "prompt": "Create a cube"},
        [{"tool": "get_scene_info", "result": "empty scene"}],
    )
    plan_prompt = build_plan_prompt({"id": "task-1", "prompt": "Create a cube"})
    tool_result_message = build_tool_result_message(
        "get_scene_info",
        {"objects": [], "token": "secret-token"},
    )

    assert "Use a ReAct loop" in react_prompt
    assert "empty scene" in react_prompt
    assert "Create a concise execution plan" in plan_prompt
    assert "Tool result for get_scene_info" in tool_result_message
    assert "secret-token" not in tool_result_message
