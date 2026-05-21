"""Tests for _resolve_max_steps priority in ReactStrategy."""
from __future__ import annotations

import pytest

from benchmark.agent.models import AgentConfig, AgentStrategyName, LlmConfig
from benchmark.agent.strategies.react import _resolve_max_steps


def _config(
    max_steps: int = 20,
    max_steps_by_category: dict | None = None,
) -> AgentConfig:
    return AgentConfig(
        agent_id="test",
        strategy=AgentStrategyName.REACT,
        llm=LlmConfig(provider="mock", model="mock"),
        mcp_profile="minimal",
        max_steps=max_steps,
        max_steps_by_category=max_steps_by_category or {},
    )


def test_config_max_steps_overrides_category_default() -> None:
    """config.max_steps takes priority over _DEFAULT_CATEGORY_MAX_STEPS."""
    config = _config(max_steps=15)
    task = {"id": "export_001", "category": "export"}
    result = _resolve_max_steps(task, config)
    assert result == 15, f"expected 15, got {result}"


def test_max_steps_by_category_overrides_config_max_steps() -> None:
    """max_steps_by_category overrides both config.max_steps and category defaults."""
    config = _config(max_steps=15, max_steps_by_category={"export": 12})
    task = {"id": "export_001", "category": "export"}
    result = _resolve_max_steps(task, config)
    assert result == 12, f"expected 12, got {result}"


def test_category_default_used_only_when_config_missing_explicit_steps() -> None:
    """Category default is used only when max_steps_by_category is empty and max_steps is the model default."""
    from benchmark.agent.strategies.react import _DEFAULT_CATEGORY_MAX_STEPS

    config = _config(max_steps=20, max_steps_by_category={})
    task = {"id": "geometry_001", "category": "geometry"}
    result = _resolve_max_steps(task, config)
    # max_steps=20 wins over _DEFAULT_CATEGORY_MAX_STEPS["geometry"]=4
    assert result == 20


def test_fallback_20_when_no_category_and_no_config() -> None:
    """Fallback to 20 when category not in defaults and max_steps is default."""
    config = _config(max_steps=20, max_steps_by_category={})
    task = {"id": "unknown_task", "category": "unknown"}
    result = _resolve_max_steps(task, config)
    assert result == 20


def test_react_openrouter_effective_max_steps_from_matrix() -> None:
    """For diagnostic_repeat_gemini_v5-style config, effective_max_steps=20 for all categories."""
    config = _config(max_steps=20)
    for category in ("geometry", "materials", "lighting", "camera", "export", "composition"):
        task = {"id": f"{category}_001", "category": category}
        result = _resolve_max_steps(task, config)
        assert result == 20, f"category={category}: expected 20, got {result}"


def test_missing_category_falls_back_gracefully() -> None:
    config = _config(max_steps=20)
    task = {"id": "task_no_category"}
    result = _resolve_max_steps(task, config)
    assert result == 20
