"""Политика экспериментальной матрицы — единый source of truth для runtime и отчёта."""

from __future__ import annotations

import random
from typing import Any

from benchmark.agent.models import AgentStrategyName
from benchmark.experiments.models import ExperimentMatrix
from benchmark.runner.models import RunConfig

_POLICY_KEYS = (
    "generation_profile",
    "strategy_limits",
    "worker_lifecycle",
    "preflight",
    "readiness_gates",
    "cost",
    "reporting",
    "analysis_policy",
    "acceptance_policy",
    "third_repetition_policy",
    "run_order",
    "scope",
    "model_set",
    "supplementary_model_policy",
)


def resolve_matrix_policy(matrix: ExperimentMatrix) -> dict[str, Any]:
    """Нормализованный snapshot policy из matrix.metadata."""
    meta = matrix.metadata if isinstance(matrix.metadata, dict) else {}
    policy: dict[str, Any] = {
        "matrix_id": matrix.matrix_id,
        "repetitions": matrix.repetitions,
        "expected_runs": meta.get("expected_runs"),
        "report_ready_mvp": meta.get("report_ready_mvp"),
        "strict_matrix_policy": bool(meta.get("strict_matrix_policy", False)),
    }
    for key in _POLICY_KEYS:
        value = meta.get(key)
        if value is not None:
            policy[key] = value
    return policy


def validate_matrix_policy(policy: dict[str, Any], *, planned_runs: int | None = None) -> list[str]:
    """Предупреждения/ошибки валидации policy (для readiness)."""
    issues: list[str] = []
    expected = policy.get("expected_runs")
    if isinstance(expected, int) and planned_runs is not None and expected != planned_runs:
        issues.append(f"planned_runs ({planned_runs}) != expected_runs ({expected})")

    strategy_limits = policy.get("strategy_limits")
    if isinstance(strategy_limits, dict):
        known = {item.value for item in AgentStrategyName}
        unknown = sorted(set(strategy_limits) - known)
        if unknown:
            issues.append(
                f"unknown strategy_limits keys: {', '.join(unknown)}; known: {', '.join(sorted(known))}"
            )

    generation_profile = policy.get("generation_profile")
    if isinstance(generation_profile, dict) and generation_profile.get("apply_to_all_models"):
        for field in ("temperature", "top_p", "max_tokens"):
            if generation_profile.get(field) is None and field != "top_k":
                if field == "max_tokens" and generation_profile.get("apply_to_all_models"):
                    issues.append("generation_profile.max_tokens is required when apply_to_all_models=true")

    return issues


def apply_policy_to_run_metadata(base: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Добавляет policy snapshot в metadata прогона."""
    merged = dict(base)
    merged["matrix_policy"] = policy
    for key in (
        "generation_profile",
        "strategy_limits",
        "worker_lifecycle",
        "cost",
        "preflight",
        "readiness_gates",
    ):
        if key in policy:
            merged[key] = policy[key]
    return merged


def apply_policy_to_experiment_metadata(base: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Добавляет полный policy snapshot в metadata эксперимента."""
    merged = dict(base)
    merged["matrix_policy"] = policy
    for key in _POLICY_KEYS:
        if key in policy:
            merged[key] = policy[key]
    merged.setdefault("repetitions", policy.get("repetitions"))
    merged.setdefault("expected_runs", policy.get("expected_runs"))
    return merged


def apply_generation_profile(llm_config: Any, profile: dict[str, Any] | None) -> Any:
    """Применяет generation_profile к LlmConfig (матрица побеждает agent YAML)."""
    if profile is None or not isinstance(profile, dict):
        return llm_config
    if not profile.get("apply_to_all_models", True):
        return llm_config

    updates: dict[str, Any] = {}
    for field in ("temperature", "top_p", "max_tokens"):
        value = profile.get(field)
        if value is not None:
            updates[field] = value
    if not updates:
        return llm_config
    return llm_config.model_copy(update=updates)


def order_runs(runs: list[RunConfig], run_order: dict[str, Any] | None) -> list[RunConfig]:
    """Перемешивает прогоны по политике run_order (stratified_interleaved)."""
    if not runs or not isinstance(run_order, dict):
        return runs
    mode = str(run_order.get("mode", "")).strip().lower()
    if mode != "stratified_interleaved":
        return runs

    stratify_by = run_order.get("stratify_by")
    if not isinstance(stratify_by, list) or not stratify_by:
        stratify_by = ["model_id", "repetition", "task_id", "agent_id", "mcp_profile"]

    seed = run_order.get("seed", 0)
    rng = random.Random(seed)

    def _key(run: RunConfig) -> tuple:
        meta = run.metadata if isinstance(run.metadata, dict) else {}
        parts: list[Any] = []
        for field in stratify_by:
            if field == "model_id":
                parts.append(meta.get("model_id", ""))
            elif field == "repetition":
                parts.append(meta.get("repetition", 0))
            elif field == "task_id":
                parts.append(run.task_id)
            elif field == "agent_id":
                parts.append(meta.get("agent_id", ""))
            elif field == "mcp_profile":
                parts.append(run.mcp_profile or meta.get("mcp_profile", ""))
            else:
                parts.append(meta.get(field, ""))
        return tuple(parts)

    buckets: dict[tuple, list[RunConfig]] = {}
    for run in runs:
        key = _key(run)
        buckets.setdefault(key, []).append(run)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    keys = list(buckets.keys())
    rng.shuffle(keys)

    ordered: list[RunConfig] = []
    max_len = max(len(buckets[k]) for k in keys)
    for index in range(max_len):
        round_keys = list(keys)
        rng.shuffle(round_keys)
        for key in round_keys:
            bucket = buckets[key]
            if index < len(bucket):
                ordered.append(bucket[index])

    return ordered if len(ordered) == len(runs) else runs
