"""Оценка third_repetition_policy — рекомендация без автозапуска 3-й repetition."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _reported_success(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    passed = sum(1 for r in rows if str(r.get("pass_type", "")).strip() in {"clean_pass", "soft_pass"})
    return passed / len(rows)


def _infra_rate(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    count = sum(1 for r in rows if str(r.get("is_infra_failure", "")).strip().lower() in {"true", "1", "yes"})
    return count / len(rows)


def _group_key(row: dict[str, str], field: str) -> str:
    if field == "model_id":
        return str(row.get("model_id", "")).strip()
    if field == "strategy":
        return str(row.get("strategy", "")).strip()
    if field == "task_category":
        task_id = str(row.get("task_id", "")).strip()
        return task_id.split("_")[0] if task_id else ""
    return str(row.get(field, "")).strip()


def evaluate_third_repetition_policy(
    policy: dict[str, Any] | None,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Возвращает рекомендацию selective third repetition по матрице."""
    if not isinstance(policy, dict) or not policy.get("enabled"):
        return {"evaluated": False, "recommend_third_repetition": False}

    triggers = policy.get("trigger_conditions")
    if not isinstance(triggers, dict):
        return {"evaluated": True, "recommend_third_repetition": False, "triggered": []}

    triggered: list[str] = []
    by_rep: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rep_raw = row.get("repetition") or row.get("run_summary.repetition")
        try:
            rep = int(rep_raw)
        except (TypeError, ValueError):
            continue
        by_rep[rep].append(row)

    reps = sorted(by_rep)
    if len(reps) >= 2:
        r1, r2 = _reported_success(by_rep[reps[0]]), _reported_success(by_rep[reps[1]])
        if r1 is not None and r2 is not None:
            delta = abs(r1 - r2)
            threshold = float(triggers.get("model_reported_success_delta_between_repetitions_gt", 1.0))
            if delta > threshold:
                triggered.append("model_reported_success_delta_between_repetitions")

        for field, key in (
            ("category", "category_success_delta_between_repetitions_gt"),
            ("strategy", "strategy_success_delta_between_repetitions_gt"),
        ):
            threshold = float(triggers.get(key, 1.0))
            groups: dict[str, dict[int, float]] = defaultdict(dict)
            for rep in reps:
                grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
                for row in by_rep[rep]:
                    grouped[_group_key(row, field)].append(row)
                for name, items in grouped.items():
                    rate = _reported_success(items)
                    if rate is not None:
                        groups[name][rep] = rate
            for name, rep_rates in groups.items():
                if len(rep_rates) >= 2:
                    values = [rep_rates[r] for r in sorted(rep_rates)]
                    if abs(values[0] - values[-1]) > threshold:
                        triggered.append(f"{field}_success_delta:{name}")

    infra_threshold = float(triggers.get("infra_error_rate_in_any_repetition_gt", 1.0))
    for rep in reps:
        rate = _infra_rate(by_rep[rep])
        if rate is not None and rate > infra_threshold:
            triggered.append(f"infra_error_rate_rep_{rep}")
            break

    recommend = bool(triggered) and not bool(policy.get("run_full_third_repetition"))
    return {
        "evaluated": True,
        "recommend_third_repetition": recommend,
        "triggered": triggered,
        "rerun_scope": policy.get("rerun_scope", []),
        "note": policy.get("note"),
    }
