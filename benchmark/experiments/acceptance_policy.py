"""Оценка acceptance_policy из metadata матрицы по результатам прогона."""

from __future__ import annotations

from typing import Any


def _rate(rows: list[dict[str, str]], predicate) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if predicate(row)) / len(rows)


def _reported_success_rate(rows: list[dict[str, str]]) -> float | None:
    return _rate(
        rows,
        lambda r: str(r.get("pass_type", "")).strip() in {"clean_pass", "soft_pass"},
    )


def _infra_error_rate(rows: list[dict[str, str]]) -> float | None:
    return _rate(rows, lambda r: str(r.get("is_infra_failure", "")).strip().lower() in {"true", "1", "yes"})


def _category_success_rate(rows: list[dict[str, str]], category: str) -> float | None:
    subset = [r for r in rows if _task_category(r) == category]
    return _reported_success_rate(subset)


def _task_category(row: dict[str, str]) -> str:
    explicit = str(row.get("task_category", "")).strip()
    if explicit:
        return explicit
    task_id = str(row.get("task_id", "")).strip()
    return task_id.split("_")[0] if task_id else ""


def _count_unclassified(rows: list[dict[str, str]]) -> int:
    return sum(1 for r in rows if str(r.get("error_type", "")).strip() == "UnclassifiedError")


def _llm_parse_error_rate(rows: list[dict[str, str]]) -> float | None:
    parse_types = {"LlmParseError", "InvalidJsonResponse"}
    return _rate(
        rows,
        lambda r: str(r.get("error_type", "")).strip() in parse_types
        or "parse" in str(r.get("error_class", "")).strip().lower(),
    )


def evaluate_acceptance_policy(
    policy: dict[str, Any] | None,
    *,
    rows: list[dict[str, str]],
    gate_result: dict[str, Any] | None,
    structural_validity: str,
    planned_runs: int | None,
    expected_runs: int | None,
) -> dict[str, Any]:
    """Возвращает decision_level и причины на основе metadata.acceptance_policy."""
    if not isinstance(policy, dict):
        return {"decision_level": None, "reasons": [], "evaluated": False}

    readiness_ok = True if gate_result is None else bool(gate_result.get("readiness_ok"))
    failed_gates = gate_result.get("failed_gates", []) if isinstance(gate_result, dict) else []
    reported = _reported_success_rate(rows)
    infra = _infra_error_rate(rows)
    unclassified = _count_unclassified(rows)
    parse_rate = _llm_parse_error_rate(rows)

    metrics = {
        "structural_validity": structural_validity,
        "readiness_ok": readiness_ok,
        "reported_success_rate": reported,
        "infra_error_rate": infra,
        "unclassified_error": unclassified,
        "llm_parse_error_rate": parse_rate,
        "planned_runs": planned_runs,
        "expected_runs": expected_runs,
    }

    rerun_cfg = policy.get("rerun_required_if") if isinstance(policy.get("rerun_required_if"), dict) else {}
    rerun_reasons: list[str] = []

    def _check_rerun() -> bool:
        if structural_validity == "failed" and rerun_cfg.get("structural_validity") == "failed":
            rerun_reasons.append("structural_validity_failed")
        if expected_runs is not None and planned_runs is not None and planned_runs != expected_runs:
            if rerun_cfg.get("expected_runs_match") is False:
                rerun_reasons.append("expected_runs_mismatch")
        if "unclassified_error_gt" in rerun_cfg and unclassified > int(rerun_cfg["unclassified_error_gt"]):
            rerun_reasons.append("unclassified_error")
        if "llm_parse_error_rate_gt" in rerun_cfg and parse_rate is not None and parse_rate > float(rerun_cfg["llm_parse_error_rate_gt"]):
            rerun_reasons.append("llm_parse_error_rate")
        if "reported_success_rate_below" in rerun_cfg and reported is not None and reported < float(rerun_cfg["reported_success_rate_below"]):
            rerun_reasons.append("reported_success_rate")
        if "infra_error_rate_gt" in rerun_cfg and infra is not None and infra > float(rerun_cfg["infra_error_rate_gt"]):
            rerun_reasons.append("infra_error_rate")
        category_below = rerun_cfg.get("category_success_below")
        if isinstance(category_below, dict):
            for category, threshold in category_below.items():
                rate = _category_success_rate(rows, str(category))
                if rate is not None and rate < float(threshold):
                    rerun_reasons.append(f"category_{category}")
        if not readiness_ok and failed_gates:
            rerun_reasons.append("readiness_gates")
        return bool(rerun_reasons)

    if _check_rerun():
        return {
            "decision_level": "rerun_required",
            "reasons": rerun_reasons,
            "metrics": metrics,
            "evaluated": True,
        }

    accept_cfg = policy.get("accept_for_report_if") if isinstance(policy.get("accept_for_report_if"), dict) else {}
    accept_reasons: list[str] = []
    can_accept = True
    if accept_cfg.get("structural_validity") == "passed" and structural_validity != "passed":
        can_accept = False
        accept_reasons.append("structural_validity")
    if accept_cfg.get("readiness_ok") is True and not readiness_ok:
        can_accept = False
        accept_reasons.append("readiness_ok")
    if accept_cfg.get("unclassified_error") == 0 and unclassified > 0:
        can_accept = False
        accept_reasons.append("unclassified_error")
    if reported is not None and reported < float(accept_cfg.get("reported_success_rate_min", 0.0)):
        can_accept = False
        accept_reasons.append("reported_success_rate_min")

    if can_accept:
        return {
            "decision_level": "accept_for_report",
            "reasons": accept_reasons,
            "metrics": metrics,
            "evaluated": True,
        }

    caveats_cfg = policy.get("accept_with_caveats_if") if isinstance(policy.get("accept_with_caveats_if"), dict) else {}
    if caveats_cfg.get("structural_validity") == "passed" and structural_validity == "passed":
        min_rate = float(caveats_cfg.get("min_reported_success_rate", 0.0))
        if reported is not None and reported >= min_rate:
            return {
                "decision_level": "accept_with_caveats",
                "reasons": accept_reasons or ["readiness_or_quality_below_accept_threshold"],
                "metrics": metrics,
                "evaluated": True,
            }

    return {
        "decision_level": "rerun_required",
        "reasons": accept_reasons or rerun_reasons or ["policy_not_satisfied"],
        "metrics": metrics,
        "evaluated": True,
    }
