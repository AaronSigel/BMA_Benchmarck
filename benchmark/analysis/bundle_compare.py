from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def compare_bundles(bundle_a: Path | str, bundle_b: Path | str, output_dir: Path | str | None = None) -> dict[str, Any]:
    a = _load_bundle(Path(bundle_a))
    b = _load_bundle(Path(bundle_b))
    result = {
        "bundle_a": str(bundle_a),
        "bundle_b": str(bundle_b),
        "total_runs": {"a": a["total_runs"], "b": b["total_runs"], "delta": b["total_runs"] - a["total_runs"]},
        "clean_pass_rate": _delta(a["clean_pass_rate"], b["clean_pass_rate"]),
        "strategy_success": _compare_maps(a["strategy_success"], b["strategy_success"]),
        "profile_success": _compare_maps(a["profile_success"], b["profile_success"]),
        "top_issues": {"a": a["top_issues"], "b": b["top_issues"]},
    }
    out = Path(output_dir) if output_dir else Path(bundle_b).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "comparison_report.md").write_text(_markdown(result), encoding="utf-8")
    return result


def _load_bundle(root: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with (root / "summary.csv").open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    total = len(rows)
    clean = sum(1 for row in rows if row.get("pass_type") == "clean_pass")
    return {
        "total_runs": total,
        "clean_pass_rate": clean / total if total else 0.0,
        "strategy_success": _success_by(rows, "strategy"),
        "profile_success": _success_by(rows, "mcp_profile"),
        "top_issues": _top_issues(rows),
    }


def _success_by(rows: list[dict[str, str]], key: str) -> dict[str, float]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get(key) or "unknown"].append(row)
    return {
        name: sum(1 for row in items if row.get("pass_type") in {"clean_pass", "soft_pass"}) / len(items)
        for name, items in groups.items()
        if items
    }


def _top_issues(rows: list[dict[str, str]]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for row in rows:
        for col in ("validation_issues", "agent_issues", "tool_issues", "all_issues"):
            value = row.get(col) or ""
            if value == "null":
                continue
            for item in value.split(";"):
                code = item.strip().split(":", 1)[0]
                if code and code != "null":
                    counter[code] += 1
    return counter.most_common(10)


def _delta(a: float, b: float) -> dict[str, float]:
    return {"a": a, "b": b, "delta": b - a}


def _compare_maps(a: dict[str, float], b: dict[str, float]) -> dict[str, dict[str, float]]:
    keys = sorted(set(a) | set(b))
    return {key: _delta(a.get(key, 0.0), b.get(key, 0.0)) for key in keys}


def _markdown(result: dict[str, Any]) -> str:
    lines = ["# Bundle Comparison", ""]
    total = result["total_runs"]
    clean = result["clean_pass_rate"]
    lines.extend([
        f"- total_runs: {total['a']} -> {total['b']} (delta {total['delta']})",
        f"- clean_pass_rate: {clean['a']:.4f} -> {clean['b']:.4f} (delta {clean['delta']:.4f})",
        "",
        "## Strategy Success",
        "",
        "| strategy | a | b | delta |",
        "| --- | --- | --- | --- |",
    ])
    for key, item in result["strategy_success"].items():
        lines.append(f"| {key} | {item['a']:.4f} | {item['b']:.4f} | {item['delta']:.4f} |")
    lines.extend(["", "## Profile Success", "", "| profile | a | b | delta |", "| --- | --- | --- | --- |"])
    for key, item in result["profile_success"].items():
        lines.append(f"| {key} | {item['a']:.4f} | {item['b']:.4f} | {item['delta']:.4f} |")
    return "\n".join(lines) + "\n"
