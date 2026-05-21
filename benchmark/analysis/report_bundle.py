from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

from benchmark.analysis.models import ExperimentAnalysisResult
from benchmark.analysis.report_builder import (
    _avg_score,
    _category,
    _cost_total,
    _group_rows,
    _issue_counts,
    _pct,
    _status_counts,
    build_key_findings,
)


def write_report_text_ru(analysis: ExperimentAnalysisResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    runs = analysis.runs
    s = analysis.summary
    models = ", ".join(sorted({r.model for r in runs if r.model})) or "N/A"
    strategies = ", ".join(sorted({r.strategy for r in runs if r.strategy})) or "N/A"
    profiles = ", ".join(sorted({r.mcp_profile for r in runs if r.mcp_profile})) or "N/A"
    best = _best_strategy_name(analysis)
    weak_categories = _weak_categories(analysis)
    top_validation = _top_issue_text(analysis, "validation")
    top_agent = _top_issue_text(analysis, "agent")
    top_tool = _top_issue_text(analysis, "tool")
    text = f"""# Текст для отчёта

## 1. Описание экспериментального запуска

В рамках диагностического прогона была использована матрица `{analysis.experiment_id}` из {s.total_runs} запусков. Прогон включал {len(set(r.task_id for r in runs))} Blender-задач, стратегии {strategies} и MCP-профили {profiles}. В качестве модели использовалась {models}. Основной целью запуска являлась проверка воспроизводимости стенда и выявление различий между режимами выполнения, а не достижение максимального качества на всех задачах.

## 2. Сводные результаты

Чисто успешных запусков зафиксировано {s.clean_pass_count}, soft pass запусков - {s.soft_pass_count}, failed validation - {s.failed_validation_count or s.failed_count}, runtime error - {s.runtime_error_count or s.error_count}. Строгая доля успеха составила {_pct(s.strict_success_rate)}, а отчётная доля успеха с учётом soft pass - {_pct(s.reported_success_rate)}.

## 3. Сравнение стратегий

Наиболее устойчивой стратегией по отчётной доле успешных запусков стала {best}. ReAct в MVP сохраняется как диагностическая стратегия; низкий success rate ReAct не является ошибкой стенда, а отражает ограничения текущей реализации agent loop на многошаговых Blender-задачах.

## 4. Сравнение MCP-профилей

Сравнение MCP-профилей проводилось по одинаковой матрице задач и стратегий. Профили рассматривались как разные режимы доступности инструментов, поэтому различия в результатах интерпретируются как влияние tool surface на устойчивость агента.

## 5. Анализ категорий задач

Наиболее проблемные категории задач в данном прогоне: {weak_categories}. Эти категории следует рассматривать как основные направления дальнейшей стабилизации benchmark-сценариев и агентных стратегий.

## 6. Анализ ошибок

Основные validation issues: {top_validation}. Основные agent issues: {top_agent}. Основные tool issues: {top_tool}. Ошибки включены в отчётный результат и классифицированы через `pass_type`, поэтому они не требуют ручной переработки raw artifacts.

## 7. Ограничения прогона

MVP-прогон использует одну модель и одну повторность, поэтому результаты отражают диагностическую картину конкретной конфигурации, а не финальную оценку всех возможных моделей и режимов. Стоимость берётся только из provider-reported данных OpenRouter; внутренние формулы оценки стоимости не используются.

## 8. Вывод

Benchmark-стенд сформировал воспроизводимый отчётный пакет: CSV-таблицу, JSON-агрегацию, Markdown/HTML-отчёты, графики и текстовый блок для вставки в работу. Основной результат MVP состоит в автоматической подготовке данных для отчёта и прозрачной фиксации ограничений агентов, стратегий и MCP-профилей.
"""
    path.write_text(text, encoding="utf-8")


def write_readme_report(path: Path) -> None:
    path.write_text(
        """# Report Bundle

Пакет содержит готовые материалы benchmark-прогона.

- `summary.csv` - основной источник данных для таблиц и анализа.
- `summary.json` и `experiment_analysis.json` - машинно-читаемые агрегаты.
- `report.md` и `report.html` - готовые отчётные таблицы и выводы.
- `report_text_ru.md` - русскоязычный текст для вставки в отчёт.
- `figures/` - PNG-графики для иллюстраций.
- `manifest.json` - сведения о матрице и runtime.

`pass_type` является основным отчётным статусом:

- `clean_pass` - validation пройдена без issues.
- `soft_pass` - сцена прошла по score, но содержит validation issues.
- `failed_validation` - агент завершил выполнение, сцена доступна, validation failed.
- `runtime_error` - запуск не дошёл до корректной итоговой сцены из-за agent/tool/runtime ошибки.

`failed_validation` показывает проблему качества итоговой сцены. `runtime_error` показывает проблему выполнения агента, инструмента или runtime. Для стоимости используются только provider-reported данные OpenRouter.
""",
        encoding="utf-8",
    )


def write_figures(analysis: ExperimentAnalysisResult, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []

    def save_bar(filename: str, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values, color="#3b6ea8")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        out = figures_dir / filename
        fig.savefig(out, dpi=160)
        plt.close(fig)
        paths.append(out)

    save_bar("success_by_strategy.png", *_success_series(analysis.runs, lambda r: r.strategy), "Success by strategy", "reported success rate")
    save_bar("success_by_profile.png", *_success_series(analysis.runs, lambda r: r.mcp_profile or "unknown"), "Success by MCP profile", "reported success rate")
    save_bar("success_by_category.png", *_success_series(analysis.runs, _category), "Success by task category", "reported success rate")

    validation = _issue_counts(analysis.runs, "validation").most_common(10)
    save_bar(
        "top_validation_issues.png",
        [item[0] for item in validation] or ["none"],
        [float(item[1]) for item in validation] or [0.0],
        "Top validation issues",
        "count",
    )
    save_bar("cost_by_strategy.png", *_cost_series(analysis.runs), "OpenRouter cost by strategy", "USD")
    save_bar("score_by_strategy.png", *_score_series(analysis.runs), "Score by strategy", "average score")
    save_bar("error_breakdown.png", ["failed_validation", "runtime_error"], [
        float(analysis.summary.failed_validation_count or analysis.summary.failed_count),
        float(analysis.summary.runtime_error_count or analysis.summary.error_count),
    ], "Error breakdown", "runs")
    return paths


def create_report_bundle(output_root: Path, analysis: ExperimentAnalysisResult, files: Iterable[Path]) -> Path:
    bundle = output_root / "report_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        if file_path.exists():
            shutil.copy2(file_path, bundle / file_path.name)
    source_figures = output_root / "figures"
    if source_figures.exists():
        target_figures = bundle / "figures"
        if target_figures.exists():
            shutil.rmtree(target_figures)
        shutil.copytree(source_figures, target_figures)
    manifest = output_root / "manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, bundle / "manifest.json")
    else:
        (bundle / "manifest.json").write_text(json.dumps({"experiment_id": analysis.experiment_id}, indent=2), encoding="utf-8")
    write_readme_report(bundle / "README_REPORT.md")
    return bundle


def _success_series(runs, key_fn) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    values: list[float] = []
    for label, items in _group_rows(runs, key_fn):
        c = _status_counts(items)
        labels.append(label)
        values.append((c["clean_pass"] + c["soft_pass"]) / len(items) if items else 0.0)
    return labels or ["none"], values or [0.0]


def _cost_series(runs) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    values: list[float] = []
    for label, items in _group_rows(runs, lambda r: r.strategy):
        labels.append(label)
        values.append(_cost_total(items))
    return labels or ["none"], values or [0.0]


def _score_series(runs) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    values: list[float] = []
    for label, items in _group_rows(runs, lambda r: r.strategy):
        labels.append(label)
        values.append(_avg_score(items) or 0.0)
    return labels or ["none"], values or [0.0]


def _best_strategy_name(analysis: ExperimentAnalysisResult) -> str:
    best_name = "N/A"
    best_rate = -1.0
    for name, items in _group_rows(analysis.runs, lambda r: r.strategy):
        c = _status_counts(items)
        rate = (c["clean_pass"] + c["soft_pass"]) / len(items) if items else 0.0
        if rate > best_rate:
            best_name = name
            best_rate = rate
    return best_name


def _weak_categories(analysis: ExperimentAnalysisResult) -> str:
    values: list[tuple[str, float]] = []
    for name, items in _group_rows(analysis.runs, _category):
        c = _status_counts(items)
        values.append((name, (c["clean_pass"] + c["soft_pass"]) / len(items) if items else 0.0))
    return ", ".join(name for name, _ in sorted(values, key=lambda item: item[1])[:2]) or "N/A"


def _top_issue_text(analysis: ExperimentAnalysisResult, kind: str) -> str:
    items = _issue_counts(analysis.runs, kind).most_common(5)
    return ", ".join(f"{code} ({count})" for code, count in items) if items else "не зафиксированы"
