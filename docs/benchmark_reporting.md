# Benchmark Reporting (Stage 7)

## 1. Назначение Stage 7

Stage 7 реализует слой анализа и отчётности поверх данных, собранных в ходе бенчмарк-прогонов.
Основная цель — превратить сырые артефакты (трассы агента, результаты валидации, статусы прогонов) в структурированные метрики, сравнительные таблицы и человекочитаемые отчёты.

Ключевые возможности:

- **Метрики на уровне прогона** — извлекаются из `AgentTrace` и `SceneValidationResult`.
- **Сводка по эксперименту** — агрегирует метрики нескольких прогонов.
- **Группировка и сравнение** — разбивает прогоны по стратегии, модели, MCP-профилю, категории задачи и т.д.
- **Ранжирование** — выявляет лучшие и худшие прогоны/группы.
- **Экспорт** — JSON, CSV, Markdown, HTML.
- **CLI** — запуск анализа и генерации отчётов из командной строки.

---

## 2. Какие данные анализируются

Анализ работает с директорией прогона или эксперимента, читая следующие артефакты:

| Файл | Содержимое |
|---|---|
| `agent_trace.json` | Шаги агента: LLM-вызовы, вызовы инструментов, наблюдения, планы, финальный шаг |
| `validation_result.json` | Результаты валидаторов сцены Blender: оценки, статусы, проблемы |
| `run_result.json` | Итоговый статус прогона, общая оценка, длительность, мета-данные |
| `scene_snapshot.json` | Снимок состояния сцены (используется валидаторами) |
| `metrics.json` | Агрегированные числовые метрики (опционально) |
| `summary.json` | Краткая сводка прогона (опционально) |

Артефакты читаются через `RunArtifactBundle` (`benchmark.analysis.trace_reader`), который допускает отсутствие любого из файлов.

---

## 3. Tool-call metrics

Извлекаются функцией `compute_tool_summary(trace)` из `benchmark.analysis.tool_metrics`.

| Метрика | Описание |
|---|---|
| `tool_call_count` | Общее число вызовов инструментов |
| `unique_tool_count` | Число уникальных инструментов |
| `invalid_tool_call_count` | Число вызовов с ошибкой (tool disabled, not found, invalid args) |
| `disabled_tool_call_count` | Вызовы отключённых инструментов |
| `tool_error_count` | Вызовы, завершившиеся ошибкой выполнения |
| `inspection_tool_count` | Вызовы инструментов категории `INSPECTION` |
| `mutation_tool_count` | Вызовы инструментов, изменяющих сцену (OBJECT, TRANSFORM, MATERIAL, LIGHT, CAMERA, EXPORT) |
| `python_tool_call_count` | Вызовы Python-скриптов |
| `asset_tool_call_count` | Вызовы инструментов работы с активами |
| `tool_repetition_count` | Число повторных вызовов одного и того же инструмента подряд |
| `average_tool_duration_sec` | Среднее время выполнения вызова инструмента |

Для каждого инструмента дополнительно строится `ToolCallMetric` с полями `total_calls`, `succeeded`, `failed`, `success_rate`, `avg_duration_sec`.

Категории инструментов определяются через `ToolCategory` из `benchmark.mcp.tool_contract`.

---

## 4. Agent metrics

Извлекаются функцией `compute_agent_summary(trace)` из `benchmark.analysis.agent_metrics`.

| Метрика | Описание |
|---|---|
| `llm_call_count` | Число LLM-шагов (`AgentStepType.LLM_CALL`) |
| `planning_step_count` | Число шагов планирования (`AgentStepType.PLAN`) |
| `observation_count` | Число шагов-наблюдений (`AgentStepType.OBSERVATION`) |
| `final_step_present` | Есть ли финальный шаг (`AgentStepType.FINAL`) |
| `retry_count` | Число повторных попыток (шаги подряд одного типа с ошибкой) |
| `step_limit_reached` | Признак достижения лимита шагов |
| `self_correction_attempts` | Попытки самокоррекции агента |
| `tool_error_recovery_count` | Случаи восстановления после ошибки инструмента |
| `error_count` | Общее число шагов с полем `error` |
| `average_step_duration_sec` | Среднее время шага |
| `max_step_duration_sec` | Максимальное время шага |
| `prompt_tokens` | Суммарные токены промпта (из usage всех LLM-шагов) |
| `completion_tokens` | Суммарные токены ответа |
| `total_tokens` | Суммарные токены (prompt + completion) |
| `estimated_cost` | Оценочная стоимость (если указана в trace) |

Стратегия агента (`react`, `direct_tool_calling`, `plan_and_execute`) влияет на интерпретацию шагов: в `plan_and_execute` ожидается наличие `PLAN`-шагов.

---

## 5. Validation metrics

Извлекаются функцией `compute_validation_summary(result)` из `benchmark.analysis.validation_metrics`.

| Метрика | Описание |
|---|---|
| `scene_total_score` | Общая оценка сцены (0.0–1.0), `None` если валидация не проводилась |
| `scene_overall_status` | Статус: `passed`, `failed`, `warning`, `unknown` |
| `passed_validator_count` | Число прошедших валидаторов |
| `failed_validator_count` | Число упавших валидаторов |
| `skipped_validator_count` | Число пропущенных валидаторов |
| `validation_error_count` | Суммарное число `error`-проблем |
| `validation_warning_count` | Суммарное число `warning`-проблем |
| `object_score` | Оценка `object_validator` (`None` если отсутствует или пропущен) |
| `transform_score` | Оценка `transform_validator` |
| `material_score` | Оценка `material_validator` |
| `light_score` | Оценка `light_validator` |
| `camera_score` | Оценка `camera_validator` |
| `export_score` | Оценка `export_validator` |

Отдельные метрики по валидатору доступны через `extract_validation_metrics(result)` — возвращает список `ValidationMetric`.

Проблемы со всех уровней (top-level и внутри каждого валидатора) объединяются через `extract_issues(result)`.

---

## 6. Error taxonomy

Реализована в `benchmark.analysis.error_taxonomy`.

### Категории ошибок (`ErrorCategory`)

| Категория | Примеры сообщений |
|---|---|
| `tool_disabled` | "tool not allowed", "tool is disabled" |
| `tool_unknown` | "tool not found", "command unrecognised" |
| `tool_invalid_arguments` | "argument '...' is invalid" |
| `tool_runtime_error` | "execution error in blender", "Traceback", "ToolInvocationError" |
| `llm_parse_error` | "failed to parse LLM response", "invalid JSON" |
| `llm_timeout` | "timed out", "AgentTimeoutError" |
| `agent_step_limit` | "max_steps reached", "step_limit reached" |
| `scene_object_missing` | issue code: `object_missing`, `object_type_mismatch`, `primitive_mismatch` |
| `scene_transform_mismatch` | issue code: `location_mismatch`, `rotation_mismatch`, `scale_mismatch` |
| `scene_material_mismatch` | issue code: `material_*_mismatch`, `object_missing_for_material` |
| `scene_light_mismatch` | issue code: `light_*_mismatch` |
| `scene_camera_mismatch` | issue code: `camera_*_mismatch`, `active_camera_missing` |
| `scene_export_missing` | issue code: `export_file_missing` |
| `mcp_connection_error` | "BlenderSocketUnavailable", "socket connection error" |
| `remote_agent_error` | "RemoteAgentError" |
| `unknown_error` | всё остальное |

### API

```python
from benchmark.analysis.error_taxonomy import (
    classify_trace_error,      # AgentStep → ErrorCategory
    classify_validation_issue, # ValidationIssue → ErrorCategory
    extract_errors,            # AgentTrace → list[ErrorRecord]
    aggregate_errors,          # RunArtifactBundle → dict[str, int]
    summarize_errors,          # list[ErrorRecord] → dict[str, int]
)
```

В `RunAnalysisResult.metrics` каждая категория сохраняется как `error.<category>: int`.

---

## 7. Group comparisons

Реализованы в `benchmark.analysis.comparison`.

### Измерения (`ComparisonDimension`)

| Значение | Группировка по |
|---|---|
| `strategy` | Стратегия агента (`react`, `direct_tool_calling`, `plan_and_execute`) |
| `model` | Имя модели LLM |
| `mcp_profile` | Профиль MCP-сервера |
| `run` | Каждый прогон отдельно |
| `agent_id` | Идентификатор агента |
| `task_category` | Категория задачи (geometry, materials, lighting, camera, export) |
| `difficulty` | Уровень сложности задачи (easy, medium, hard) |
| `remote_provider` | Провайдер LLM (anthropic, openai, openrouter, mock) |

### Метрики группы (`ComparisonGroup`)

| Поле | Описание |
|---|---|
| `run_count` | Число прогонов в группе |
| `success_rate` | Доля успешных прогонов (0.0–1.0) |
| `avg_score` | Средняя оценка сцены |
| `avg_tool_calls` | Среднее число вызовов инструментов |
| `avg_duration_sec` | Средняя длительность прогона |

### Функции

```python
from benchmark.analysis.comparison import (
    compare_runs,           # list[RunAnalysisResult] × ComparisonDimension → ComparisonReport
    group_by_strategy,
    group_by_model,
    group_by_mcp_profile,
    group_by_task_category,
    analyze_experiment,     # Path → ExperimentAnalysisResult
)
```

`analyze_experiment(dir)` рекурсивно ищет `agent_trace.json` в поддиректориях и строит полный `ExperimentAnalysisResult`.

---

## 8. Ranking

Реализовано в `benchmark.analysis.comparison`.

### Ранжирование прогонов

```python
from benchmark.analysis.comparison import rank_runs_by_score

ranked: list[RankedRun] = rank_runs_by_score(
    results,      # list[RunAnalysisResult]
    top_n=5,      # вернуть только топ-N (None = все)
    bottom_n=3,   # вернуть только худших-N (None = все)
)
```

`RankedRun.score_used` — оценка, использованная для ранжирования (`total_score`, при `None` — 0.0).
Ранг 1 = лучший результат.

### Ранжирование групп

```python
from benchmark.analysis.comparison import (
    rank_groups_by_average_score,   # по avg_score
    rank_groups_by_success_rate,    # по success_rate
    rank_groups_by_efficiency,      # по score / max(duration_sec, 1.0)
)
```

`rank_groups_by_efficiency` вычисляет `time_efficiency = avg_score / max(avg_duration_sec, 1.0)` и ранжирует по убыванию эффективности.

Все функции поддерживают `top_n` и `bottom_n` для ограничения результата.

---

## 9. ReportConfig

Конфигурация отчёта хранится в модели `ReportConfig` (`benchmark.analysis.models`).

```python
class ReportConfig(BaseModel):
    report_id: str = "default"
    title: str = "Benchmark Report"
    input_dir: Path = Path("./results")   # директория с прогонами
    output_dir: Path = Path("./reports")  # куда писать отчёты

    # Форматы вывода
    formats: list[str] = ["json", "csv", "markdown", "html"]

    # Переключатели разделов
    include_runs: bool = True              # таблица по прогонам
    include_group_comparison: bool = True  # сравнение по группам
    include_error_taxonomy: bool = True    # таксономия ошибок
    include_trace_details: bool = True     # детали трассы
    include_artifact_links: bool = True    # ссылки на артефакты

    metadata: dict[str, Any] = {}
```

### YAML-формат

```yaml
# configs/reports/default_report.yaml
report_id: default
title: "Benchmark Report"
input_dir: "./results"
output_dir: "./reports"
formats: [json, csv, markdown, html]
include_runs: true
include_group_comparison: true
include_error_taxonomy: true
include_trace_details: true
include_artifact_links: true
```

Загрузка из YAML:

```python
import yaml
from benchmark.analysis.models import ReportConfig

config = ReportConfig(**yaml.safe_load(Path("configs/reports/default_report.yaml").read_text()))
```

---

## 10. CLI examples

Все команды доступны через `python -m benchmark.analysis.cli`.

### Анализ одного прогона

```bash
python -m benchmark.analysis.cli analyze-run \
    --run-dir artifacts/runs/run_001 \
    --output artifacts/runs/run_001
# Создаёт: artifacts/runs/run_001/run_analysis.json
```

### Анализ эксперимента

```bash
python -m benchmark.analysis.cli analyze-experiment \
    --experiment-dir artifacts/experiments/exp_001 \
    --output artifacts/experiments/exp_001
# Создаёт: artifacts/experiments/exp_001/experiment_analysis.json
```

### Генерация отчётов по конфигу

```bash
python -m benchmark.analysis.cli build-report \
    --config configs/reports/default_report.yaml
# Создаёт: JSON, CSV, report.md, report.html в output_dir

# Переопределение директорий из CLI:
python -m benchmark.analysis.cli build-report \
    --config configs/reports/default_report.yaml \
    --input  artifacts/experiments/exp_001 \
    --output reports/exp_001
```

### Сравнение по измерению

```bash
python -m benchmark.analysis.cli compare \
    --input artifacts/experiments/exp_001 \
    --group-by strategy

# Вывод в stdout:
# Value                  Runs  Success%  AvgScore  AvgTools  AvgDur(s)
# react                  8     75.0%     0.8500    12.3      45.2
# direct_tool_calling    4     100.0%    0.9200    5.1       18.7

# Допустимые значения --group-by:
# strategy | model | mcp_profile | run | agent_id | task_category | difficulty | remote_provider
```

---

## 11. Integration with ExperimentRunner

После завершения пакетного прогона runner может автоматически запустить анализ и генерацию отчётов.

### Флаги runner CLI

```bash
# Только анализ (пишет experiment_analysis.json):
python -m benchmark.runner.cli experiment \
    --config configs/experiment.yaml \
    --analyze

# Анализ + Markdown/HTML отчёты (--report подразумевает --analyze):
python -m benchmark.runner.cli experiment \
    --config configs/experiment.yaml \
    --report
```

### Логика интеграции

При передаче `--analyze` или `--report` runner после завершения всех прогонов:

1. Определяет директорию вывода эксперимента (`_experiment_output_dir`).
2. Вызывает `analyze_experiment(output_dir)` — строит `ExperimentAnalysisResult`.
3. Сохраняет `experiment_analysis.json` через `write_experiment_analysis_json`.
4. Если `--report`: создаёт `report.md` (`build_markdown_report`) и `report.html` (`build_html_report`) в той же директории с `ReportConfig(title=experiment_id)`.

### Программное использование

```python
from benchmark.analysis.comparison import analyze_experiment
from benchmark.analysis.export import write_experiment_analysis_json
from benchmark.analysis.models import ReportConfig
from benchmark.analysis.report_builder import build_html_report, build_markdown_report

analysis = analyze_experiment(Path("artifacts/experiments/exp_001"))
write_experiment_analysis_json(analysis, "reports/experiment_analysis.json")

cfg = ReportConfig(title="My Experiment", output_dir=Path("reports"))
Path("reports/report.md").write_text(build_markdown_report(analysis, cfg))
Path("reports/report.html").write_text(build_html_report(analysis, cfg))
```

---

## 12. Ограничения

### Нет human-in-the-loop

Весь анализ полностью автоматизирован. Система не предусматривает:

- Ручной разметки или верификации результатов валидации.
- Экспертной оценки качества выполнения задачи.
- Интерактивного просмотра трасс агента.
- Обратной связи от человека в процессе анализа.

Отчёты основаны исключительно на структурированных данных из артефактов прогона. Субъективные аспекты качества (естественность диалога, корректность стратегии выбора инструментов, читаемость кода Python) не измеряются.

### Нет render similarity

Система не сравнивает визуальное сходство рендеров. Отсутствует:

- Сравнение изображений (pixel diff, SSIM, LPIPS и т.п.).
- Оценка корректности освещения, теней, материалов на основе рендера.
- Проверка соответствия сцены целевому скриншоту.

Вся оценка сцены основана на **структурных проверках** (наличие объектов, числовые свойства трансформации, параметры материалов/источников света/камеры, наличие экспортированного файла), реализованных в `benchmark.validation`.

### Прочие ограничения

- **Категория задачи** определяется эвристически по `task_id` (регулярные выражения). Ошибки классификации возможны для нестандартных имён задач.
- **Сложность задачи** (`difficulty`) извлекается по словам `easy`, `medium`, `hard` в `task_id` с границей `\b`. Подчёркивания не являются границей слова — рекомендуется использовать дефисы: `geometry-easy-cube`.
- **Оценка токенов и стоимости** — только если trace содержит поле `usage` в LLM-шагах. При его отсутствии поля `prompt_tokens`, `completion_tokens`, `estimated_cost` не заполняются.
- **Анализ экспортированных файлов** (`export_score`) — только структурная проверка наличия файла, не его содержимого.
