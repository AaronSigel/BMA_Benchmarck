# Experimental Matrix — Stage 8

## 1. Назначение Stage 8

Stage 8 закрывает практическую часть benchmark-стенда. Его цель — собрать
воспроизводимые экспериментальные сценарии и провести полный путь от постановки
задачи до отчёта:

```
BenchmarkTask
    ↓ AgentRuntime
    ↓ MCP / Blender
    ↓ SceneSnapshot
    ↓ SceneValidator
    ↓ ExperimentRunner
    ↓ Analysis / Report
```

Сравниваются: модели, агентные стратегии (Direct, ReAct, Plan-and-Execute),
MCP-профили (minimal, no_python, inspection_enabled) и категории Blender-задач.
Такая матрица прямо соответствует исходной исследовательской постановке.

---

## 2. Что такое экспериментальная матрица

Матрица описывает **декартово произведение** осей эксперимента:

```
модель / remote-agent
    × агентная стратегия
    × MCP-профиль
    × категория Blender-задач
    × уровень сложности
    × (число повторений)
```

Из матрицы автоматически генерируется `ExperimentConfig` — список `RunConfig`,
каждый из которых соответствует одному запуску агента.

Матрица хранится как YAML-файл в `configs/matrices/`. Для каждого сценария
предусмотрен отдельный файл:

| Файл | Назначение |
|---|---|
| `smoke_matrix.yaml` | Быстрая локальная проверка без API и Blender |
| `baseline_matrix.yaml` | Основная серия экспериментов |
| `api_models_matrix.yaml` | Сравнение API-провайдеров, opt-in |
| `remote_agents_matrix.yaml` | Внешние агенты (Codex, Claude Code и др.), opt-in |

---

## 3. Формат matrix YAML

```yaml
matrix_id: <строка, уникальный идентификатор>
title: <человекочитаемое название>
description: <произвольный текст>

tasks:
  ids:                        # конкретные ID задач
    - geometry_001_basic_primitives
  categories: [geometry]      # или фильтр по категории
  difficulties: [easy]        # или фильтр по уровню сложности

agents:
  ids:
    - direct_openrouter
  config_root: configs/agents  # каталог с agent YAML

mcp_profiles:
  - minimal
  # - no_python
  # - inspection_enabled

models:
  ids:
    - default                  # или конкретный model_id

execution_modes:
  - agent_mcp                  # реальный запуск через MCP+Blender
  # - external_snapshot        # из готового снимка сцены (mock/offline)
  # - remote_agent             # через внешний агент

repetitions: 3                 # число повторений каждой комбинации

output_root: artifacts/experiments/<matrix_id>

report_config_path: null       # опционально: путь к ReportConfig YAML

metadata:
  snapshot_path: ...           # путь к снимку для external_snapshot
  artifacts_dir: ...           # каталог с тестовыми артефактами
  opt_in: true                 # помечает матрицы, требующие внешних сервисов
  strict_readiness: false      # превращать предупреждения в ошибки
```

**Идентификатор запуска** строится автоматически по шаблону:

```
<matrix_id>__<task_id>__<agent_id>__<mcp_profile>__r<repetition>
```

---

## 4. Smoke matrix

**Файл:** `configs/matrices/smoke_matrix.yaml`

Быстрая проверка всего pipeline без внешних сервисов, API-ключей и Blender.
Использует `execution_modes: [external_snapshot]` и агент с `provider: mock`.

```yaml
matrix_id: smoke_matrix
tasks:
  ids: [geometry_001_basic_primitives]
agents:
  ids: [mock_agent]
mcp_profiles: [minimal]
execution_modes: [external_snapshot]
repetitions: 1
output_root: artifacts/experiments/smoke_matrix
metadata:
  snapshot_path: tests/fixtures/validation/valid_geometry_snapshot.json
  artifacts_dir: tests/fixtures/validation
```

**Запуск:**

```bash
python -m benchmark.experiments.cli run-and-report \
  --matrix configs/matrices/smoke_matrix.yaml
```

Ожидаемые артефакты в `artifacts/experiments/smoke_matrix/`:

```
experiment_result.json
experiment_analysis.json
manifest.json
report.md
report.html
```

Smoke-матрица пригодна для CI и локальной разработки. Не требует сетевого
соединения.

---

## 5. Baseline matrix

**Файл:** `configs/matrices/baseline_matrix.yaml`

Основная серия экспериментов для научной работы. Сравнивает три агентные
стратегии на трёх MCP-профилях и пяти категориях Blender-задач.

```yaml
matrix_id: baseline_matrix
tasks:
  ids:
    - geometry_001_basic_primitives   # + 14 других задач
    - materials_001_basic_colors
    - lighting_001_area_light
    - camera_001_front_view
    - export_001_blend_file
    # ...
agents:
  ids:
    - direct_openrouter
    - react_openrouter
    - plan_execute_openrouter
mcp_profiles:
  - minimal
  - no_python
  - inspection_enabled
execution_modes: [agent_mcp]
repetitions: 3
output_root: artifacts/experiments/baseline_matrix
```

**Предусловия:**

- Blender установлен (`BMA_BLENDER_BIN` или в `PATH`)
- MCP-сервер доступен на `localhost:9876`
- Переменные окружения для API-ключей агентов выставлены

**Запуск:**

```bash
python -m benchmark.experiments.cli readiness \
  --matrix configs/matrices/baseline_matrix.yaml

python -m benchmark.experiments.cli run-and-report \
  --matrix configs/matrices/baseline_matrix.yaml
```

**Что исключено по умолчанию:**

- `python_enabled`, `full` — отдельные расширенные сценарии
- Внешние asset-инструменты
- `remote_agent`

---

## 6. API models matrix

**Файл:** `configs/matrices/api_models_matrix.yaml`

**Opt-in сценарий.** Сравнивает API-провайдеров (`openrouter`, `openai_compatible`,
`anthropic`) и три агентные стратегии на репрезентативном наборе задач.

```yaml
matrix_id: api_models_matrix
tasks:
  ids:
    - geometry_001_basic_primitives
    - materials_001_basic_colors
    - lighting_001_area_light
    - camera_001_front_view
    - export_001_blend_file
agents:
  ids:
    - direct_openrouter
    - direct_openai_compatible
    - react_openrouter
    - react_anthropic
    - plan_execute_openrouter
mcp_profiles: [no_python, inspection_enabled]
execution_modes: [agent_mcp]
repetitions: 1
```

**Требования:**

- `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` в окружении
- API-ключи не хранить в YAML-файлах, манифестах и фикстурах
- Blender + MCP-сервер

**Не запускать** в обычных unit-тестах и CI smoke-проверках — потребляет платные
квоты и может упереться в rate limits.

```bash
python -m benchmark.experiments.cli readiness \
  --matrix configs/matrices/api_models_matrix.yaml

python -m benchmark.experiments.cli run-and-report \
  --matrix configs/matrices/api_models_matrix.yaml
```

---

## 7. Remote agents matrix

**Файл:** `configs/matrices/remote_agents_matrix.yaml`

**Opt-in сценарий.** Запускает задачи через внешние серверные агенты
(Codex, Claude Code, generic HTTP, generic command).

```yaml
matrix_id: remote_agents_matrix
tasks:
  ids: [geometry_001_basic_primitives]
agents:
  ids:
    - remote_agent_codex
    - remote_agent_claude
    - generic_http
    - generic_command
  include_remote_agents: true
mcp_profiles: [minimal]
execution_modes: [remote_agent]
repetitions: 1
```

**Требования:**

- Настроенный внешний агент-сервер
- Переменные окружения для его аутентификации
- Не включать в обычные unit-тесты

Readiness-проверка выдаёт предупреждение для каждого remote-агента — это
ожидаемое поведение.

```bash
python -m benchmark.experiments.cli readiness \
  --matrix configs/matrices/remote_agents_matrix.yaml

python -m benchmark.experiments.cli run-and-report \
  --matrix configs/matrices/remote_agents_matrix.yaml
```

---

## 8. Readiness checks

Перед запуском экспериментов система проверяет готовность окружения:

```bash
python -m benchmark.experiments.cli readiness \
  --matrix configs/matrices/<matrix>.yaml \
  --output readiness.json          # опционально
```

**Что проверяется:**

| Условие | Режим |
|---|---|
| Все task ID существуют в реестре | обязательно |
| Все agent YAML найдены | обязательно |
| Все MCP-профили найдены | обязательно |
| API-ключи выставлены | WARNING (или ERROR при `strict_readiness: true`) |
| Remote-агент сконфигурирован | WARNING |
| `output_root` доступен для записи | обязательно |
| Blender найден (для `agent_mcp`, `blender_smoke`) | обязательно |
| MCP-сокет доступен (для `agent_mcp`, `mcp_smoke`) | обязательно |

**Формат вывода:**

```
status: pass          # или fail
WARNING: ...
ERROR: ...
```

При наличии ошибок `E2EBenchmarkRunner` прерывает выполнение до запуска батча.

---

## 9. Run manifest

Перед каждым запуском батча автоматически создаётся `manifest.json` в
`output_root/`. Манифест фиксирует состояние окружения для воспроизводимости.

**Содержимое:**

```json
{
  "matrix_id": "smoke_matrix",
  "generated_at": "2026-05-16T12:00:00Z",
  "git_commit": "abc123...",
  "python_version": "3.12.0",
  "platform": "Linux-...",
  "task_ids": ["geometry_001_basic_primitives"],
  "agent_ids": ["mock_agent"],
  "mcp_profiles": ["minimal"],
  "execution_modes": ["external_snapshot"],
  "repetitions": 1,
  "config_hash": "<sha256 конфига матрицы>",
  "env_requirements": [...],
  "metadata": {
    "readiness_ok": true,
    "readiness_warnings": [],
    "readiness_errors": []
  }
}
```

`config_hash` — SHA-256 от санированного (без секретов) дампа матрицы.
Позволяет убедиться, что два запуска использовали идентичные конфигурации.

Секретные ключи (`api_key`, `token`, `password` и др.) автоматически удаляются
из манифеста.

---

## 10. Запуск run-and-report

Полный цикл: матрица → эксперимент → анализ → отчёт.

```bash
python -m benchmark.experiments.cli run-and-report \
  --matrix configs/matrices/<matrix>.yaml
```

**Порядок выполнения:**

1. Загрузить и провалидировать матрицу
2. `readiness` — проверить окружение; при ошибках — прервать
3. `generate` — построить `ExperimentConfig` (список RunConfig)
4. Записать `manifest.json`
5. `BatchRunner.run_experiment` — выполнить все запуски
6. Записать `experiment_result.json`
7. `analyze_experiment` — рассчитать метрики
8. Записать `experiment_analysis.json`, `metrics.csv`, `summary.csv`, `summary.json`
9. Построить `report.md` и `report.html`

**Другие подкоманды:**

```bash
# Только сгенерировать ExperimentConfig
python -m benchmark.experiments.cli generate \
  --matrix configs/matrices/smoke_matrix.yaml \
  --output experiment.yaml

# Только проверить готовность
python -m benchmark.experiments.cli readiness \
  --matrix configs/matrices/smoke_matrix.yaml

# Запустить без анализа
python -m benchmark.experiments.cli run \
  --matrix configs/matrices/smoke_matrix.yaml

# Запустить + анализ (без отчёта)
python -m benchmark.experiments.cli run-and-analyze \
  --matrix configs/matrices/smoke_matrix.yaml

# Показать доступные матрицы
python -m benchmark.experiments.cli list-matrices \
  --directory configs/matrices
```

---

## 11. Как интерпретировать результаты

После `run-and-report` в `output_root/` появляются:

| Файл | Содержимое |
|---|---|
| `experiment_result.json` | Сырые результаты всех запусков, статусы |
| `experiment_analysis.json` | Агрегированные метрики и сравнения |
| `metrics.csv` | Метрики по запускам в табличном виде |
| `summary.csv` / `summary.json` | Сводка по матрице |
| `report.md` / `report.html` | Читаемый отчёт с таблицами |
| `manifest.json` | Воспроизводимый снимок конфигурации |
| `<run_id>/` | Артефакты отдельного запуска |

**Ключевые метрики в отчёте:**

- `avg_scene_score` — средняя оценка сцены (0–1) от SceneValidator
- `passed_runs` / `total_runs` — доля успешных запусков
- `avg_tool_calls` — среднее число вызовов инструментов
- `avg_duration_sec` — среднее время выполнения
- `avg_llm_calls` — среднее число обращений к LLM

**Категории ошибок** (`ErrorCategory`):

```
tool_disabled / tool_unknown / tool_invalid_arguments / tool_runtime_error
llm_parse_error / llm_timeout / agent_step_limit
scene_object_missing / scene_transform_mismatch / scene_material_mismatch
scene_light_mismatch / scene_camera_mismatch / scene_export_missing
mcp_connection_error / remote_agent_error
```

**Что смотреть в первую очередь:**

1. `passed_runs` — базовая работоспособность стратегии
2. `avg_scene_score` по стратегиям — качество выполнения задач
3. `avg_tool_calls` — эффективность агента (меньше = лаконичнее)
4. Топ ошибок — куда смотреть при деградации

---

## 12. Обязательные сценарии для научной работы

Минимальная матрица для первой реальной серии экспериментов:

| Ось | Значения |
|---|---|
| Task category | geometry, materials, lighting, camera, export |
| Difficulty | easy, medium |
| Agent strategy | direct_tool_calling, react, plan_and_execute |
| MCP profile | no_python, inspection_enabled |
| Models | 2 API-модели (OpenRouter + Anthropic) |
| Repetitions | 3 |

**Smoke (обязательно, без API):**

```bash
python -m benchmark.experiments.cli run-and-report \
  --matrix configs/matrices/smoke_matrix.yaml
```

Проходит в CI. Проверяет pipeline целиком без внешних зависимостей.

**Baseline (обязательно для основных результатов):**

```bash
python -m benchmark.experiments.cli run-and-report \
  --matrix configs/matrices/baseline_matrix.yaml
```

Сравнивает Direct vs ReAct vs Plan-and-Execute на `no_python` и
`inspection_enabled`. Требует Blender + MCP + API-ключи.

**API models (opt-in, для сравнения провайдеров):**

```bash
python -m benchmark.experiments.cli run-and-report \
  --matrix configs/matrices/api_models_matrix.yaml
```

Нужен отдельный пул API-ключей. Запускать вне CI.

**Remote agents (opt-in, для внешних агентов):**

```bash
python -m benchmark.experiments.cli run-and-report \
  --matrix configs/matrices/remote_agents_matrix.yaml
```

Требует настроенного внешнего агента. Запускать вручную.

**Не включать по умолчанию:**

- `python_enabled` / `full` MCP-профили — расширенные сценарии
- External asset tools
- Human-in-the-loop
- Visual feedback workflow
