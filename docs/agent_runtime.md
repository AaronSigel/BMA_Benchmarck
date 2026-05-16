# Agent Runtime (Stage 6)

## 1. Назначение

Stage 6 добавляет слой **Agent Runtime** — исполняющую оболочку, которая принимает `BenchmarkTask` и `AgentConfig`, вызывает внешний LLM или удалённый агент через выбранную стратегию, исполняет инструменты Blender MCP и возвращает `AgentRunResult` со ссылкой на `AgentTrace`.

Поток данных:

```
BenchmarkTask
    ↓
AgentConfig  →  AgentRuntime
                    ↓
            AgentStrategy (Direct / ReAct / Plan-and-Execute / Remote)
                    ↓
            LlmClient  |  RemoteAgentClient
                    ↓
            ToolExecutor  →  Blender MCP Server
                    ↓
            AgentTrace  →  artifacts/agent_runs/<run_id>/
                    ↓
            AgentRunResult  →  ExperimentRunner
```

---

## 2. Почему не используются локальные модели

Stage 6 намеренно ограничен **API-моделями и удалёнными агентами**. Локальный inference (Ollama, llama.cpp, GGUF/ONNX) не входит в область видимости:

- Локальные модели требуют специфичного окружения (GPU, большой RAM), несовместимого с воспроизводимостью benchmark.
- `provider: ollama` отклоняется при валидации `LlmConfig` с `ValidationError`.
- Поддержка локальных моделей может быть добавлена в отдельном этапе как независимый `LlmClient`.

---

## 3. Поддерживаемые provider-типы

### `openrouter`

OpenAI-compatible endpoint `https://openrouter.ai/api/v1`. API-ключ читается из env-переменной, указанной в `api_key_env` (по умолчанию `OPENROUTER_API_KEY`). Поддерживает любую модель, доступную через OpenRouter.

### `openai_compatible`

Универсальный клиент для любого OpenAI-compatible API. `base_url` обязателен. Используется для OpenAI, Azure OpenAI, Ollama-proxy, vLLM и других совместимых серверов.

### `anthropic`

Прямой клиент к Anthropic Messages API (`https://api.anthropic.com/v1/messages`). Аутентификация через заголовок `x-api-key`. Системные сообщения автоматически выносятся в поле `system`. Инструменты адаптируются из OpenAI-формата в Anthropic-формат (`input_schema`).

### `remote_agent`

Стратегия делегирования: вместо прямого LLM-вызова задача передаётся внешнему агенту через HTTP или дочерний процесс. Используется для Codex, Claude Code и любого `generic_http`/`generic_command` агента.

### `mock`

Детерминированный клиент для unit-тестов. Возвращает заранее заданную последовательность `LlmResponse`. Не требует API-ключей и внешних вызовов.

---

## 4. Формат AgentConfig

```yaml
agent_id: my_agent              # уникальный идентификатор
strategy: react                 # direct_tool_calling | react | plan_and_execute | remote_agent
mcp_profile: minimal            # профиль инструментов MCP
llm:                            # обязателен для всех стратегий кроме remote_agent
  provider: anthropic
  model: claude-3-5-sonnet-latest
  api_key_env: ANTHROPIC_API_KEY
  temperature: 0.2
  top_p: 0.9
  max_tokens: 2048
  timeout_sec: 120
max_steps: 30                   # лимит шагов стратегии
max_retries: 1                  # повторы при ошибке шага
step_timeout_sec: 120           # таймаут одного шага (сек)
allow_python_tools: false       # разрешить execute_blender_code
allow_inspection_tools: true    # разрешить get_scene_info и аналоги
allowed_tools: []               # явный allowlist инструментов (пусто = все разрешённые)
trace_enabled: true             # писать agent_trace.json
```

Готовые конфиги находятся в [`configs/agents/`](../configs/agents/).

---

## 5. Формат LlmConfig

```yaml
provider: openrouter             # openrouter | openai_compatible | anthropic | mock
model: openai/gpt-4.1-mini       # идентификатор модели у провайдера
base_url: https://openrouter.ai/api/v1  # обязателен для openai_compatible
api_key_env: OPENROUTER_API_KEY  # имя env-переменной с ключом (не сам ключ)
temperature: 0.2                 # 0.0–2.0
top_p: 0.9                       # 0.0–1.0
max_tokens: 2048                 # максимум токенов в ответе
timeout_sec: 120                 # HTTP-таймаут запроса
extra_headers: {}                # дополнительные HTTP-заголовки
```

Ключ никогда не сохраняется в `AgentTrace` — только имя env-переменной.

---

## 6. Формат RemoteAgentConfig

```yaml
provider: generic_http           # codex | claude_code | generic_http | generic_command | mock
agent_id: my_remote_agent
endpoint_url: https://agent.example/run   # обязателен для generic_http
api_key_env: REMOTE_AGENT_KEY
command: claude                  # обязателен для generic_command
args: ["--print", "--json"]
workspace_dir: /tmp/agent_ws
timeout_sec: 300
```

Провайдеры `codex` и `claude_code` требуют `metadata.transport` (`generic_http` или `generic_command`) для маппинга на реальный транспорт.

---

## 7. Direct Tool-Calling

Самая простая стратегия: один LLM-вызов → разбор `tool_calls` → исполнение инструментов → финальный ответ. Нет итеративного цикла планирования.

```
LLM call → tool_calls → execute tools → LLM call (with results) → final
```

Останавливается при достижении `max_steps` или когда LLM возвращает ответ без `tool_calls`.

Конфиг: [`configs/agents/direct_openrouter.yaml`](../configs/agents/direct_openrouter.yaml)

---

## 8. ReAct

Реализует цикл Reason + Act: на каждом шаге агент получает наблюдение (результат инструмента или сообщение об ошибке) и решает следующий шаг. Продолжает до финального ответа или `max_steps`.

```
[Thought → Action → Observation] × N → Final Answer
```

Ошибки инструментов передаются обратно как наблюдения — агент может исправить аргументы или выбрать другой инструмент.

Конфиг: [`configs/agents/react_anthropic.yaml`](../configs/agents/react_anthropic.yaml)

---

## 9. Plan-and-Execute

Двухфазная стратегия:

1. **Plan** — LLM генерирует JSON-план в виде упорядоченного списка шагов.
2. **Execute** — каждый шаг плана исполняется последовательно.

Формат плана:

```json
{
  "plan": [
    {"step": 1, "description": "Inspect scene", "tool": "get_scene_info", "arguments": {}},
    {"step": 2, "description": "Add cube",      "tool": "create_object",  "arguments": {"type": "CUBE"}}
  ]
}
```

Если LLM возвращает невалидный JSON или план не содержит шагов — выбрасывается `LlmResponseParseError`.

Конфиг: [`configs/agents/plan_execute_openrouter.yaml`](../configs/agents/plan_execute_openrouter.yaml)

---

## 10. Remote Agent mode

Задача передаётся внешнему агенту целиком — локальный LLM не вызывается.

**HTTP-транспорт** (`generic_http`): POST-запрос на `endpoint_url` с JSON-телом `{task, tool_contracts, mcp_config_path, output_dir}`. Ответ: `{ok, scene_snapshot_path, error}`.

**Command-транспорт** (`generic_command`): дочерний процесс получает JSON через stdin, возвращает JSON в stdout.

```yaml
strategy: remote_agent
remote_agent:
  provider: generic_http
  agent_id: my_agent
  endpoint_url: https://agent.example/run
  api_key_env: AGENT_KEY
  timeout_sec: 300
```

---

## 11. Безопасность: no_python / minimal

| Флаг | Поведение |
|---|---|
| `allow_python_tools: false` | Инструмент `execute_blender_code` исключён из схем и prompt |
| `allow_inspection_tools: true` | `get_scene_info` и read-only инструменты доступны |
| `mcp_profile: minimal` | Только базовый набор инструментов из MCP-профиля |

Попытка включить `execute_blender_code` при `allow_python_tools: false` приводит к `ValidationError` при загрузке конфига. Попытка вызова запрещённого инструмента во время выполнения — к `ToolInvocationError`.

---

## 12. Формат AgentTrace

Trace записывается как JSON в `artifacts/agent_runs/<run_id>/agent_trace.json`.

```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "task_id": "geometry_001_basic_primitives",
  "agent_id": "react_anthropic",
  "strategy": "react",
  "model": "claude-3-5-sonnet-latest",
  "steps": [
    {
      "step_index": 0,
      "step_type": "llm_call",
      "thought": "I need to inspect the scene first.",
      "tool_name": "get_scene_info",
      "tool_arguments": {},
      "observation": {"objects": ["Cube", "Light", "Camera"]},
      "duration_sec": 1.2
    }
  ],
  "final_message": "Scene contains 3 objects.",
  "success": true,
  "started_at": "2025-05-16T12:00:00Z",
  "finished_at": "2025-05-16T12:00:05Z",
  "duration_sec": 5.0
}
```

Типы шагов: `llm_call`, `tool_call`, `observation`, `plan`, `final`, `error`.

Чтение/запись:

```python
from benchmark.agent.trace import read_agent_trace, write_agent_trace, summarize_trace

trace = read_agent_trace("artifacts/agent_runs/abc/agent_trace.json")
summary = summarize_trace(trace)
# {'run_id': ..., 'steps_count': 3, 'tool_calls_count': 2, 'errors_count': 0, ...}
```

---

## 13. CLI

```bash
# Запустить агента на одной задаче
python -m benchmark.agent.cli run \
  --task tasks/geometry/geometry_001_basic_primitives.yaml \
  --agent-config configs/agents/react_anthropic.yaml \
  --output-dir artifacts/

# Просмотреть сводку trace
python -m benchmark.agent.cli trace-summary \
  --trace artifacts/agent_runs/<run_id>/agent_trace.json

# Список стратегий
python -m benchmark.agent.cli list-strategies

# Список провайдеров (ollama отсутствует)
python -m benchmark.agent.cli list-providers
```

Пример вывода `run`:

```
ok: true
trace_path: artifacts/agent_runs/550e8400.../agent_trace.json
```

Пример вывода `trace-summary`:

```
run_id: 550e8400-...
task_id: geometry_001_basic_primitives
strategy: react
success: True
steps: 4
tools: 2
errors: 0
```

---

## 14. Интеграция с ExperimentRunner

`AgentExecutionBackend` реализует интерфейс `ExecutionBackend` и регистрируется для режимов `AGENT_MCP` и `REMOTE_AGENT`.

```python
from benchmark.agent.execution_backend import AgentExecutionBackend
from benchmark.runner.models import RunConfig, ExecutionMode

backend = AgentExecutionBackend(agent_config_path="configs/agents/react_anthropic.yaml")
result = backend.execute(RunConfig(
    task_id="geometry_001",
    task_path=Path("tasks/geometry/geometry_001_basic_primitives.yaml"),
    execution_mode=ExecutionMode.AGENT_MCP,
    output_dir=Path("artifacts/"),
    agent_config_path=Path("configs/agents/react_anthropic.yaml"),
))
# result.ok, result.scene_snapshot_path, result.artifacts_dir
```

`AgentRunResult.trace_path` пробрасывается в `ExecutionResult.output_files`, что позволяет `ExperimentRunner` ссылаться на trace для дальнейшей валидации.

---

## 15. Pytest markers

| Маркер | Когда применять |
|---|---|
| `@pytest.mark.llm` | Тест требует реального LLM API-ключа |
| `@pytest.mark.remote_agent` | Тест требует запущенного удалённого агента |
| `@pytest.mark.agent_integration` | Тест требует MCP + Blender + внешний API/агент |
| `@pytest.mark.mcp` | Тест требует запущенного MCP-сервера |
| `@pytest.mark.blender` | Тест требует установленного Blender |

**Обычный запуск** (без внешних зависимостей):

```bash
pytest
# эквивалентно:
pytest -m "not llm and not remote_agent and not agent_integration"
```

**Запуск с реальным LLM:**

```bash
ANTHROPIC_API_KEY=sk-ant-... pytest -m llm
```

**Запуск всех тестов включая интеграционные:**

```bash
pytest -m ""
```
