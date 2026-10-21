# Контракты переходов · v0.1.0

Все примеры синтетические. C00–C24 — предлагаемые внутренние операции, не HTTP API платформы. Общие правила находятся в PROTOCOL.md.

## C00 · Создать запуск

**CLI → Runtime** · владелец A

`POST /v1/runs`

Python: `runtime.start`

Таймаут операции: 10 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: run_id.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c00",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c00",
  "payload": {
    "task_path": "/task/task.md",
    "work_dir": "/work/run-demo-001",
    "time_budget_seconds": 10800,
    "max_workers": 2
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c00",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "state": "initializing",
    "task_path": "/task/task.md",
    "work_dir": "/work/run-demo-001",
    "time_budget_seconds": 10800
  },
  "error": null
}
```

### Правила

- Внешний интерфейс платформы — argv и env. Этот JSON создаёт CLI-адаптер внутри решения.
- Дедлайн отсчитывается от самого начала solution.py через монотонные часы, включая импорты и LLM.
- Каждый запуск имеет новый run_id и отдельное рабочее пространство.

Ошибки: `TASK_NOT_FOUND`, `INVALID_CONFIG`.

Схемы: `#/$defs/C00_Request`, `#/$defs/C00_Response` в contracts.schema.json. Пример ошибки — examples/C00/error.json.

## C01 · Прочитать условие и пути

**Runtime → TaskReader** · владелец B

`POST /v1/tasks/read`

Python: `task_reader.read`

Таймаут операции: 10 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: read.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c01",
  "run_id": "run-demo-001",
  "idempotency_key": null,
  "payload": {
    "task_path": "/task/task.md"
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c01",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "task_text_artifact_id": "art-task-md",
    "task_text": "Иллюстративная задача: предсказать label; основная метрика Macro F1.",
    "task_root": "/task",
    "candidate_paths": {
      "train": "/task/train.csv",
      "test": "/task/test.csv",
      "dictionary": "/task/data_dictionary.csv",
      "sample_submission": "/task/sample_submission.csv"
    }
  },
  "error": null
}
```

### Правила

- UTF-8; относительные пути разрешаются от каталога task.md.
- Если путь из условия не существует, допускается только однозначное разрешение файла внутри каталога задачи с записью решения в лог. Неоднозначность возвращает ошибку.
- Исходный текст сохраняется без переписывания LLM.

Ошибки: `TASK_NOT_FOUND`, `AMBIGUOUS_PATH`.

Схемы: `#/$defs/C01_Request`, `#/$defs/C01_Response` в contracts.schema.json. Пример ошибки — examples/C01/error.json.

## C02 · Извлечь смысл задачи

**Runtime → TaskInterpreter** · владелец B

`POST /v1/tasks/interpret`

Python: `task_interpreter.interpret`

Таймаут операции: 90 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: read.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c02",
  "run_id": "run-demo-001",
  "idempotency_key": null,
  "payload": {
    "task_text_artifact_id": "art-task-md",
    "candidate_paths": {
      "train": "/task/train.csv",
      "test": "/task/test.csv"
    },
    "tools_catalog_version": "tools-v1"
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c02",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "draft": {
      "task_kind": "multiclass_classification",
      "target_column": "label",
      "metric": {
        "metric_id": "f1_macro",
        "direction": "maximize",
        "prediction_kind": "labels",
        "parameters": {
          "labels": [
            "A",
            "B",
            "C"
          ],
          "zero_division": 0
        }
      },
      "output": {
        "path": "/task/submission.csv",
        "columns": [
          "id",
          "label"
        ],
        "id_columns": [
          "id"
        ],
        "prediction_columns": [
          "label"
        ],
        "row_order": "test",
        "encoding": "utf-8",
        "separator": ",",
        "write_index": false
      }
    },
    "evidence": [
      "Тип задачи и метрика указаны в условии."
    ],
    "unresolved_fields": []
  },
  "error": null
}
```

### Правила

- LLM вызывается через C10 с purpose=task_interpretation.
- Структурированный ответ проверяется локально; ограниченное исправление JSON расходует общий бюджет.
- Схема не разрешает молча заменить неизвестную метрику привычной.

Ошибки: `LLM_UNAVAILABLE`, `INVALID_LLM_OUTPUT`, `UNSUPPORTED_METRIC`.

Схемы: `#/$defs/C02_Request`, `#/$defs/C02_Response` в contracts.schema.json. Пример ошибки — examples/C02/error.json.

## C03 · Получить технический профиль

**Runtime → DataProfiler** · владелец C

`POST /v1/data/profile`

Python: `tools.profile_data`

Таймаут операции: 120 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: read.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c03",
  "run_id": "run-demo-001",
  "idempotency_key": null,
  "payload": {
    "candidate_paths": {
      "train": "/task/train.csv",
      "test": "/task/test.csv"
    },
    "max_report_bytes": 24000
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c03",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "dataset_id": "dataset-demo-v1",
    "profile_artifact_id": "art-profile",
    "train_artifact_id": "art-train",
    "test_artifact_id": "art-test",
    "dictionary_artifact_id": "art-dictionary",
    "sample_submission_artifact_id": "art-sample",
    "summary": {
      "train_rows": 1000,
      "test_rows": 200,
      "train_columns": [
        "id",
        "x_numeric",
        "x_category",
        "label"
      ],
      "test_columns": [
        "id",
        "x_numeric",
        "x_category"
      ]
    }
  },
  "error": null
}
```

### Правила

- Все размеры, названия и значения в этом пакете синтетические; приложенные CSV при подготовке схемы не читались.
- Большие таблицы и отчёты сохраняются артефактами, в контекст передаётся ограниченная сводка.
- dataset_id привязан к содержимому и схеме файлов; повторное изменение данных создаёт новую версию.

Ошибки: `DATA_NOT_FOUND`, `INVALID_TABLE`.

Схемы: `#/$defs/C03_Request`, `#/$defs/C03_Response` в contracts.schema.json. Пример ошибки — examples/C03/error.json.

## C04 · Зафиксировать контракт задачи

**Runtime → TaskRegistry** · владелец C

`POST /v1/tasks/resolve`

Python: `task_registry.resolve`

Таймаут операции: 10 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: dataset_id + normalized TaskSpec.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c04",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c04",
  "payload": {
    "interpretation_request_id": "req-c02",
    "profile_artifact_id": "art-profile",
    "dataset_id": "dataset-demo-v1"
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c04",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "task_spec": {
      "task_spec_id": "task-v1",
      "dataset_id": "dataset-demo-v1",
      "task_kind": "multiclass_classification",
      "target_column": "label",
      "train_artifact_id": "art-train",
      "test_artifact_id": "art-test",
      "dictionary_artifact_id": "art-dictionary",
      "sample_submission_artifact_id": "art-sample",
      "metric": {
        "metric_id": "f1_macro",
        "direction": "maximize",
        "prediction_kind": "labels",
        "parameters": {
          "labels": [
            "A",
            "B",
            "C"
          ],
          "zero_division": 0
        }
      },
      "allowed_labels": [
        "A",
        "B",
        "C"
      ],
      "output": {
        "path": "/task/submission.csv",
        "columns": [
          "id",
          "label"
        ],
        "id_columns": [
          "id"
        ],
        "prediction_columns": [
          "label"
        ],
        "row_order": "test",
        "encoding": "utf-8",
        "separator": ",",
        "write_index": false
      },
      "excluded_features": [
        "id"
      ]
    },
    "adapter_id": "multiclass-v1",
    "metric_implementation_id": "sklearn-f1-macro-v1"
  },
  "error": null
}
```

### Правила

- TaskSpec проверяет код: наличие таргета, входных файлов, направления метрики и формата вывода.
- Первый MVP поддерживает регрессию, бинарную и многоклассовую классификацию; список расширяется адаптерами.
- Неизвестный тип или метрика возвращает явную ошибку. Приватные задачи не обещаны организаторами как только эти типы.

Ошибки: `TASK_SPEC_CONFLICT`, `UNSUPPORTED_TASK_KIND`, `UNSUPPORTED_METRIC`.

Схемы: `#/$defs/C04_Request`, `#/$defs/C04_Response` в contracts.schema.json. Пример ошибки — examples/C04/error.json.

## C05 · Создать общую валидацию

**Orchestrator → Evaluator** · владелец C

`POST /v1/validations`

Python: `evaluation.create_protocol`

Таймаут операции: 120 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: dataset_id + proposal + metric.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c05",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c05",
  "payload": {
    "task_spec_id": "task-v1",
    "dataset_id": "dataset-demo-v1",
    "proposal": {
      "strategy": "stratified_holdout",
      "parameters": {
        "valid_fraction": 0.2
      },
      "seed": 42,
      "reason": "Иллюстративный протокол; реальный выбирается после анализа текущей задачи."
    }
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c05",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "validation_id": "validation-v1",
    "train_rows_artifact_id": "art-split-train",
    "valid_rows_artifact_id": "art-split-valid",
    "labels_visibility": "evaluator_only",
    "metric": {
      "metric_id": "f1_macro",
      "direction": "maximize",
      "prediction_kind": "labels",
      "parameters": {
        "labels": [
          "A",
          "B",
          "C"
        ],
        "zero_division": 0
      }
    },
    "protocol_sha256": "d370c5d2d3c183bd576fb9d064a281ea31a14fdc08bce428f15ce4919612dbcb"
  },
  "error": null
}
```

### Правила

- Стратегия в примере не является общим требованием; протокол может быть временным, групповым или иным.
- Индексы, seed, метрика и правила предобработки версионируются.
- При смене validation_id кандидаты переоцениваются; метрики разных протоколов не сравниваются. Разделение labels_visibility описывает контракт, а не гарантирует OS-изоляцию произвольного Python.

Ошибки: `INVALID_SPLIT`, `LEAKAGE_POLICY_VIOLATION`.

Схемы: `#/$defs/C05_Request`, `#/$defs/C05_Response` в contracts.schema.json. Пример ошибки — examples/C05/error.json.

## C06 · Получить рецепты и инструменты

**Orchestrator → Catalog** · владелец B

`POST /v1/catalog/select`

Python: `catalog.select`

Таймаут операции: 10 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: read.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c06",
  "run_id": "run-demo-001",
  "idempotency_key": null,
  "payload": {
    "task_spec_id": "task-v1",
    "adapter_id": "multiclass-v1",
    "cpu_only": true
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c06",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "catalog_version": "recipes-v1",
    "recipe_ids": [
      "dummy-classifier-v1",
      "catboost-classifier-v1",
      "linear-classifier-v1"
    ],
    "tools_catalog_version": "tools-v1",
    "tools_markdown_artifact_id": "art-tools-md",
    "tools": [
      {
        "name": "write_code",
        "description": "Записать Python-код эксперимента в отдельный артефакт.",
        "input_schema": {
          "type": "object",
          "properties": {
            "researcher_id": {
              "type": "string",
              "minLength": 1
            },
            "filename": {
              "type": "string",
              "minLength": 1
            },
            "content": {
              "type": "string",
              "minLength": 1
            },
            "declared_entrypoint": {
              "enum": [
                "make_pipeline"
              ]
            }
          },
          "required": [
            "researcher_id",
            "filename",
            "content",
            "declared_entrypoint"
          ],
          "additionalProperties": false
        },
        "timeout_seconds": 10,
        "side_effect": "artifact_write"
      }
    ]
  },
  "error": null
}
```

### Правила

- Каталог содержит применимость, параметры, ограничения и точки изменения кода.
- Markdown и JSON-схемы инструментов генерируются из одного описания.
- Dummy-рецепт обеспечивает ранний технически корректный ответ; он не считается внешним бейзлайном организаторов.

Ошибки: общие ошибки envelope.

Схемы: `#/$defs/C06_Request`, `#/$defs/C06_Response` в contracts.schema.json. Пример ошибки — examples/C06/error.json.

## C07 · Прочитать память исследования

**Orchestrator → ExperimentStore** · владелец A

`POST /v1/state/snapshot`

Python: `store.snapshot`

Таймаут операции: 10 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: read.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c07",
  "run_id": "run-demo-001",
  "idempotency_key": null,
  "payload": {
    "task_spec_id": "task-v1",
    "validation_id": "validation-v1",
    "limit_recent": 6,
    "limit_best": 3
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c07",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "snapshot_id": "snapshot-001",
    "store_revision": 7,
    "best_evaluated_experiment_id": "exp-baseline",
    "published_champion_experiment_id": "exp-baseline",
    "recent_experiment_ids": [
      "exp-baseline"
    ],
    "reserved_fingerprints": [],
    "summary_artifact_id": "art-research-summary"
  },
  "error": null
}
```

### Правила

- Единый писатель — диспетчер; читатели получают согласованный снимок.
- Snapshot содержит факты и ссылки, а не полную историю stdout.
- best_evaluated и published_champion могут различаться, пока идёт финальное обучение.

Ошибки: общие ошибки envelope.

Схемы: `#/$defs/C07_Request`, `#/$defs/C07_Response` в contracts.schema.json. Пример ошибки — examples/C07/error.json.

## C08 · Собрать компактный контекст

**Orchestrator → ContextBuilder** · владелец B

`POST /v1/contexts`

Python: `context.build`

Таймаут операции: 10 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: read.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c08",
  "run_id": "run-demo-001",
  "idempotency_key": null,
  "payload": {
    "task_spec_id": "task-v1",
    "snapshot_id": "snapshot-001",
    "tools_catalog_version": "tools-v1",
    "researcher_id": "researcher-1",
    "assignment": "exploit_incumbent",
    "max_input_tokens": 8000
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c08",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "context_id": "context-001",
    "context_version": 1,
    "messages_artifact_id": "art-context-001",
    "estimated_input_tokens": 5300,
    "omitted_experiment_count": 0,
    "provider_prefix_cache_required": false
  },
  "error": null
}
```

### Правила

- Постоянный префикс: условие, правила, инструменты; динамическая часть: сводка и поручение.
- Оценка токенов приблизительная, использование API берётся из ответа провайдера, когда оно доступно.
- Локальная память и серверный prefix cache — разные механизмы; поддержка второго не предполагается.

Ошибки: `CONTEXT_TOO_LARGE`.

Схемы: `#/$defs/C08_Request`, `#/$defs/C08_Response` в contracts.schema.json. Пример ошибки — examples/C08/error.json.

## C09 · Выдать исследовательское поручение

**Orchestrator → Researcher** · владелец B

`POST /v1/research/propose`

Python: `researcher.propose`

Таймаут операции: 120 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: context_id + researcher_id.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c09",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c09",
  "payload": {
    "researcher_id": "researcher-1",
    "context_id": "context-001",
    "assignment": "exploit_incumbent",
    "base_experiment_ids": [
      "exp-baseline"
    ],
    "forbidden_fingerprints": [],
    "budget": {
      "wall_seconds_remaining": 9200,
      "finalization_reserve_seconds": 1200,
      "cpu_threads_available": 6,
      "memory_mb_available": 20480,
      "llm_calls_remaining": 30,
      "search_allowed": true
    },
    "proposal_limit": 1
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c09",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "proposal_id": "proposal-001",
    "researcher_id": "researcher-1",
    "plan": {
      "plan_id": "plan-001",
      "researcher_id": "researcher-1",
      "parent_ids": [
        "exp-baseline"
      ],
      "hypothesis": "Сравнить альтернативную регуляризацию с базовым рецептом.",
      "expected_effect": "Улучшение основной метрики при той же валидации.",
      "recipe_id": "catboost-classifier-v1",
      "task_spec_id": "task-v1",
      "dataset_id": "dataset-demo-v1",
      "validation_id": "validation-v1",
      "seed": 42,
      "fidelity": "full",
      "model": {
        "family": "catboost",
        "hyperparameters": {
          "iterations": 300,
          "depth": 6,
          "learning_rate": 0.05
        }
      },
      "feature_steps": [
        {
          "operation": "impute_numeric",
          "strategy": "median"
        }
      ],
      "code_artifact_id": "art-code-001",
      "resources": {
        "cpu_threads": 3,
        "memory_limit_mb": 10240,
        "timeout_seconds": 600
      }
    },
    "decision": "propose",
    "stop_reason": null
  },
  "error": null
}
```

### Правила

- Researcher-1 и Researcher-2 используют одну предоставленную модель с разными контекстами.
- Роль формулирует гипотезу и готовит код через C10/C11; обучение планирует диспетчер.
- decision=propose требует plan; decision=stop требует stop_reason и plan=null. При baseline выбирается быстрый рецепт.

Ошибки: `LLM_UNAVAILABLE`, `INVALID_LLM_OUTPUT`.

Схемы: `#/$defs/C09_Request`, `#/$defs/C09_Response` в contracts.schema.json. Пример ошибки — examples/C09/error.json.

## C10 · Вызвать предоставленную LLM

**TaskInterpreter / Researcher → LLMGateway** · владелец B

`POST /v1/llm/complete`

Python: `llm_gateway.complete`

Таймаут операции: 90 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: at_least_once; API charge may repeat.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c10",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c10",
  "payload": {
    "purpose": "research_proposal",
    "context_artifact_id": "art-context-001",
    "response_schema_id": "ExperimentPlan",
    "tool_schema_ids": [
      "write_code"
    ],
    "max_output_tokens": 3000,
    "deadline_seconds": 90,
    "attempt_limit": 2
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c10",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "completion_id": "completion-001",
    "output_artifact_id": "art-llm-output-001",
    "parsed_payload": {
      "plan_id": "plan-001",
      "researcher_id": "researcher-1",
      "parent_ids": [
        "exp-baseline"
      ],
      "hypothesis": "Сравнить альтернативную регуляризацию с базовым рецептом.",
      "expected_effect": "Улучшение основной метрики при той же валидации.",
      "recipe_id": "catboost-classifier-v1",
      "task_spec_id": "task-v1",
      "dataset_id": "dataset-demo-v1",
      "validation_id": "validation-v1",
      "seed": 42,
      "fidelity": "full",
      "model": {
        "family": "catboost",
        "hyperparameters": {
          "iterations": 300,
          "depth": 6,
          "learning_rate": 0.05
        }
      },
      "feature_steps": [
        {
          "operation": "impute_numeric",
          "strategy": "median"
        }
      ],
      "code_artifact_id": "art-code-001",
      "resources": {
        "cpu_threads": 3,
        "memory_limit_mb": 10240,
        "timeout_seconds": 600
      }
    },
    "tool_calls": [],
    "usage": {
      "input_tokens": 5300,
      "output_tokens": 1200
    },
    "attempts_used": 1
  },
  "error": null
}
```

### Правила

- Внешний API вызывается адаптером OpenAI-совместимого клиента из baseline; модель и URL берутся из env.
- Здесь показан внутренний контракт. Тело внешнего API формирует адаптер messages/tools; организаторы не принимают этот JSON.
- parsed_payload проверяется против response_schema_id до передачи вызывающему коду; пример ниже является полным ExperimentPlan.
- Общий лимит запросов, таймаут, ограниченные retry с backoff; ключи не попадают в логи. На каждом retry проверяется оставшийся бюджет.

Ошибки: `LLM_RATE_LIMIT`, `LLM_UNAVAILABLE`, `INVALID_LLM_OUTPUT`, `BUDGET_EXHAUSTED`.

Схемы: `#/$defs/C10_Request`, `#/$defs/C10_Response` в contracts.schema.json. Пример ошибки — examples/C10/error.json.

## C11 · Сохранить код эксперимента

**Researcher → ArtifactStore** · владелец C

`POST /v1/artifacts/code`

Python: `tools.write_code`

Таймаут операции: 10 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: run_id + code sha256.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c11",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c11",
  "payload": {
    "researcher_id": "researcher-1",
    "filename": "experiment.py",
    "content": "def make_pipeline(task_spec, params):\n    from catboost import CatBoostClassifier\n    return CatBoostClassifier(**params, thread_count=3, verbose=False)\n",
    "declared_entrypoint": "make_pipeline"
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c11",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "artifact": {
      "artifact_id": "art-code-001",
      "run_id": "run-demo-001",
      "kind": "python_source",
      "relative_path": "experiments/plan-001/experiment.py",
      "sha256": "fe1b98a2038e60db754a844569a56ffeade70fed57fc70881fc2839f330413b7",
      "size_bytes": 153,
      "media_type": "text/x-python"
    },
    "syntax_valid": true,
    "entrypoint": "make_pipeline"
  },
  "error": null
}
```

### Правила

- Код — иллюстрация формы, не готовый универсальный рецепт. Категориальные признаки и fit kwargs определяет рецепт.
- В MVP код экспортирует make_pipeline(task_spec: dict, params: dict) -> объект с fit/predict; runner владеет загрузкой, разбиением и вызовом fit.
- Версия протокола fit/predict и разрешённые model fit kwargs фиксируются в recipes-v1. Число потоков runner задаёт из lease.
- Только относительный filename внутри рабочего каталога; никаких изменений task.md или итогового CSV.
- Проверка синтаксиса не гарантирует успешное обучение и не является песочницей безопасности.

Ошибки: `INVALID_CODE`, `PATH_OUTSIDE_WORKSPACE`.

Схемы: `#/$defs/C11_Request`, `#/$defs/C11_Response` в contracts.schema.json. Пример ошибки — examples/C11/error.json.

## C12 · Зарезервировать уникальный эксперимент

**Researcher → Orchestrator** · владелец A

`POST /v1/experiments`

Python: `orchestrator.submit_plan`

Таймаут операции: 10 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: fingerprint.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c12",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c12",
  "payload": {
    "plan": {
      "plan_id": "plan-001",
      "researcher_id": "researcher-1",
      "parent_ids": [
        "exp-baseline"
      ],
      "hypothesis": "Сравнить альтернативную регуляризацию с базовым рецептом.",
      "expected_effect": "Улучшение основной метрики при той же валидации.",
      "recipe_id": "catboost-classifier-v1",
      "task_spec_id": "task-v1",
      "dataset_id": "dataset-demo-v1",
      "validation_id": "validation-v1",
      "seed": 42,
      "fidelity": "full",
      "model": {
        "family": "catboost",
        "hyperparameters": {
          "iterations": 300,
          "depth": 6,
          "learning_rate": 0.05
        }
      },
      "feature_steps": [
        {
          "operation": "impute_numeric",
          "strategy": "median"
        }
      ],
      "code_artifact_id": "art-code-001",
      "resources": {
        "cpu_threads": 3,
        "memory_limit_mb": 10240,
        "timeout_seconds": 600
      }
    }
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c12",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "experiment_id": "exp-001",
    "fingerprint": "fe0070161cc652a36ed0b8e9ab1fd0f85279638bc1ff291c4667c37105d0c94d",
    "disposition": "created",
    "state": "queued",
    "existing_experiment_id": null
  },
  "error": null
}
```

### Правила

- SQLite UNIQUE(run_id, fingerprint) и транзакция создают единственную логическую запись.
- Fingerprint = SHA256 канонического JSON: dataset_id, task_spec_id, validation_id, recipe_version, code_sha256, model params, feature steps, seed, fidelity. Все defaults раскрываются; worker, текст гипотезы и пути исключаются.
- disposition=duplicate возвращает существующий experiment_id; новый процесс не запускается.
- Повтор неудачного запуска — новая attempt_no внутри той же записи, после подтверждённой остановки прошлого процесса.

Ошибки: `INVALID_PLAN`, `PARENT_NOT_FOUND`, `STALE_VALIDATION`.

Схемы: `#/$defs/C12_Request`, `#/$defs/C12_Response` в contracts.schema.json. Пример ошибки — examples/C12/error.json.

## C13 · Получить или продлить ресурсы

**Orchestrator / Finalizer → BudgetManager** · владелец A

`POST /v1/resources/lease`

Python: `budget.acquire_or_renew`

Таймаут операции: 10 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: owner_id + phase + operation + token.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c13",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c13",
  "payload": {
    "operation": "acquire",
    "owner_id": "exp-001",
    "phase": "train",
    "requested": {
      "cpu_threads": 3,
      "memory_limit_mb": 10240,
      "timeout_seconds": 600
    },
    "lease_id": null,
    "fencing_token": null
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c13",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "granted": true,
    "lease": {
      "lease_id": "lease-001",
      "fencing_token": 1,
      "cpu_threads": 3,
      "memory_limit_mb": 10240,
      "timeout_seconds": 600,
      "heartbeat_interval_seconds": 10,
      "expires_at": "2026-10-21T12:20:00Z"
    },
    "retry_after_seconds": null,
    "budget": {
      "wall_seconds_remaining": 9200,
      "finalization_reserve_seconds": 1200,
      "cpu_threads_available": 6,
      "memory_mb_available": 20480,
      "llm_calls_remaining": 30,
      "search_allowed": true
    }
  },
  "error": null
}
```

### Правила

- Стартовые внутренние лимиты: два обучения по 3 CPU и до 10240 MiB, 2 CPU и 8192 MiB оставлены служебным процессам. Это настройка, не требование ТЗ.
- timeout <= min(запрошенный лимит, остаток до дедлайна фазы). Запрет нового поиска учитывает запас на refit и запись.
- renew/release требуют актуальных lease_id и fencing_token; release идемпотентен. heartbeat каждые 10 секунд, lease TTL 30 секунд.
- Истёкшая аренда не освобождает ресурсы для нового обучения до подтверждённой остановки дерева процессов; поздние сообщения старой попытки отклоняются.

Ошибки: `BUDGET_EXHAUSTED`, `STALE_LEASE`.

Схемы: `#/$defs/C13_Request`, `#/$defs/C13_Response` в contracts.schema.json. Пример ошибки — examples/C13/error.json.

## C14 · Запустить обучение на split

**Orchestrator → Runner** · владелец A

`POST /v1/jobs/train`

Python: `runner.start_train`

Таймаут операции: 10 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: experiment_id + attempt_no.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c14",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c14",
  "payload": {
    "experiment_id": "exp-001",
    "attempt_no": 1,
    "plan": {
      "plan_id": "plan-001",
      "researcher_id": "researcher-1",
      "parent_ids": [
        "exp-baseline"
      ],
      "hypothesis": "Сравнить альтернативную регуляризацию с базовым рецептом.",
      "expected_effect": "Улучшение основной метрики при той же валидации.",
      "recipe_id": "catboost-classifier-v1",
      "task_spec_id": "task-v1",
      "dataset_id": "dataset-demo-v1",
      "validation_id": "validation-v1",
      "seed": 42,
      "fidelity": "full",
      "model": {
        "family": "catboost",
        "hyperparameters": {
          "iterations": 300,
          "depth": 6,
          "learning_rate": 0.05
        }
      },
      "feature_steps": [
        {
          "operation": "impute_numeric",
          "strategy": "median"
        }
      ],
      "code_artifact_id": "art-code-001",
      "resources": {
        "cpu_threads": 3,
        "memory_limit_mb": 10240,
        "timeout_seconds": 600
      }
    },
    "lease": {
      "lease_id": "lease-001",
      "fencing_token": 1,
      "cpu_threads": 3,
      "memory_limit_mb": 10240,
      "timeout_seconds": 600,
      "heartbeat_interval_seconds": 10,
      "expires_at": "2026-10-21T12:20:00Z"
    },
    "train_rows_artifact_id": "art-split-train",
    "valid_rows_artifact_id": "art-split-valid",
    "output_kinds": [
      "validation_predictions",
      "fitted_pipeline",
      "execution_log"
    ]
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c14",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "job_id": "job-train-001",
    "state": "accepted",
    "event_target": "orchestrator.on_job_event",
    "heartbeat_seconds": 10
  },
  "error": null
}
```

### Правила

- Ответ accepted подтверждает постановку в исполнение, а не успешное обучение.
- Все обучаемые преобразования fit выполняются только на train-части; validation используется через transform/predict.
- Runner ограничивает процессы и потоки; cwd и выходные файлы отдельные для experiment_id/attempt_no.
- Предсказания содержат __row_index и prediction; для вероятностей — согласованные колонки классов. Метрики runner не утверждает.

Ошибки: `STALE_LEASE`, `WORKER_START_FAILED`.

Схемы: `#/$defs/C14_Request`, `#/$defs/C14_Response` в contracts.schema.json. Пример ошибки — examples/C14/error.json.

## C15 · Доставить событие выполнения

**Runner → Orchestrator / Finalizer** · владелец A

`POST /v1/jobs/events`

Python: `runtime.on_job_event`

Таймаут операции: 10 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: event_id.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c15",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c15",
  "payload": {
    "event": {
      "event_id": "evt-job-train-001-done",
      "job_id": "job-train-001",
      "experiment_id": "exp-001",
      "attempt_no": 1,
      "lease_id": "lease-001",
      "fencing_token": 1,
      "state": "succeeded",
      "output_artifact_ids": [
        "art-valid-predictions-001",
        "art-fitted-pipeline-001",
        "art-log-001"
      ],
      "duration_seconds": 84.2,
      "peak_memory_mb": 2300,
      "error": null
    }
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c15",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "event_id": "evt-job-train-001-done",
    "disposition": "applied",
    "job_state": "succeeded",
    "experiment_state": "trained",
    "submission_state": null
  },
  "error": null
}
```

### Правила

- Доставка at-least-once; UNIQUE(run_id,event_id) предотвращает повторную фиксацию.
- Поздние события с устаревшим fencing_token отклоняются. C15 также доставляет завершение refit в Finalizer по job_id.
- Для train success переводит эксперимент в trained; для refit меняется состояние submission job, evaluated-эксперимент не откатывается в trained.
- При failed/timed_out/cancelled error обязателен; ресурсы освобождаются после остановки всего дерева процессов.

Ошибки: `STALE_LEASE`, `UNKNOWN_JOB`, `ARTIFACT_MISSING`.

Схемы: `#/$defs/C15_Request`, `#/$defs/C15_Response` в contracts.schema.json. Пример ошибки — examples/C15/error.json.

## C16 · Независимо оценить предсказания

**Orchestrator → Evaluator** · владелец C

`POST /v1/evaluations`

Python: `evaluation.evaluate`

Таймаут операции: 120 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: prediction sha256 + protocol sha256.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c16",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c16",
  "payload": {
    "experiment_id": "exp-001",
    "task_spec_id": "task-v1",
    "dataset_id": "dataset-demo-v1",
    "validation_id": "validation-v1",
    "predictions_artifact_id": "art-valid-predictions-001",
    "fidelity": "full"
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c16",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "evaluation": {
      "evaluation_id": "eval-001",
      "experiment_id": "exp-001",
      "valid": true,
      "primary_metric": {
        "metric_id": "f1_macro",
        "direction": "maximize",
        "value": 0.714,
        "validation_id": "validation-v1",
        "dataset_id": "dataset-demo-v1",
        "fidelity": "full"
      },
      "diagnostics_artifact_id": "art-diagnostics-001",
      "violations": []
    }
  },
  "error": null
}
```

### Правила

- Оценщик проверяет количество и порядок строк, типы, конечность чисел, допустимые классы; затем считает метрику по своим labels.
- Для invalid результата C16 возвращает ok=false с кодом INVALID_PREDICTIONS, эксперимент становится invalid. Numeric metric не выдумывается.
- Сравнимость требует одинаковых dataset_id, validation_id, metric_id, direction, fidelity и версии оценщика.
- Дополнительные диагностические метрики не заменяют основную.

Ошибки: `INVALID_PREDICTIONS`, `STALE_VALIDATION`, `METRIC_FAILED`.

Схемы: `#/$defs/C16_Request`, `#/$defs/C16_Response` в contracts.schema.json. Пример ошибки — examples/C16/error.json.

## C17 · Зафиксировать оценку и выбрать кандидата

**Orchestrator → ExperimentStore** · владелец A

`POST /v1/experiments/commit-evaluation`

Python: `store.commit_evaluation`

Таймаут операции: 10 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: evaluation_id.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c17",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c17",
  "payload": {
    "evaluation_id": "eval-001",
    "experiment_id": "exp-001",
    "selection_policy_id": "same-protocol-primary-v1"
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c17",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "experiment_state": "evaluated",
    "best_evaluated_experiment_id": "exp-001",
    "published_champion_experiment_id": "exp-baseline",
    "should_prepare_submission": true,
    "store_revision": 11
  },
  "error": null
}
```

### Правила

- Оценка берётся из доверенного реестра по evaluation_id, а не из поля score исследователя.
- Выбор и запись происхождения — одна транзакция SQLite. У evaluated-статуса отдельно хранится признак best_evaluated.
- Первый валидный baseline создаёт первоначального кандидата. При равенстве метрики сохраняется более ранний проверенный кандидат.
- Screening-кандидат проходит full-проверку перед продвижением в финальный ответ.

Ошибки: `EVALUATION_NOT_FOUND`, `INCOMPARABLE_EVALUATIONS`.

Схемы: `#/$defs/C17_Request`, `#/$defs/C17_Response` в contracts.schema.json. Пример ошибки — examples/C17/error.json.

## C18 · Подготовить полный ответ кандидата

**Orchestrator → Finalizer** · владелец C

`POST /v1/submissions/prepare`

Python: `submission.prepare`

Таймаут операции: 10 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: experiment_id + task_spec_id + mode.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c18",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c18",
  "payload": {
    "experiment_id": "exp-001",
    "evaluation_id": "eval-001",
    "task_spec_id": "task-v1",
    "mode": "refit_full_train",
    "expected_champion_version": 1
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c18",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "submission_job_id": "subjob-001",
    "state": "preparing",
    "published_champion_experiment_id": "exp-baseline",
    "output_path": "/task/submission.csv"
  },
  "error": null
}
```

### Правила

- Предыдущий готовый submission.csv остаётся доступен всё время refit.
- Одновременно активна одна финализация; более новый кандидат коалесцируется в очередь, чтобы не расходовать ресурс на много refit.
- Finalizer использует C13 для аренды refit и C19 для запуска. При нехватке времени остаётся опубликованный ответ.

Ошибки: `CANDIDATE_NOT_EVALUATED`, `BUDGET_EXHAUSTED`.

Схемы: `#/$defs/C18_Request`, `#/$defs/C18_Response` в contracts.schema.json. Пример ошибки — examples/C18/error.json.

## C19 · Обучить на всём train и предсказать test

**Finalizer → Runner** · владелец A

`POST /v1/jobs/refit`

Python: `runner.start_refit`

Таймаут операции: 10 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: submission_job_id.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c19",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c19",
  "payload": {
    "submission_job_id": "subjob-001",
    "experiment_id": "exp-001",
    "task_spec_id": "task-v1",
    "code_artifact_id": "art-code-001",
    "lease": {
      "lease_id": "lease-refit-001",
      "fencing_token": 1,
      "cpu_threads": 3,
      "memory_limit_mb": 10240,
      "timeout_seconds": 900,
      "heartbeat_interval_seconds": 10,
      "expires_at": "2026-10-21T12:20:00Z"
    },
    "fit_scope": "all_train",
    "predict_artifact_id": "art-test",
    "output_relative_path": "submissions/subjob-001/candidate.csv"
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c19",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "job_id": "job-refit-001",
    "state": "accepted",
    "completion_contract": "C15"
  },
  "error": null
}
```

### Правила

- Параметры и feature pipeline берутся из проверенного experiment_id; после валидации их не меняют незаметно.
- Если рецепт использует early stopping, число итераций фиксируется по валидации; refit не требует скрытых test labels.
- Runner пишет уникальный кандидат, итоговый путь submission.csv ему не выдаётся. Событие C15 связывается с subjob по job_id.

Ошибки: `STALE_LEASE`, `REFIT_FAILED`.

Схемы: `#/$defs/C19_Request`, `#/$defs/C19_Response` в contracts.schema.json. Пример ошибки — examples/C19/error.json.

## C20 · Проверить формат полного ответа

**Finalizer → OutputValidator** · владелец C

`POST /v1/submissions/validate`

Python: `submission.validate_output`

Таймаут операции: 120 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: candidate sha256 + TaskSpec version.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c20",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c20",
  "payload": {
    "submission_job_id": "subjob-001",
    "task_spec_id": "task-v1",
    "candidate_artifact_id": "art-submission-001",
    "reference_test_artifact_id": "art-test"
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c20",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "validation_report_id": "outcheck-001",
    "valid": true,
    "rows": 200,
    "columns": [
      "id",
      "label"
    ],
    "id_order_matches": true,
    "finite_predictions": true,
    "allowed_labels_only": true,
    "candidate_sha256": "e0decf976d8b07013e498ed23ab8279286c75708f527e8ad8f582a04d0f95368"
  },
  "error": null
}
```

### Правила

- Формат и путь берутся из TaskSpec, порядок и id — из test или шаблона в соответствии с условием.
- Проверяются точный набор и порядок колонок, строки, пропуски, дубли идентификаторов там, где они запрещены, значения и классы.
- Неприменимые проверки считаются пройденными: например finite_predictions для строковых меток.
- Любое нарушение возвращает ok=false; прежний опубликованный ответ сохраняется.

Ошибки: `INVALID_OUTPUT_SCHEMA`, `INVALID_OUTPUT_VALUES`, `ID_ORDER_MISMATCH`.

Схемы: `#/$defs/C20_Request`, `#/$defs/C20_Response` в contracts.schema.json. Пример ошибки — examples/C20/error.json.

## C21 · Опубликовать готовый лучший ответ

**Finalizer → Publisher** · владелец C

`POST /v1/submissions/publish`

Python: `submission.publish`

Таймаут операции: 120 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: run_id + candidate sha256.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c21",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c21",
  "payload": {
    "experiment_id": "exp-001",
    "evaluation_id": "eval-001",
    "candidate_artifact_id": "art-submission-001",
    "validation_report_id": "outcheck-001",
    "candidate_sha256": "e0decf976d8b07013e498ed23ab8279286c75708f527e8ad8f582a04d0f95368",
    "expected_champion_version": 1
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c21",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "disposition": "published",
    "champion": {
      "experiment_id": "exp-001",
      "evaluation_id": "eval-001",
      "primary_metric": {
        "metric_id": "f1_macro",
        "direction": "maximize",
        "value": 0.714,
        "validation_id": "validation-v1",
        "dataset_id": "dataset-demo-v1",
        "fidelity": "full"
      },
      "submission_artifact_id": "art-submission-001",
      "output_sha256": "e0decf976d8b07013e498ed23ab8279286c75708f527e8ad8f582a04d0f95368",
      "champion_version": 2
    },
    "output_path": "/task/submission.csv",
    "publication_event_id": "evt-publish-002"
  },
  "error": null
}
```

### Правила

- Единственный Publisher под mutex сравнивает кандидата с текущим опубликованным чемпионом по проверенной основной метрике.
- Сначала проверяются версия, отчёт и hash. Затем уникальный temp в каталоге итогового файла, flush/fsync и os.replace.
- SQLite и filesystem не образуют общую атомарную транзакцию: до replace пишется publish journal, после — champion metadata. Восстановление сверяет hash файла и journal.
- При VERSION_CONFLICT перечитать чемпиона и повторно сравнить; поздний более слабый кандидат не ухудшает ответ.
- Если готового ответа ещё нет, первый валидный baseline допускается независимо от порога улучшения.

Ошибки: `VERSION_CONFLICT`, `CANDIDATE_CHANGED`, `OUTPUT_WRITE_FAILED`, `INCOMPARABLE_EVALUATIONS`.

Схемы: `#/$defs/C21_Request`, `#/$defs/C21_Response` в contracts.schema.json. Пример ошибки — examples/C21/error.json.

## C22 · Остановить поиск по бюджету

**BudgetMonitor → Orchestrator** · владелец A

`POST /v1/runs/control`

Python: `orchestrator.control`

Таймаут операции: 10 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: action + reason + control epoch.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c22",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c22",
  "payload": {
    "action": "stop_search",
    "reason": "finalization_reserve_reached",
    "wall_seconds_remaining": 1200,
    "cancel_job_ids": []
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c22",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "run_state": "finalizing",
    "new_experiments_allowed": false,
    "published_champion_present": true,
    "active_job_ids": [
      "job-refit-001"
    ]
  },
  "error": null
}
```

### Правила

- Монитор работает без LLM. Запас на финализацию оценивается по длительности fit/predict плюс запас на запись, а не навсегда фиксированные 20 минут.
- stop_search запрещает новые эксперименты; cancel_jobs вызывает C24 и только после подтверждения остановки освобождает lease через C13.
- К моменту жёсткого дедлайна ранний готовый ответ должен уже существовать.

Ошибки: `UNKNOWN_JOB`.

Схемы: `#/$defs/C22_Request`, `#/$defs/C22_Response` в contracts.schema.json. Пример ошибки — examples/C22/error.json.

## C23 · Завершить запуск

**Orchestrator → Runtime** · владелец A

`POST /v1/runs/finish`

Python: `runtime.finish`

Таймаут операции: 120 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: run_id + finish.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c23",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c23",
  "payload": {
    "reason": "search_budget_finished",
    "require_valid_output": true
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c23",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "state": "completed",
    "exit_code": 0,
    "output_path": "/task/submission.csv",
    "champion_experiment_id": "exp-001",
    "summary_artifact_id": "art-run-summary"
  },
  "error": null
}
```

### Правила

- До C23 новые задания запрещены, все незавершённые jobs остановлены через C24, активная публикация закончена.
- Runtime проверяет существование и соответствие выходного файла текущему publication record.
- На платформу не вызывается endpoint отправки: процесс завершает работу, платформа читает файл.
- При отсутствии корректного файла ok=false, NO_VALID_SUBMISSION и ненулевой exit code. При доступном baseline и сбое LLM допускается degraded с сохранённым ответом.

Ошибки: `NO_VALID_SUBMISSION`.

Схемы: `#/$defs/C23_Request`, `#/$defs/C23_Response` в contracts.schema.json. Пример ошибки — examples/C23/error.json.

## C24 · Остановить дерево процессов задания

**Orchestrator / Finalizer → Runner** · владелец A

`POST /v1/jobs/cancel`

Python: `runner.cancel`

Таймаут операции: 15 с. Длительная фоновая работа ограничена своим Lease. Идемпотентность: job_id + attempt_no + terminal state.

### Запрос

```json
{
  "schema_version": "1.0",
  "request_id": "req-c24",
  "run_id": "run-demo-001",
  "idempotency_key": "key-c24",
  "payload": {
    "job_id": "job-train-001",
    "experiment_id": "exp-001",
    "attempt_no": 1,
    "fencing_token": 1,
    "reason": "search_deadline_reached",
    "terminal_state": "cancelled",
    "grace_seconds": 5
  }
}
```

### Ответ

```json
{
  "schema_version": "1.0",
  "request_id": "req-c24",
  "run_id": "run-demo-001",
  "ok": true,
  "data": {
    "job_id": "job-train-001",
    "disposition": "stopped",
    "terminal_state": "cancelled",
    "process_tree_stopped": true,
    "remaining_processes": 0,
    "completion_event_id": "evt-job-train-001-cancelled"
  },
  "error": null
}
```

### Правила

- Проверяется token нужной попытки; остановка допустима и после истечения её lease, если процесс ещё существует.
- Сначала мягкая остановка группы процессов, затем принудительная в пределах дедлайна. Ответ stopped допустим только после подтверждения завершения дочерних процессов.
- Терминальное событие доставляется по C15 с тем же completion_event_id. Только после подтверждения остановки BudgetManager освобождает ресурсы через C13 release.
- Повторный cancel завершённого задания возвращает already_stopped и фактический terminal_state, без изменения истории.

Ошибки: `UNKNOWN_JOB`, `STALE_LEASE`, `PROCESS_STOP_FAILED`.

Схемы: `#/$defs/C24_Request`, `#/$defs/C24_Response` в contracts.schema.json. Пример ошибки — examples/C24/error.json.

## Альтернативный TaskSpec: регрессия

```json
{
  "task_spec_id": "task-regression-v1",
  "dataset_id": "dataset-demo-v1",
  "task_kind": "regression",
  "target_column": "target",
  "train_artifact_id": "art-train",
  "test_artifact_id": "art-test",
  "dictionary_artifact_id": "art-dictionary",
  "sample_submission_artifact_id": "art-sample",
  "metric": {
    "metric_id": "rmse",
    "direction": "minimize",
    "prediction_kind": "values",
    "parameters": {}
  },
  "allowed_labels": [],
  "output": {
    "path": "/task/submission.csv",
    "columns": [
      "id",
      "target"
    ],
    "id_columns": [
      "id"
    ],
    "prediction_columns": [
      "target"
    ],
    "row_order": "test",
    "encoding": "utf-8",
    "separator": ",",
    "write_index": false
  },
  "excluded_features": [
    "id"
  ]
}
```
