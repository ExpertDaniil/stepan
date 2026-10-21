# AutoML Agent Skeleton · v0.1.0

Запускаемый каркас для команды из трёх человек. Согласованные C00–C24 реализованы как внутренние Python-вызовы с проверяемыми JSON-обёртками. Платформе передаётся ZIP с `solution.py` в корне; отдельный HTTP-сервер не нужен.

Первый проход создаёт проверенный базовый `submission.csv`, затем два исследовательских контекста предлагают альтернативы. Метрики считает независимый оценщик. Обновление итогового файла выполняет единственный Publisher. Все запросы, ответы, код, контексты и переходы состояний оставляют трассировку.

## Быстрый локальный старт

Нужен Python **3.12**. Рекомендуемая среда для разработки и проверки процессов — Linux или Docker.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/make_demo.py demo_tasks/classification
python solution.py demo_tasks/classification/task.md --llm-mode mock
```

На Windows активируйте `.venv\Scripts\Activate.ps1` в PowerShell либо запускайте `.venv\Scripts\python.exe`. Полную проверку Linux-групп процессов выполняйте в Docker/WSL2.

Демонстрация регрессии:

```bash
python scripts/make_demo.py demo_tasks/regression --kind regression
python solution.py demo_tasks/regression/task.md --llm-mode mock
```

Mock — явно выбранный адаптер **только для синтетических задач** генератора. При ошибке live API переключения в mock нет. Обучение, метрики, процессы и выходной CSV в mock-режиме настоящие; имитируются ответы LLM.

## Предоставленный API

Скопируйте `.env.example` в `.env` и заполните три значения из инструкций организаторов:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_BASE_URL`

```bash
python solution.py /absolute/path/to/task.md --env-file .env
```

Без `--env-file` читаются только переменные процесса. `.env` автоматически не загружается. Существующее окружение имеет приоритет; поддерживаются строки `KEY=value` и кавычки, без shell-подстановок. Значения модели/endpoint не зашиты в код. Все live-запросы идут через один Gateway к указанному сервису.

```bash
python scripts/doctor.py
```

Диагностика показывает наличие API-переменных и совпадение версий библиотек, не раскрывая значения ключей.

## Точки расширения

| Каталог / модуль | Назначение | Владелец |
|---|---|---|
| `solution.py`, `agent/runtime.py` | Запуск, bootstrap, завершение | A |
| `agent/orchestrator.py`, `agent/budget.py` | Очередь и общие лимиты | A |
| `agent/runner.py`, `agent/worker.py` | Процессы обучения и refit | A |
| `agent/store.py`, `agent/schema.sql` | SQLite, DAG экспериментов, публикации | A |
| `agent/contracts.py`, `agent/observability/` | Контракты, трассировка, маскирование | A |
| `agent/llm/`, `agent/task_interpreter.py` | API и понимание задачи | B |
| `agent/context.py`, `agent/researcher.py`, `prompts/` | Контекст и гипотезы | B |
| `agent/catalog.py`, `tools/files.py` | Каталог и чтение условий | B |
| `agent/ml.py`, `recipes/` | Реализации моделей и пространство параметров | C |
| `agent/task.py`, `tools/data.py`, `tools/code.py` | Проверка задачи и инструменты | C |
| `agent/evaluation.py`, `agent/submission.py` | Валидация, метрики, готовый ответ | C |

Интерфейсы и схемы: `contracts/`. Большая схема и исходная спецификация: `docs/architecture/`. Практические правила: `docs/IMPLEMENTATION.md`, `docs/LOGGING.md`, `CONTRIBUTING.md`.

## Логи и результат

Итоговый путь берётся из текущей задачи, обычно `submission.csv` рядом с `task.md`. В `runs/<run_id>/` находятся:

| Путь | Что смотреть |
|---|---|
| `events.jsonl` | Сквозная хронология, trace/span/parent IDs, Cxx, ошибки, время |
| `messages/` | Полные JSON-запросы и ответы между модулями |
| `llm/` | Доступные запросы/ответы API и проверенные структурированные результаты |
| `contexts/` | Контексты исследователей и сводки экспериментов |
| `task/` | Исходное условие, TaskSpec, полный профиль, TOOL-каталог |
| `code/` | Python, реально переданный исполнителю |
| `experiments/` | JobSpec, stdout/stderr, результат процесса, snapshot модели |
| `submissions/` | Кандидаты CSV, проверки и состояния финализации |
| `state.sqlite` | Эксперименты, родители, оценки, события и журнал публикации |
| `summary.md` | Читаемый отчёт по запуску |

```bash
python scripts/show_run.py runs/ИМЯ_ЗАПУСКА --follow
python scripts/show_run.py runs/ИМЯ_ЗАПУСКА --contract C10
python scripts/show_run.py runs/ИМЯ_ЗАПУСКА --experiment exp-ID
```

«Размышления» фиксируются как доступные ответы модели, гипотеза, краткое обоснование и ожидаемый эффект; дополнительные скрытые рассуждения провайдера не предполагаются. Ключ API маскируется, в процессы обучения ключи не передаются. Каталоги запусков имеют отдельные run_id и не переносятся между задачами.

## Настройки

Официальные переменные отделены от внутренних в `.env.example`. Основные внутренние значения:

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `TIME_BUDGET_SECONDS` | 10800 | Полное время, включая импорты, API и вывод; больше 3 часов запрещено |
| `MAX_WORKERS` | 2 | Независимые исследовательские контексты и максимум обучений |
| `TRAIN_CPU_THREADS` | 3 | До 3 потоков на обучение; до 6 суммарно |
| `TRAIN_MEMORY_MB` | 10240 | Лимит RSS на задание; два обучения до 20 GiB |
| `EXPERIMENT_TIMEOUT_SECONDS` | 600 | Запрошенный таймаут одного train-job |
| `MAX_RESEARCH_EXPERIMENTS` | 4 | Число новых обучений после раннего baseline |
| `LLM_MAX_CALLS` | 24 | Общий предел API-попыток, включая retry и tool calls |
| `LLM_TIMEOUT_SECONDS` | 90 | Ограничение операции LLM с учётом остатка времени |
| `MAX_CONTEXT_TOKENS` | 16000 | Консервативный byte-based предел контекста, не точный tokenizer |
| `LOG_MODEL_IO` | 1 | Сохранение raw API snapshots; структурированные планы и события сохраняются всегда |

Запас на финализацию рассчитывается по наблюдаемой длительности fit; на каждом запуске ранний CSV сохраняется до основного поиска. Четыре исследовательских эксперимента — ограничение стартовой версии, а не обязательное требование ТЗ. Качество поиска и бюджет можно развивать независимо.

## Поддерживаемая граница v0.1

- Табличные CSV UTF-8, запятая, одна целевая переменная и одна колонка предсказания.
- Регрессия, бинарная и многоклассовая классификация.
- RMSE, MAE, R², Macro F1, Weighted F1, Accuracy.
- Случайный, стратифицированный, групповой и временной holdout. Временная колонка — числовой порядок либо ISO-8601.
- Dummy, линейный рецепт, HistGradientBoosting, RandomForest; CatBoost включается при наличии разрешённой зависимости.
- Python `make_pipeline(task_spec, params)`, предобработка внутри pipeline, общий runner и оценщик.

Неизвестный тип/метрика/формат возвращает явную ошибку. Приватные задачи заранее неизвестны; дополнительные адаптеры и метрики нужно расширять в реестрах, сохраняя общий протокол. Пример не обещает соревновательное качество без дальнейшего исследования.

## Проверка и упаковка

```bash
python scripts/check.py
python scripts/package_solution.py
```

Второй скрипт создаёт `dist/solution.zip` с `solution.py` непосредственно в корне. В него входят только runtime-модули, рецепты, промпты и схемы. Данные, ключи, логи, тесты и документация исключены. Внешний архив для команды содержит весь репозиторий и готовую копию этого платформенного ZIP в `dist/`.

Для локальной проверки контейнера:

```bash
docker build -t alfa-automl-agent .
docker run --rm --cpus=8 --memory=32g --env-file .env -e WORK_DIR=/runs -v "$PWD/demo_tasks/classification:/task" -v "$PWD/runs:/runs" alfa-automl-agent /task/task.md
```

Для mock-десктопного прогона добавьте к Docker-команде после пути задачи `--llm-mode mock`. Dockerfile — локальный образ; официальную среду запуска предоставляет платформа. Установка пакетов выполняется при подготовке среды, не во время решения задачи.

Фактически выполненные проверки и версии среды зафиксированы в `docs/VALIDATION.md`. Live API и CatBoost требуют проверки в среде с установленными pinned-зависимостями и доступом организаторов.
