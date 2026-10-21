# Наблюдаемость

Все записи используют UTC ISO-8601. `run_id` разделяет запуски, `trace_id` объединяет переходы запуска, `request_id` определяет попытку вызова, `span_id`/`parent_span_id` связывают вложенные C09 → C10 → C11. `experiment_id`, `job_id`, `submission_job_id` привязывают события к вычислению и публикации.

`ContractBus.call` проверяет запрос, пишет snapshot, вызывает handler, проверяет и сохраняет ответ. Для invalid JSON snapshot имеет суффикс `.invalid.json` с ограниченным текстовым представлением. При повторе успешного idempotency key handler не исполняется снова; одновременные одинаковые вызовы ждут один результат. SQLite mutations выполняются только основным dispatcher-thread.

Основные события:

- `contract.started/completed/failed/replayed`
- `llm.call_started/completed/failed`, `llm.output_rejected`
- `llm.daily_budget_exhausted` — сервер сообщил об исчерпании общего бюджета; reset_time_msk=03:00, повторов нет
- `llm.provider_credits_exhausted` — внешний провайдер ответил Insufficient credits; ERROR, без retry и новых вызовов; остаток командного бюджета и время сброса неизвестны
- `context.built`, `research.hypothesis_proposed`
- `artifact.registered`, `experiment.reserved/evaluated/duplicate_skipped`
- `job.started/finished`, C13 renew/release, C24 cancellation
- `submission.state_changed/published/publication_recovered`
- `run.state_changed/finished/failed`

Сначала ищите contract_id и request_id в events.jsonl, затем открывайте пару `messages/<request_id>.request.json` / `.response.json`. Для модели — `llm/NNNN.request.json` и `.response.json`. API snapshots содержат фактически отправленные messages/tools, включая схему ответа. В `contexts/` хранится базовый собранный контекст; Gateway добавляет схему и сообщения ограниченного tool/repair-цикла.

«Мысли» — доступное содержание ответа и явно сформулированные гипотезы/обоснования. Система не восстанавливает недоступные внутренние рассуждения. Если SDK возвращает дополнительные поля, raw snapshot сохраняет их при LOG_MODEL_IO=1. Токены берутся только из API usage; если usage отсутствует, записывается null.

Известный OPENAI_API_KEY и Bearer-значения маскируются перед журналированием. Headers и все env не дампятся; endpoint в событии очищается от userinfo/query. В training env действует allowlist. Structured планы и артефакты остаются доступны при LOG_MODEL_IO=0: этот флаг отключает только raw API snapshots.

stdout/stderr дренируются во время работы процесса и сохраняются раздельно, до 8 MiB на поток. После лимита поток продолжает читаться, чтобы pipe не заблокировал worker. Важные JSONL-события сразу flush-ятся. Контексты, raw ответы и код могут содержать сведения задачи: `runs/` исключён из Git и создаётся приватным каталогом.

На платформе обязательно оценивается итоговый CSV. Сохранение дополнительных файлов после завершения зависит от её интерфейса; поэтому важные статусы также пишутся в stderr. Локально полный run workspace остаётся у команды.

`AMBIGUOUS_PATH` перечисляет спорные пути в message/details и подсказывает выбрать `task.md` одной задачи. Это ошибка разрешения входов C01, до создания Gateway и до API-вызовов. Для проверки структуры без обучения используйте `python3 scripts/list_tasks.py <каталог>`.

## Подробный терминал v0.3

CONSOLE_LOG_FORMAT=human выводит все поля события с run_id/request_id, без прежнего обрезания всей
строки до 600 символов; одно длинное поле ограничено 1600 символами. Полное значение остаётся в JSONL.
CONSOLE_LOG_FORMAT=json выводит ровно JSONL-запись, включая вложенные параметры. LOG_LEVEL управляет
видимостью терминала; events.jsonl сохраняет события независимо от порога. Traceback выводится при
exception-событиях. Redactor применяется до обоих представлений.

LLM: call_no, attempt, purpose, model, max_output_tokens, отдельные request/operation seconds,
finish_reason, content_chars, tool_calls, input/output/reasoning/cache tokens и общий token_budget.
llm.waiting каждые LLM_PROGRESS_SECONDS секунд показывает ожидание; это не оценка прогресса генерации.
llm.output_truncated и llm.output_budget_increased отличают обрыв от синтаксической ошибки JSON.
llm.output_validated означает, что структурированный ответ принят; llm.call_completed лишь фиксирует
получение API-ответа. Raw reasoning не печатается в терминал; доступный provider snapshot сохраняется.

Поиск: search.catalog_warmup, search.round_started, фактические параметры/гипотеза, validation.created,
worker.constructing/fit_started/prediction_completed, job.progress, оценка и относительное улучшение
к baseline. job.finished с failed/timed_out становится ERROR и содержит error/exit_code/stderr_log_path.
search.finished сообщает точную причину остановки: experiment_limit, time_reserve, no_valid_proposals,
round_failed или round_limit. Увеличение TIME_BUDGET_SECONDS само по себе не меняет лимит экспериментов.

v0.3.1 добавляет job.descendants_cleanup_started/completed: job/experiment/phase, exit_code,
число оставшихся процессов и подтверждение остановки группы. Это внутренняя уборка после выхода
worker, до C15/C13 release. Она не устанавливает cancelled. Фактический результат обучения
отдельно сообщает job.finished после проверки exit code/result.json; C20 всё ещё проверяет CSV.

## Граф v0.4

G00–G14/I00–I03 проходят ту же трассировку ContractBus. graph.committed содержит operation и
graph_revision. context.built дополнительно связывает work_order_id/branch_id с контекстом peer.
llm.request_budget_checked проверяет полное отправляемое тело со schema/tools/repair, а не только
исходные messages. graph.proposal_rejected и graph.component_gate_failed не выдают текст LLM за score.

Root events.jsonl остаётся telemetry. Authoritative graph events находятся в SQLite g_events;
их export — exports/<scope>-<cut>/graph-events.jsonl. Каждый event содержит typed record_schema,
проверенный record, run_id, request_id и event_seq. Graph export фиксирует read-cut до создания
самих экспортных артефактов; последующие G09/G11 поэтому не включаются в этот же срез.
