# Наблюдаемость

Все записи используют UTC ISO-8601. `run_id` разделяет запуски, `trace_id` объединяет переходы запуска, `request_id` определяет попытку вызова, `span_id`/`parent_span_id` связывают вложенные C09 → C10 → C11. `experiment_id`, `job_id`, `submission_job_id` привязывают события к вычислению и публикации.

`ContractBus.call` проверяет запрос, пишет snapshot, вызывает handler, проверяет и сохраняет ответ. Для invalid JSON snapshot имеет суффикс `.invalid.json` с ограниченным текстовым представлением. При повторе успешного idempotency key handler не исполняется снова; одновременные одинаковые вызовы ждут один результат. SQLite mutations выполняются только основным dispatcher-thread.

Основные события:

- `contract.started/completed/failed/replayed`
- `llm.call_started/completed/failed`, `llm.output_rejected`
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
