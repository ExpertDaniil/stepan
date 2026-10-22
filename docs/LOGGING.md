# Наблюдаемость

В adaptive `llm.request_budget_checked` содержит `limit_bytes=null` и
`input_limit_owner=provider`: локальная отсечка входного контекста отключена.
`llm.final_decision_requested` отмечает последний ход диалога без инструментов.
`research.context_failed` фиксирует ошибку сборки одного поручения без прерывания всей партии;
если ни один peer не смог начать работу, причина остановки — `context_unavailable`.

В adaptive-режиме `research.analysis_completed` связывает запрос со scope, числом строк и полным
артефактом отчёта; недоступные метки holdout не журналируются. `research.action_proposed` и
`research.action_admitted` фиксируют тип действия, evidence IDs и явный бюджет локальной серии.
`research.stop_requested` фиксирует добровольное завершение. Поручение сохраняется в
`research/intent-plan-*.json`; обычные experiment/evaluation/publication события содержат его
измеренный исход через plan_id/experiment_id. Реальный API-request snapshot учитывает сокращение
после добавления схем и ответов инструментов. Подробности: [OPEN-RESEARCH.md](OPEN-RESEARCH.md).

Исправления adaptive: `search.tuning_base_reused` означает использование известной базы перед
новыми разрешёнными испытаниями. `research.feedback_recorded` сохраняет причину ошибки/отказа
для следующего контекста. `search.fallback_started` явно отмечает резервный каталог после
недоступности API. `submission.refit_failed` с `retained_output=true` сохраняет проверенный CSV
и исключает этот кандидат из повторной финализации. `search.finished.stop_reason` различает
`researchers_stopped`, `research_unavailable` и `no_affordable_candidates`.

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

stdout/stderr дренируются во время работы процесса и сохраняются раздельно, до 1 MiB на поток по умолчанию (WORKER_LOG_MAX_BYTES). После лимита поток продолжает читаться, чтобы pipe не заблокировал worker. Важные JSONL-события сразу flush-ятся. Контексты, raw ответы и код могут содержать сведения задачи: `runs/` исключён из Git и создаётся приватным каталогом.

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
search.finished сообщает причину остановки: time_reserve, no_affordable_candidates либо причину
решения G14 о закрытии бюджета. Старые experiment_limit, round_limit и квоты LLM не завершают поиск.

v0.5 добавляет search.admission: семейство, параметры, прогноз длительности, оставшееся время,
запас на итоговое обучение и решение о допуске. search.duration_observed содержит фактическое
время процесса и fit_seconds, когда оно известно. search.local_trial показывает выбранный локально
вариант и его ветвь; search.proposal_replaced объясняет замену повторного, слишком долгого или
отказного предложения. search.time_reserve_reached означает штатную остановку у границы времени.
submission.refit_deferred сохраняет предыдущий проверенный CSV, если новый итоговый refit уже не помещается.

v0.3.1 добавляет job.descendants_cleanup_started/completed: job/experiment/phase, exit_code,
число оставшихся процессов и подтверждение остановки группы. Это внутренняя уборка после выхода
worker, до C15/C13 release. Она не устанавливает cancelled. Фактический результат обучения
отдельно сообщает job.finished после проверки exit code/result.json; C20 всё ещё проверяет CSV.

## Граф v0.4

G00–G15/I00–I03 проходят ту же трассировку ContractBus. graph.committed содержит operation и
graph_revision. context.built дополнительно связывает work_order_id/branch_id с контекстом peer.
llm.request_budget_checked проверяет полное отправляемое тело со schema/tools/repair, а не только
исходные messages. graph.proposal_rejected и graph.component_gate_failed не выдают текст LLM за score.

Root events.jsonl остаётся telemetry. Authoritative graph events находятся в SQLite g_events;
их export — exports/<scope>-<cut>/graph-events.jsonl. Каждый event содержит typed record_schema,
проверенный record, run_id, request_id и event_seq. Graph export фиксирует read-cut до создания
самих экспортных артефактов; последующие G09/G11 поэтому не включаются в этот же срез.

v0.4.2 добавляет search.decision, graph.idea_registered, graph.branch_forked,
graph.branch_state_changed, graph.plateau_assessed и research.component_transfer_prepared.
В терминале видны причина перехода, обе дочерние ветки, source/base/pilot и статус плато.
research.constraints_applied раскрывает отличие предложенных и эффективных параметров;
context.built показывает assignment, base_experiment_id и сохранённую/пропущенную историю.
Числа плато учитывают только допустимые для ветки references и время после её запуска/возобновления.
llm_call_limit и llm_output_budget в search.finished — штатные причины остановки, проверенные dispatcher.


## Фазы поиска v0.5.1

В v0.5.1 `search.phase_changed` отмечает единственный переход в refine: source_experiment_id,
branch_id, reason, elapsed_seconds, remaining_seconds, phase_limits. `search.round_started`
и `search.finished` содержат phase=search/refine. `research.phase_budget_reached` означает
достижение поискового cap с сохранением резерва доводки. Typed G15/GRefinement в graph-events
позволяет проверить исходные и новые heads; token_budget показывает фазовые и глобальные остатки.

`search.budget_trial_timed_out` отмечает train-job, у которого lease была сокращена остатком
общего бюджета. Попытка остаётся timed_out без метрики; сама по себе она не переводит успешный
запуск с проверенным CSV в degraded. Полный per-job timeout, spawn/validation/API ошибки и
неуспешный refit по-прежнему считаются ошибками запуска.

## Размеры и исследования v0.4.3

graph.branch_seeded показывает реальные стартовые heads детей; authoritative GBranchSeed хранится
в SQLite/events и G11 branch_seeds. graph.plateau_assessed добавляет consecutive_no_gain и required_no_gain.
Контекст содержит ограниченные evaluator diagnostics: ошибки по четырём группам, до шести классов,
без строк с истинными ответами и без полной матрицы ошибок.

Снимки pipeline.joblib сохраняются с joblib compress=3. MODEL_SNAPSHOT_MAX_BYTES=16777216
ограничивает один сжатый снимок, MODEL_SNAPSHOT_TOTAL_BYTES=67108864 — их сумму за запуск,
с резервированием одновременно работающих jobs. Нулевое значение отключает снимки.
job.model_snapshot сообщает saved/size_limit/disabled/unavailable; превышение не отменяет fit/predict.
Частичный файл удаляется. Закреплённые code/params/protocol и предсказания остаются для реконструкции.

WORKER_LOG_MAX_BYTES=1048576 ограничивает каждый stdout/stderr именно в UTF-8 байтах, включая
маркер усечения; job.log_truncated сообщает лимит, поток продолжает дренироваться. Веса, CSV,
API snapshots и журнал — разные категории. scripts/analyze_run.py показывает total_bytes,
by_category и десять крупнейших файлов, читая только metadata/размеры без загрузки pickle/CSV.
Это ограничения отдельных категорий, а не обещание полного фиксированного размера каталога runs.

## Согласованность v0.5.2

graph.plateau_assessed и GPlateau включают peer_trials_since_progress; причина продолжения показывает, когда одному peer ещё не хватает разных завершённых проверок. search.admission включает phase_seconds_remaining. search.phase_changed.reason может быть next_research_batch_would_cross_phase_boundary: допуск очередной партии учитывает прогноз LLM и fit.

Компактная память находится в контексте profile_summary.measured_search: base_experiment_id, entries с измеренными score и изменениями относительно базы, omitted_count. Старые запросы LLM имеют приоритет над свежими локальными соседями. Текстовые предположения остаются гипотезами; authoritative метрика — только оценщик.
