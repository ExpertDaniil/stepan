# Research Graph v1 · реализация v0.4.3

Основа: текущая v0.3.1, commit 3b6d7d84879fffbe76d111d05a9d58816b009e77. Фикс завершения worker
и три его теста сохранены. Интерфейс платформы — прежний solution.py; C00–C24 не меняются.
G00–G14/I00–I03 — работающие локальные операции, их схемы и request/success/error fixtures находятся в contracts/.
Точная карта модулей и форматов также дана в graph-io.json. Runtime использует stdlib и уже разрешённые зависимости.

Изменения v0.4.2, регрессионные проверки, новые необязательные поля контекста и продолжительный
профиль поиска описаны в [RESEARCH-REVIEW.md](RESEARCH-REVIEW.md). Читаемая плоская поставка v0.4.1 сохранена.

## Как идёт работа

1. После существующего bootstrap фиксируются TaskSpec, validation protocol, каталог, окружение и собственный runtime.
   G00 создаёт корневую ветвь и отдельные головы равноправных peers в той же state.sqlite. G12 проверяет её целостность.
2. Ранний Dummy выдаётся одному peer как доверенный baseline work order. Он проходит C11–C21 и публикует CSV.
   Остальные peers получают тот же результат через свои duplicate bindings без повторного fit.
3. Каталожные warmups и последующие исследования привязаны к собственным heads. Роли exploit/explore
   выдаются каждому peer в отношении 3:1, постоянного лидера-инженера нет. Dispatcher связывает каждый Future с ticket задания.
4. G03 собирает свежий контекст: задача/метрика/output, work order, собственный code/config, ограничения,
   до восьми относящихся к линии фактов и явно назначенные donors. Другие ветви не подмешиваются как global recent.
5. C09/C10 по умолчанию возвращают ResearchChoice-v1. C11 создаёт точный owned factory. I00–I03 проверяют
   исполняемое содержимое, компонентный closure, identity и родителей. В T4 C12 reserve и graph binding коммитятся вместе.
6. Отдельный процесс обучается; C15 принимает текущий attempt/fence; C16 вычисляет метрику по сохранённым valid indices.
   T5 одновременно выполняет C17, CAS головы конкретного peer, событие, outbox и idempotency. Ошибки не получают score=0.
7. Для нового переносимого компонента ищется контролируемая пара: тот же recipe/code/seed/protocol,
   изменён только заявленный feature. G06 сохраняет фактический delta и provisional идею.
8. G14 назначает один pilot другому peer от его собственной базы. После положительного source и recipient
   эффектов идея становится confirmed новой неизменяемой записью. min_paired_seeds=1 — инженерный gate, не доказательство значимости.
9. G02 сохраняет checkpoint до идеи, запечатывает родителя и атомарно создаёт accept/reject.
   GBranchSeed явно задаёт каждому peer собственную измеренную базу, совместимую с новой линией.
   Accept требует feature в новых планах; reject наследует запрет.
10. G07 проверяет плато по distinct full solutions и времени с существенного улучшения. G14/G08 возвращают
    поиск к paused альтернативе. Также резервируется хотя бы одна проверка альтернативы перед исчерпанием trial cap.
    При достижении live-limit законченную линию можно архивировать, сохранив весь её результат.
11. После C22/cleanup G11 сравнивает все valid comparable full experiments: предков, старые/закрытые ветви,
    late results и текущие листья. Лучший evaluated и реально published champion хранятся отдельно.

## Две структуры и идентичности

Верхнее дерево — решения команды принять/отклонить компонент. У узла ровно два ребёнка, повторный fork sealed
узла запрещён. Нижний DAG — версии конвейеров с primary/idea_donor/ensemble_member provenance; количество
его продолжений не равно двум. Ensemble compiler в первой реализации не включён: соответствующее действие явно отклоняется.

Solution hash включает фактический code/config/components/interface/environment, но не seed/fidelity/protocol.
Fingerprint C12 сохранён: dataset_id, task_spec_id, validation_id, recipe_id, model, feature_steps, seed,
fidelity, code_sha256. Разный seed создаёт другой experiment той же solution; повторный exact fingerprint — binding,
а не ещё одно обучение. Альтернативное происхождение существующей solution сохраняется в proposal, не создавая цикл DAG.

store_revision — прежняя selections.revision, только новый global incumbent. graph_revision — отдельная ревизия журнала.
branch_revision меняется при pause/resume/seal/archive; head_revision относится к одному peer внутри одной ветви.
Прогресс соседа не инвалидирует собственную работу. Технические attempts/fencing не смешиваются с этими ревизиями.

## Контекст и запоздалые результаты

Задание содержит branch/head revisions, base code/IDs, checkpoint, protocol SHA, TTL и allowlist donors.
Его identity задаёт controller; researcher_id из ответа сам по себе недостаточен. C08 хранит совместимый
messages/metadata с graph_binding; user.content проверяется как GContextUser. В profile_summary находятся data,
доступные recipes, task_text и context_budget. Schema ResearchChoice-v1 добавлена к старому ResearchProposal-v1.

Сначала сокращается избыточный профиль. Измеренная база (`base_result`), явные donors и ближайшие
факты защищены; затем сокращается оставшаяся история. Собственные наблюдения предков ограничены
captured_event_seq соответствующего fork checkpoint. Чужая линия не становится global recent.
Условия, constraints и base code не обрезаются. research_strategy явно передаёт exploit/explore.
Gateway перед каждым реальным API-вызовом проверяет полный сериализованный request: schemas, tools и repair rounds
также занимают бюджет. Без tokenizer byte-based estimate не выдаётся за точное число токенов или известное окно провайдера.
При превышении mandatory части возвращается явная ошибка; ранее опубликованный CSV остаётся.

Stale/expired issued proposal не резервирует experiment. Уже submitted работа возвращается в origin binding.
После смены branch/head epoch её результат архивируется и не двигает новую голову; актуальный full результат
может стать global best. Устаревший execution fence отвергается до оценки. TTL предложения не отменяет уже законно
запущенную тренировку. Бюджет, процессы и publication никогда не откатываются вместе с checkpoint.

## Компоненты и запреты

Сейчас зарегистрированы model.family, feature.class_weight, feature.target_transform и feature.numeric_interactions.
Последний доступен для недамми-каталожных моделей; квадраты/пары выбираются только по fit-столбцам и ограничены 36 новыми признаками.
class_weight=balanced применяется к поддерживающему recipe; target_transform=shifted_log1p — только регрессия,
fit labels и обратное преобразование предсказаний. Их наличие проверяется по фактическому детерминированному конструктору.

Каталожный режим — catalog_strict. Явный RESEARCH_MODE=code допускает declared_only на unrestricted линии.
Hard required/forbidden ветвь принимает только catalog_verified. Hash/AST не доказывают отсутствие произвольного
переписанного аналога идеи. Неизвестные компоненты не регистрируются по словам модели. Новый capability добавляется
в owned registry/recipe/implementation и сопровождается тестами actual transform + controlled transfer.

## Authority, файлы, транзакции

| Хранилище | Что содержит | Писатель |
|---|---|---|
| state.sqlite: native tables | Эксперименты, C16 evaluations, selections, publications | Store dispatcher |
| state.sqlite: g_* | Branches/heads, solutions/edges, orders/bindings, trials/ideas, decisions/events, idempotency/outbox | Graph dispatcher |
| Native named paths | code, validation, contexts, job/results, candidates, telemetry | Существующие модули; refs pin-ятся dispatcher |
| objects/<sha256>/content | Immutable manifests, checkpoints, capsules, proposals, trials, exports | G09 |
| graph/owned-runtime.zip | Только собственный код/ресурсы и dependency lock; без data/packages/weights | Bootstrap |
| exports/<scope>-<cut>/ | Читаемые derived graph/ranking/events | G11 |

Графовая миграция agent/graph/schema.sql применяется один раз после native schema в той же DB.
Связь g_heads с experiment/solution защищена составным FK; DAG cycle и immutable history — SQL triggers.
Store.transaction использует SAVEPOINT при вложении. reserve_tx/commit_evaluation_tx не выполняют внутренний commit.
T4/T5 откатываются целиком. Ошибка публикации не удаляет старый verified CSV.

Файловая система и SQLite не объявлены общей атомарной транзакцией. G09: fsync staging → durable intent →
rename на том же FS → fsync каталогов → SQL READY. Сбой между этапами закрывает G12. READY bytes не перезаписываются.
Legacy external ArtifactRef.relative_path может быть виртуальным alias: resolve идёт по g_artifact_locations и task_root,
проверяет границы/size/SHA. artifacts-index.json — совместимая проекция; pins/locations восстанавливаются из DB.

Durable key — (run_id,operation,idempotency_key) с canonical payload SHA. Replay возвращает прежнее data с новым
request_id, не создаёт новый experiment. Планировщик затем читает текущее состояние experiment: исторический queued
в idempotent reply не означает повторный запуск уже evaluated модели. C14/spawn/timeout/invalid и C22 закрывают bindings,
чтобы peer не оставался submitted навсегда. Ресурсы освобождаются после подтверждённой остановки группы.

G12 проверяет FK/bytes, завершает intents, отзывает старые issued contexts, закрывает не начатые задания и может
завершить T5 для уже сохранённой доверенной C16 evaluation с matching successful attempt. Неизвестный живой job,
corrupt pin или trained без trusted evaluation блокируют поиск. CLI по-прежнему начинает новый run_id;
автоматического продолжения старого контейнера и восстановления потраченного бюджета нет.

## Маршруты

| Операции | Handler | JSON request → response |
|---|---|---|
| G00/G01/G02/G04/G05/G08 | GraphService | Gxx_Request → Gxx_Response |
| G03 | WorkService + ContextBuilder | G03_Request → G03_Response; вложенный C09_Request |
| G06/G07/G14 | SearchPolicy | Gxx_Request → Gxx_Response |
| G09/G10 | GraphArtifacts | Gxx_Request → Gxx_Response; ArtifactRef |
| G11/G13 | GraphExport | Gxx_Request → Gxx_Response |
| G12 | GraphRecovery | G12_Request → G12_Response |
| I00 | Constructor.verify | I00_Request → I00_Response |
| I01 | constraints.check | I01_Request → I01_Response |
| I02 | GraphService.resolve_lineage | I02_Request → I02_Response |
| I03 | identity.solution | I03_Request → I03_Response |

Все Request/Response/Error — в contracts/contracts.schema.json. Исторические C fixtures не переписаны.
GEnvironmentManifest дополнен optional owned_runtime_artifact_id/owned_runtime_sha256; runtime записывает оба
для проверки архива. Это уточнение нового G-слоя, не изменение C wire. Все R/G records остаются JSON, raw weights/CSV/ZIP — bytes по ArtifactRef.

## Стартовые настройки

| Env | Default |
|---|---|
| GRAPH_ENABLED | 1 |
| GRAPH_PEERS / GRAPH_TRAIN_WORKERS | 2 / 1; каждый ≤ MAX_WORKERS |
| GRAPH_MAX_LIVE_BRANCHES / GRAPH_MAX_TOTAL_BRANCHES | 3 / 15 |
| GRAPH_PLATEAU_WINDOW / GRAPH_PLATEAU_SECONDS | 4 / 300 |
| GRAPH_MIN_EFFECT | 0.001 |
| GRAPH_CONTEXT_BYTES | 48000 для полного API request |
| GRAPH_MAX_ARTIFACT_BYTES / GRAPH_MIN_FREE_DISK_BYTES | 8000000000 / 2000000000 для новых graph blobs |

Рецептный/LLM бюджет v0.3 сохранён. Параметры ветвления/плато — инженерные defaults, не математически оптимальные.
Поиск ограничен общими wall/API/token/training caps. Screening не выполняется: текущий worker реально реализует full.
Тюнинг параметров не превращается автоматически в fork; перенос требует контролируемой пары и recipient pilot.

## Как смотреть и восстановить

```bash
python3 scripts/show_graph.py runs/run-ID
python3 scripts/show_graph.py runs/run-ID --json
python3 scripts/reconstruct_solution.py runs/run-ID exp-ID rebuilt --task-root /path/to/original/task-root
python3 rebuilt/run_refit.py
```

Первая реконструкция только проверяет hashes и создаёт candidate.py, manifest/plan/environment/protocol,
inputs.json, refit-job.json и owned runtime. run_refit.py запускает обучение отдельно, проверяя package versions
и input hashes; ничего не устанавливает и не обращается к LLM. Результат — rebuilt/candidate.csv, не замена
опубликованного ответа. Для восстановления нужны исходные train/test; абсолютные старые пути можно заменить
через task-root. Сериализованные модели при просмотре/реконструкции не загружаются.

## Проверки

`python3 scripts/check.py` проверяет весь проект. Новые тесты в test_graph*.py покрывают отдельные heads,
reverse completion, stale proposals/fences, late result, exact dedup/seed identity, controlled source/pilot,
binary/nested forks, bans, backtrack, full-request size, failure cleanup, T4/T5 crash rollback, durable replay,
intent recovery, SQL integrity, все ветви финального ranking и реальное восстановление refit из собственного runtime.

State-machine тесты явно инъецируют синтетические scores для проверки переходов. Сквозные тесты обучают модели
и вычисляют настоящие метрики на синтетических CSV, имитируя только LLM-ответы. Внешний live API, CatBoost
и официальный контейнер проверяются отдельно. Результат общего прогона и точные версии — VALIDATION-GRAPH.md.


## Старт ветвей и плато v0.4.3

Причинный fork_checkpoint_id остаётся неизменным. При G02 создаются GBranchSeed для каждого peer
и ребёнка: branch_id, parent_branch_id, fork_checkpoint_id, selection_cut, head и selection_reason.
Из собственных валидных full результатов родительской линии выбирается лучший, совместимый
с constraints ребёнка. Если для accept такого ещё нет, берётся собственная база без новой фичи
и используется прежний механизм controlled transfer. Чужой глобальный лидер не становится базой.
G03/visible_history включают эти явные ссылки; обычный ancestor cut остаётся прежним. G11 добавляет
optional branch_seeds, а typed events полностью описывают происхождение каждого старта.

Плато: четыре новые distinct solution подряд, суммарный gain относительно фиксированного anchor
не превышает epsilon, и прошло GRAPH_PLATEAU_SECONDS после существенного gain/активации.
Сумма маленьких улучшений выше epsilon сбрасывает счётчик. Seed, exact cache и другой seed
той же solution не считаются новыми шагами. Ошибка обучения расходует бюджет, но не является оценкой плато.
Новые optional GPlateau поля: consecutive_no_gain, required_no_gain, cumulative_gain_since_reset.
Epsilon задают GRAPH_PLATEAU_ABSOLUTE_EPSILON / GRAPH_PLATEAU_RELATIVE_EPSILON; он не запрещает
обновление head/итогового CSV при меньшем улучшении. Бюджетный переход остаётся отдельной причиной;
обычное переключение по доле бюджета ждёт четыре испытания текущей линии, аварийный резерв
альтернативы действует перед жёстким лимитом.

GRAPH_EXPLOIT_STEPS=3 задаёт три own-base доработки на одну exploration на каждого peer;
счётчик зависит от его завершённых поручений, а не количества проходов цикла.
