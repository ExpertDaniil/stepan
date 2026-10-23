# Исследовательский цикл: реализация и границы проверки, 2026-10-23

Исходная удалённая ветка `feat/open-research-loop` проверена через GitHub API:
`b79c7e055d9bcbc417eb7c01c4b9125b3d5bb7f4`, дерево `1f438203271fff70dff997fb2ecd8b86fe324da2`.
Из-за отсутствия Git credentials checkout получен через GitHub-коннектор: сверены все 334 blob,
исходное дерево и сам объект commit. До правок локальное дерево было чистым.
`main` не меняется; разрешён только обычный fast-forward указанной ветки.
Историческое указание AGENTS о main уступает прямому поручению пользователя работать в этой ветке.

## Диагноз и доказательства

| Утверждение | Статус и источник | Изменение / проверяемое свойство |
|---|---|---|
| Разработка и платформа имеют одинаковые бюджеты | Опровергнуто QA, вопрос 12 | platform не получает числовые квоты разработки по умолчанию |
| Ресурсы общие на две задачи | Опровергнуто QA 32, 35 | На задачу: 8 CPU, 32 ГБ, 30 ГБ диска; до 3 часов по гиду |
| Ранний лимит LLM | Факт исходного config/Budget/RefinementPolicy | 32 calls, 262144 output и доля фазы заменены явными профилями |
| В основном прогоне токены были главным ограничителем | Опровергнуто сохранённым `run-20261023T092215Z-f5e641a1.zip` | 27 вызовов: 2 bootstrap + 25 research; 78885 reported completion при лимите 1048576 |
| Резерв terminal/repair отсутствовал | Опровергнуто исходным b79c7e0 | Существующие атомарные резервы сохранены; расширен путь terminal при усечении |
| Семь tool-ходов не оставили вызова для JSON | Наблюдение исторического запуска, не доказательство текущей ошибки каждого диалога | Новый лимит задаётся как tool turns + final + repair, по умолчанию 8+2 |
| JSON потерял решение | Наблюдение лога: invalid JSON на восьмой попытке | Ограниченный addressed repair и сохранение отвергнутого/усечённого content |
| Optuna отсутствует/не работает | Опровергнуто кодом и логом | ask/tell и 12 trial уже были; оптимизатор не переписан |
| Optuna улучшила RMSE с 44.9226 до 43.1259, fallback до 42.3074 | Числа исторического прогона; не результат этого изменения | Нельзя переносить их на новый validation protocol или приватные задачи |
| Результат серии действительно получала следующая LLM | Опровергнуто для завершённой серии исторического прогона | Последний запрос 09:29:19Z, серия закончилась 09:32:51Z; теперь есть dispatch-time feedback и отдельное событие доставки |
| Простой около 103 секунд | Сообщённое наблюдение; сам общий барьер подтверждён исходным controller | При одном train worker свободный peer может снова вызвать LLM, пока другой ждёт |
| Thinking был выключен | Опровергнуто usage последнего ответа | 4962 reasoning / 5288 completion; это не измерение качества рассуждений |
| CatBoost всегда медианно импутирует и ordinal-encode категории | Опровергнуто текущим owned pipeline | Native numeric NaN и native categorical policy остаются источником фактов |
| reasoning_content всегда надо удалять или всегда пересылать | Неподтверждённые универсальные утверждения | Совместимый default и явный opt-in протокол с ограничением истории |
| Один target / один столбец | Факт исходных TaskSpec, worker, evaluator, publisher; QA 16,17,55 допускает другое | Добавлены вероятности и несколько числовых целей сквозным контрактом |
| Все типы ML поддержаны | Опровергнуто | Ranking/unsupervised/мультитаргетная классификация отклоняются явно до fit |
| Один holdout гарантирует честное обобщение после многих попыток | Неверная гарантия; риск адаптивного переобучения | Отложенная часть validation используется один раз при финальном выборе |
| Ошибки ZIP/агрегированный public score доказывают внутренний дефект или прирост | Не следует из доступных данных; проблема платформы сообщена пользователем | Packager сохранён, лидерборд не вызывается |
| KeyboardInterrupt вызван внутренней политикой агента | Не установлено | Архив содержит прерывание; причина извне не приписывается scheduler |
| Больше токенов или новый prompt гарантируют более умную 27B | Гипотеза без измерений | Контроллер проверяет исполнение, ссылки, лимиты и формат, не истинность интерпретаций |

Гид, QA и baseline прочитаны. В baseline используется argv task.md и OPENAI_*.
QA 54 описывает MCP неоднозначно; не добавлялся новый процессный протокол в обход фактического
CLI baseline. Обе копии requirements побайтово совпадают с приложенным официальным файлом:
SHA256 `9173b0388532fbd2a709e4f4b3029e3b9ccab685fce43a85261ffa1ba95efbe6`.

## Бюджеты и завершение диалога

`Config.from_env()`/CLI без локальной настройки выбирает **platform**. `--profile development`
или `AUTOML_PROFILE=development` задаёт прежние 32 calls / 262144 output. `.env.example`
явно выбирает development. Прямой `Config()` оставляет development defaults для существующих fixtures.

В platform обе суммарные числовые квоты равны `None`. Это отсутствие квоты, а не большая
константа. JSON snapshot передаёт `null`; Cxx и GRefinement-схемы обновлены. Явно заданные
`LLM_MAX_CALLS`/`LLM_MAX_TOTAL_OUTPUT_TOKENS` остаются действующими пользовательскими
ограничениями в любом профиле; `0`, `none`, `unlimited` отключают такую квоту.
Это видно в `llm.configuration`. Старый `.env` с 32 не отменяется молча.

Platform по умолчанию отключает необратимый переход 85/15 в refine. Исследователь вправе
вернуться к данным, признакам и другой модели. Опциональный `GRAPH_REFINEMENT_ENABLED=1`
сохраняет прежнюю стратегию, но не создаёт числовые квоты из отсутствующих лимитов.
Ни MAX_RESEARCH_EXPERIMENTS, ни RESEARCH_SERIES_SIZE не становятся лимитом всего поиска.

Budget под RLock резервирует будущие final/repair API attempts и output, отдельно от inflight
запросов. Другой диалог не может потратить эти средства. Неиспользованный резерв возвращается
в `finally`; timeout без usage консервативно списывает запрос, не выдумывая reported usage.
При усечении увеличение max_tokens допускается только за свободные средства; иначе остаётся
компактная terminal-попытка в уже защищённом лимите. Диалог конечен по attempts, calls при
наличии квоты, output одного ответа, operation timeout и времени задачи.

Отдельно учитываются ожидание semaphore, HTTP timeout, operation deadline, опциональная фаза
и fit/refit/predict/проверка/запись. Финальный ответ не гарантирован: медленный сервер или
невалидный JSON могут исчерпать даже защищённые попытки/время. Это фиксируется как ошибка,
а не выдаётся за исследовательское решение.

## Обратная связь, планирование и память

Dispatcher после train/evaluate/Optuna tell публикует FeedbackMailbox отдельно для каждой
активной ветки, соблюдая ancestor cuts. API threads читают только immutable JSON и защищённую
копию mailbox, без SQL. Каждый фактический research request получает актуальную доступную
сводку результатов/серий. `research.feedback_delivered` связывает её с call_no, context_id,
artifact, experiment IDs и series IDs. Доставленная сводка входит в evidence provenance решения.
Это свидетельство передачи, не доказательство внимания модели к каждому полю.

По умолчанию один train worker. Rolling dispatcher запускает следующий диалог освободившегося
peer до завершения медленного. Уже разрешённая Optuna-серия продолжает Runner/Evaluator/Store
во время ожидания API, с прежними max_trials/search_space/deadline. Никакие дополнительные
гипотезы или trial вне авторизации не создаются под видом продолжения серии.
С двумя train workers остаётся прежняя batch-реализация; её API-барьер этим изменением не снят.

Точный повтор действия без новых измерений дважды откладывает peer до новой версии feedback.
Это явная эвристика предотвращения повторов, не семантическая проверка научной новизны.
Успешная ветка и совместные изменения доступны. Ошибки и stop имеют причины; fallback
показывает причину входа и длительность. Опоздавший/отменённый HTTP-ответ учитывается и
архивируется, но не становится новым экспериментом. После исчерпания bounded retries
transport error включает cooldown: до 60 секунд при временной ошибке, до конца задачи при
невосстановимой ошибке endpoint/auth. В это время доступен fallback; после временного
cooldown LLM снова доступна. Это реакция на ошибку транспорта, не квота разработки.
Жёсткое прерывание зависшего HTTP thread
не реализуется средствами Python: окончательный срок зависит также от SDK/транспорта.

Исследовательская память различает proposed/running/measured/failed/deferred. Статус measured
не означает подтверждённую причинную гипотезу. Альтернативы сохраняют причину отсрочки и ссылки;
неудачное совместное преобразование не опровергает каждый компонент. Полные решения остаются
артефактами, компактная проекция содержит последние 6 решений и до 24 идей с числами пропусков.
Старая полная память не обязана целиком входить в каждый prompt.

Optuna scope уже включает код, fixed parameters, task/dataset/validation/seed и пространство;
для custom — также feature steps. Совместимая история импортируется только после доверенной
оценки. Это сохранено; добавлен regression на различающиеся scopes. Kaggle-корпус также сохранён:
карточки/тексты/4 полных owned примера доставляются как текстовые ресурсы; ссылки не выдаются
за прочитанные ноутбуки. Отбор по типу задачи и ограничения применения остаются действующими.

## Контекст и reasoning

Adaptive больше не имеет бесконтрольной истории. `LLM_REQUEST_MAX_BYTES=196608` ограничивает
всё тело запроса; дополнительно оставляется место ответу в `LLM_CONTEXT_WINDOW_TOKENS=262144`.
Подсчёт байтов — защитная граница, не точный tokenizer и не подтверждение окна прокси.
Schema/tools/history включены в проверку. Статические core/base/code/контракт защищены; старые
детали сокращаются со ссылками и явными counts/events. Несмещаемый контекст, который не
помещается, даёт CONTEXT_TOO_LARGE, а не скрыто урезанное решение.

`LLM_REASONING_PROTOCOL=compatible` не отправляет неподтверждённых Qwen-полей. При этом видимый
assistant content, нормализованные tool_calls и tool_call_id сохраняются. Для заранее
проверенного совместимого сервера можно явно выбрать `qwen_chat_template` и effort
low/medium/xhigh. В этом режиме SDK extra_body переносит chat_template_kwargs enable_thinking,
preserve_thinking и необязательный reasoning_effort, а provider reasoning_content возвращается
в assistant message (`LLM_REASONING_FIELD`: reasoning_content или reasoning по контракту сервера).
История reasoning ограничена 32768 UTF-8 bytes, старые поля удаляются
целиком с событием. Она не вставляется как user-текст и не переносится между исследователями.

Проверенные первичные источники:
- [Официальная карточка Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B): thinking default,
  xhigh/medium/low и preserve_thinking, формат self-hosted Chat Completions.
- [openai-python v1.68.2 create/extra_body](https://github.com/openai/openai-python/blob/v1.68.2/src/openai/resources/chat/completions/completions.py).
- [SDK BaseModel extra=allow](https://github.com/openai/openai-python/blob/v1.68.2/src/openai/_models.py)
  и [сохранение неизвестных полей messages](https://github.com/openai/openai-python/blob/v1.68.2/src/openai/_utils/_transform.py).

Контракт SDK подтверждён чтением исходников; применение полей предоставленным прокси не
проверялось. HTTP 200 не считается таким подтверждением. Default остаётся совместимым.
Влияние усилия reasoning и replay на качество/скорость нужно сравнивать отдельными факторами.

## Предсказания, подтверждение и Finalizer

TaskDraft → TaskSpec → worker → Evaluator → Finalizer → CSV теперь поддерживают:
- метки одного классификационного target;
- вероятности log_loss и бинарного roc_auc с явно заданным positive_label;
- вероятности в ответе независимо от того, нужны ли метрике метки;
- все классы в объявленном class_labels порядке либо один бинарный positive_label;
- несколько числовых целей, по одному estimator на target, последовательно в пределах lease;
- точную произвольную перестановку колонок sample_submission при test row order.

Мультитаргетная RMSE: sqrt(mean squared error по строкам и целям); MAE: среднее абсолютное
отклонение; R2: uniform target averaging. Иные требуемые агрегации не подменяются этими.
Число/порядок targets и классов входит в frozen protocol. Все targets исключаются из features;
secondary targets также защищены от analysis feature queries. Нет flatten многомерного ответа.
predict_proba требует classes_; пропущенные в fit известные классы получают нулевую вероятность.
Проверяются размерность, конечность, [0,1], сумма вероятностей, классы, id и порядок колонок.

Ranking, unsupervised, multi-target classification, quantiles/intervals и неизвестные метрики
ещё не имеют исполняемого адаптера. Для них есть явный unsupported draft до обучения;
расширение enum не выдаётся за поддержку. Ранний Dummy тоже обязан пройти конкретный формат.

Evaluator замораживает selection/confirmation внутри общего validation. Обычно 25% validation
удерживаются от adaptive score и diagnostics; train-only analysis не получает их меток.
Группы/периоды не дробятся; маленький/неразделимый holdout получает явную причину пропуска.
Все trials предсказывают полный validation, поэтому финальное сравнение использует уже
сохранённые predictions без повторного обучения. После закрытия API dispatcher сравнивает
двух финалистов один раз. Это дополнительная проверка обобщения, не доказательство
статистической значимости; многократная настройка всё ещё адаптивна на selection subset.

Finalizer читает сохранённый plan/code. Runner перед первым fit фиксирует execution.json,
проверяет fingerprint, SHA кода и используемого runtime; refit проверяет неизменность снимка.
Не генерируется похожий pipeline. Изменение кода после reserve/между оценкой и refit отклоняется.
Ошибка C19 до spawn теперь освобождает lease. Поздний неудачный кандидат сохраняет предыдущий
CSV. Final confirmation может разрешить замену primary winner только через текущий доверенный
одноразовый report, с проверкой версии Publisher и проверкой готового нового CSV.

Родительские BLAS/OpenMP/NumExpr ограничены двумя потоками; до двух train jobs по 3 потока
сохраняют общий план 8 CPU. Рабочие группы, RSS-watchdog, cleanup перед release и лимиты
снимков/логов сохранены. Это не отдельный cgroup на каждый job: hard resource ceiling у платформы.

## Что проверено в этой сессии

Разрешены чтение, сравнение и статические проверки. Не запускались tests, обучение, agent
smoke/full run, Optuna studies, платные/live LLM-запросы или pip. Не создавался submission ZIP:
проверен сам путь упаковки, его allowlist и реальная доступность всех ресурсов/новых модулей.
Результаты статического аудита фиксируются в `VALIDATION-MASTER-UPDATE.md`.

Написаны сценарии: concurrent reserves; final after tools; late cancellation; feedback in actual
request; free peer rolling dispatch; study scope/dedup; class/column alignment; multiple target
pipeline; frozen refit и lease cleanup; withheld confirmation и one-shot publication.
Они **не выполнены**. Синтаксис и исходники не доказывают успешный fit, работу прокси или gain.

## Локальная проверка (в существующей подготовленной среде)

```bash
python3 -m unittest tests.test_platform_research tests.test_prediction_contracts tests.test_confirmation tests.test_dialogue_progress tests.test_optuna_search tests.test_research_memory tests.test_submission -v
python3 scripts/check.py
```

Mock smoke включает настоящее обучение синтетических данных, но не вызывает провайдера:

```bash
python3 scripts/make_demo.py demo_tasks/master-smoke --kind regression --rows 360
TIME_BUDGET_SECONDS=120 CATALOG_WARMUP_EXPERIMENTS=1 python3 solution.py demo_tasks/master-smoke/task.md --llm-mode mock --profile development
```

Полный live run (команда сознательно перекрывает старые квоты из .env):

```bash
RESEARCH_MODE=adaptive LLM_MAX_CALLS=0 LLM_MAX_TOTAL_OUTPUT_TOKENS=0 GRAPH_REFINEMENT_ENABLED=0 python3 solution.py "data_for_participants/Регрессия/task.md" --env-file .env --profile platform
RESEARCH_MODE=adaptive LLM_MAX_CALLS=0 LLM_MAX_TOTAL_OUTPUT_TOKENS=0 GRAPH_REFINEMENT_ENABLED=0 python3 solution.py "data_for_participants/Многоклассовая классификация/task.md" --env-file .env --profile platform
python3 scripts/package_solution.py
```

Чтение результатов (подставить реальный RUN_DIR из run.finished):

```bash
RUN_DIR="$(ls -td runs/run-* | head -n 1)"
python3 scripts/analyze_run.py "$RUN_DIR" --json
python3 scripts/show_run.py "$RUN_DIR" --event research.
cat "$RUN_DIR/summary.md"
cat "$RUN_DIR/validation/final-confirmation.json"
rg 'research.feedback_delivered|research.peer_resumed_while_waiting|search.fallback|search.finished|llm.late_result_discarded|validation.finalists_confirmed|submission.published' "$RUN_DIR/events.jsonl"
```

Если confirmation пропущена, отсутствие final-confirmation.json штатно; причина в событиях.
Проверять больше завершённых содержательных решений и качество на нескольких задачах при
сопоставимом wall time. Новый split изменяет числовой смысл research score: сравнивать итоговый
test score или явно одинаковые протоколы. Токены, число признаков и названия направлений не KPI.
