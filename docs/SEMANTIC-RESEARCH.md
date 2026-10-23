# Компактный контекст, смысл данных и проверяемое исполнение

Изменения подготовлены 24 октября 2026 для `feat/final-data-research` от
`75087993062e30ea7ce48242bd428c0d35b323f5`. Это инженерное изменение контрактов,
памяти и вычислительных инструментов. Его влияние на качество ML ещё не измерено.

## Проверенный диагноз

| Проблема или утверждение | Доказательство | Статус | Изменение / проверяемое свойство |
|---|---|---|---|
| Adaptive обходит графовый byte cap | Старые ContextBuilder.build_graph и Gateway._request применяли общий лимит вместо graph limit | Подтверждено исходником исходного HEAD | Единый effective_limit для сборки и каждой отправки, включая final/repair |
| Ограничение числом решений недостаточно | compact_memory сохранял до 6 больших решений с trials, параметрами и интерпретациями | Подтверждено исходником и архивом | Ограничен весь JSON-проект памяти; подробности читаются по scope-bound reference |
| Граф полностью не работает | lineage_cuts/visible_history ограничивают ветви и моменты fork | Опровергнуто | Существующая область видимости сохранена и применяется до создания документов |
| Отсутствует любая полезная память | research_memory, feature_memory, comparison_evidence уже существовали | Опровергнуто | Расширены существующие immutable решения, без новой БД/второго ExperimentStore |
| Замысел взаимодействия не проверялся | Архивная гипотеза называла площадь × градусо-дни, helper выбирал другие поля по позиции | Подтверждено архивом и исходником | Именованные программы, явные helper inputs, детерминированный fallback и execution report |
| Логи доказывают необходимость конкретного нового признака | Только два сохранённых анализа и концентрация ошибок | Гипотеза, не доказательство полезности | Инструменты позволяют проверить вопрос; решение о бюджете остаётся у исследователя |
| Упирание в output cap объясняет плохой поиск | В просмотренном архивном срезе нет output_truncated | Не подтверждено этим срезом | Output quota, reasoning и input bytes остаются разными величинами |

### Исторический архив, перепроверенный в этой сессии

Источник: `run-20261023T201249Z-04cac26e.zip`, **старая реализация**, не результат
этого изменения. Первое событие 20:12:49.412 UTC, последнее 20:41:12.646 UTC:
28 минут 23.234 секунды; последнее событие — ожидание LLM, не завершение запуска.
В срезе 57 evaluated событий, лучший RMSE `40.8831247513459`,
`exp-e53bc219845a`. Сохранены только schema и group-анализ по facility_type.

У последнего запроса content messages содержит 143493 символа, сериализованная
research_memory — 44258 символов; для запросов 0034–0038 память занимала
47421–47803 символа. Это **символы**, не токены. Повторная JSON-сериализация
последнего сохранённого request занимает 161004 UTF-8 bytes; это не измерение
исторического HTTP wire body. Максимум content среди запросов среза — 154669 символов.

SHA256 train из официального архива совпал с train fingerprint старого run.
Сохранённые predictions + protocol + labels воспроизвели RMSE `40.8831247513459`;
на худшие ceil(1%) из 18308 validation-строк приходится `54.82098753863645%` SSE.
Selection и all-validation в этом архиве совпадают. Этот расчёт не содержит fit,
не доказывает наличие ошибочных выбросов и не обещает достижимый выигрыш.

В decision-3e12e8291c114485 заявлено взаимодействие floor_area с degree-days;
первые восемь числовых inputs старого helper не содержат градусо-дней. Реальные
категориальные cross/frequency признаки могли быть полезны; это не основание
запретить статистические преобразования. Других причин завершения фрагмента
или результатов новой версии из этого архива не выводилось.

## Владение бюджетом

`agent/context_budget.py` — общий владелец вычисления эффективного ограничения.

| Настройка | Default | Значение |
|---|---:|---|
| LLM_REQUEST_MAX_BYTES | 196608 | Верхний предел полного JSON body |
| GRAPH_CONTEXT_BYTES | 98304 | Верхний предел полного graph request во всех режимах, включая adaptive |
| issued policy context.max_api_request_bytes | seed 98304 | Зафиксированный лимит конкретного поручения |
| LLM_CONTEXT_WINDOW_TOKENS | 262144 | Клиентское предположение о provider window; не найденная автоматически спецификация сервера |
| LLM_MAX_OUTPUT_TOKENS / LIMIT | 16384 / 32768 | Резерв генерации; не бюджет памяти и не число API calls |

Config.from_env читает окружение с документированными defaults; GraphService.default_policy
накладывает config на исходную policy. Для отправки берётся минимум client bytes,
graph config и issued graph policy. Legacy дополнительно сохраняет MAX_CONTEXT_TOKENS*3
как **приближённую byte-границу**, не tokenizer. При неизвестном tokenizer provider
window проверяется отдельным консервативным proxy: один input byte на предполагаемый
токен после вычета output и 1024 токенов framing. Это не точный расчёт токенов и
не гарантия tokenizer proxy; provider остаётся окончательным арбитром.

Сборщик оставляет min(24576, limit/3) bytes для schema/tools/feedback/продолжения.
Gateway после всех дополнений повторно измеряет сериализованный JSON. HTTP request
hook дополнительно проверяет **фактические bytes outgoing.content до сети**. Размер
sections записывается отдельно; headers/ключи не логируются. Не поместившийся
необходимый контракт, база или свежий результат дают CONTEXT_TOO_LARGE с размерами
и пределами, а не молчаливое удаление ограничений. Слишком маленький прежний env cap
сохраняет своё значение и может потребовать явного увеличения пользователем.

## Память и чтение

Dispatcher сначала применяет действующие lineage/fork/donor правила и сохраняет
`research_context_details`. Scope содержит run, task ID и hash текста условия,
dataset ID и SHA256 обеих таблиц, validation ID/full protocol hash, work order,
branch cuts и donor allowlist. SQL остаётся у dispatcher; tool читает только этот
снимок и разрешённые immutable документы, сверяя source SHA256.

В обычной выдаче остаются точная текущая база, task/metric/output/split/restrictions,
core schema/missingness, до шести релевантных source excerpts и короткая память.
Literal Markdown-строки таблиц, называющие входные колонки, выносятся в подробности;
все narrative constraints и строки таргета сохраняются. Полный исходный task text
доступен как task_description. Произвольный текст условия не пересказывается моделью.

`compact_memory` по умолчанию ограничивает весь JSON 12000 bytes; поздняя request
compaction использует 8000. Приоритет: связь с базой, пересечение слов вопроса,
затем давность. Длинная интерпретация не выигрывает у определения. Это прозрачная
эвристика, не embeddings и не семантический поиск. Omission counters сохраняются
при повторном сокращении. Полная controller-side schema не зависит от compaction.

`read_memory` реально читает sections, в том числе decisions/index, ideas/index,
semantics/index, dictionary/index, analyses/index, task_description, tool_guide,
experiments и experiment/ID, code/ID, diagnostics/ID, feature_execution/ID, execution/ID.
Ответ <=6000 bytes текста; offset/next_offset — UTF-8 byte offsets. Остальная оболочка
также учитывается Gateway. Ссылка не разрешает произвольный artifact/file access.
Для большой записи можно прочитать все страницы; частичный фрагмент не разрешает
цитировать ещё не доставленные ID. Большой документ может потребовать увеличения
настраиваемого числа tool turns или выбора отдельной записи/раздела.

Свежий branch feedback имеет отдельные разрешённые references; full updates читаются
по updates/results/index. Смена task/dataset/validation отвергается. Анализы из более
поздней родительской работы не попадают в frozen history; чужой report_id не расширяет
scope. Наличие pending trial ID в старой серии не разрешает читать его будущие ошибки
после fork. Для error tool нужны evaluated документы исходного scope либо доставленные
новые измерения этого branch. Runtime-память не переносится между задачами.

## Семантические изменения

Optional semantic_updates добавлены к ResearchAction и сохраняются в прежних
immutable decision artifacts. Legacy research_notes и отсутствие новых полей допустимы.
Есть виды definition, observation, hypothesis, data_error, implementation_error,
execution_error; короткое содержание связано с колонками, существующими source/evidence
IDs, scope, revision и optional supersedes. Можно записать смысл/единицы/шкалу,
сущность/гранулярность, доступность во времени, нормировку/отношения и неизвестное.

Контроллер проверяет существование/видимость ссылок, идентичность темы/колонок при
исправлении и область применимости. **Текст модели всегда model_statement_unverified**.
Источник в словаре обозначен как verbatim_source_excerpt, а не доказанный закон.
Предсказательная полезность отдельно not_established. Новая версия не перезаписывает
старую; stale/concurrent correction создаёт conflicted records. Техническая ошибка
не получает метрику и не закрывает метод как бесполезный. Полная конфигурационная
метрика и comparison_evidence по-прежнему принадлежат независимому Evaluator.

Topic/columns дают структурный ключ записи; одинаковый payload не создаёт новую версию.
Для executable programs есть schema version, dependencies и program ID. Равнозначность
свободно перефразированного текста и произвольного Python не решена текстовым хэшем.
Не вводились онтология, отдельная БД, critic agent или дополнительные LLM calls на шаг.

## Исполнимые признаки

Owned feature_program сохраняет последовательность именованных operators/inputs,
формулу, missing/zero/unseen поведение, fit scope, version/program ID и optional
**декларированные** единицы. Внутренняя несогласованность единиц суммы/разности даёт
предупреждение, а не общий запрет на исследовательские признаки. Единицы не выводятся
из имён или dtype; автоматического понимания анонимизированной шкалы нет.

NumericInteractions принимает columns=[...]. Legacy automatic selection использует
sorted numeric names + limit и записывает разрешённые inputs/products; это не отбор
релевантности. Перестановка CSV не меняет набор выбранных взаимодействий. Transform
восстанавливает исходный порядок признаков для downstream estimator.

Worker сохраняет features.json для реально исполненных owned transformers. Evaluator
связывает его с **тем же job**, чьи predictions приняты; подробности доступны в scope.
Frozen execution хранит source/params/runtime hashes для refit. Custom feature_steps
остаются декларацией, даже после хорошей метрики: произвольный Python не получает
семантическую сертификацию. Fit-only frequency/group/rank статистики и воспроизводимый
refit/predict сохранены. Feature focus теперь подсказка, без обязательного типа действия.

## Анализ и ошибки

Расширен существующий analyze_data, а не добавлен shell/eval: восемь stateless derived
операций, all/any до восьми typed clauses, до трёх grouping columns, sample coverage,
страницы полных отчётов. Фильтр применяется до выборки <=20000 строк. Присутствуют
partition/matched/sample counts, seed/rule, support/omissions и scope. Неограниченных
групп/ответов, файлов, сети, secrets или установки пакетов этот контракт не даёт.

Один logical analysis slot резервирует 1 CPU/1024 MiB в том же Budget, уменьшая
доступность train leases. Считаются elapsed и total analysis seconds. Занятость слота или исчерпание
разрешения на анализ откладывает подготовку фоновых seeds, не прерывая поиск. Сохраняются общий
deadline, phase/finalization reserve и отдельная блокировка анализов. Read allowance
120 секунд проверяется кооперативно; sample >64 MiB или diagnostic projection >256 MiB
отвергается. Это admission/кооперативные проверки, не OS watchdog для каждого pandas
вызова: чтение одного chunk/native operation не прерывается отдельно. Абсолютные
8 CPU/32 GB/3 часа по-прежнему обеспечивает платформа.

errors использует trusted saved predictions и selection labels внутри Evaluator.
Исследователь получает только aggregates; include_target запрещён. Confirmation
остается вне направления поиска. Для RMSE есть SSE, support, bias, RMSE и вклад в
полный selection SSE. Для классификации есть precision/recall/F1 и корректная slice
metric; misclassification_rate не называется Macro F1. Undefined slice AUC — null.
Группы разных колонок пересекаются, доли нельзя складывать как разбиение. Группы с
support<5 подавляются; это не обещание differential privacy от последовательных запросов.
Filtered/sample SSE явно относится к реально разобранным строкам, не экстраполяции.

## Проверка и дальнейшее сравнение

См. [VALIDATION-SEMANTIC-RESEARCH.json](VALIDATION-SEMANTIC-RESEARCH.json) — **только
статические результаты этой сессии**. AST/syntax, JSON/schema vocabulary/references,
data-only равенство schema declarations, owned imports, flat module names, allowlisted
ресурсы, frozen feature dependencies, knowledge hashes и byte equality официальных
requirements проверены без импорта agent/tools. `git diff --check` также выполнен.
Дополнительная независимая JSON Schema проверка примеров не выполнена: jsonschema
отсутствует в среде; зависимости не устанавливались. Fixtures разобраны как JSON.

Добавлены 24 regression-метода в test_semantic_research.py и обновлены существующие
fixtures: budget после tools/feedback/repair и HTTP bytes; чтение больших записей;
scope/task/fork/sibling/pending; stale conflicts и unverified model text; named inputs;
fit-only statistics/refit; custom declarations; SSE/F1; confirmation isolation;
compound filters до sample; общий resource slot; legacy/пустая память. **Ни один
тест в этой сессии не запускался.** Не запускались обучение, smoke/full агент, live
LLM, установка пакетов и сборка/исполнение ZIP. Статическая упаковка не доказывает
побайтовую правильность готового ZIP или работу на платформе.

Команды пользователя из корня checkout, в уже подготовленной среде:

```bash
python3 scripts/static_audit.py --official-data-archive /absolute/path/data_for_participants-20261021T120712Z-1-001.zip
python3 -m unittest tests.test_semantic_research tests.test_research_memory tests.test_feature_programs tests.test_feature_admission tests.test_analysis_access tests.test_evidence_scopes tests.test_tool_semantics tests.test_platform_research -v
python3 scripts/check.py --jobs 2
python3 scripts/package_solution.py --output dist/solution.zip
```

Полный отдельный live запуск (расходует API/CPU; здесь не выполнялся):

```bash
RESEARCH_MODE=adaptive GRAPH_CONTEXT_BYTES=98304 python3 solution.py /absolute/path/to/task.md --profile platform --env-file .env
```

Вторую задачу запускают отдельно с её task.md; ответ — submission.csv в месте из
условия. На leaderboard в этой сессии ничего не отправлялось. Optuna, peer coordination,
fit-only transforms, ограниченный NumPy cache, frozen validation, последний valid CSV
и platform packager сохранены у прежних владельцев.

Для сравнения старой и новой версии нужен одинаковый task/split/seed, сопоставимый
wall-time и API-budget: соответствие названных inputs фактическим operations; чтение
и использование новой проверяемой информации; повторяемость технических ошибок;
цена решений (calls, реальные usage, analysis/train/refit seconds) и качество лучшего
валидного submission. Сокращение контекста, число признаков и длина reasoning сами
по себе успехом не считаются. Контроллер не доказывает физический закон, истинность
свободного объяснения, причинную пользу каждого компонента или прирост leaderboard.
