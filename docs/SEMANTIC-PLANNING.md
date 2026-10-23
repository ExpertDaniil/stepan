# Планирование по смыслу задачи

Этот апдейт подключает определения величин к выбору вопроса и проверке реализации.
Он дополняет [SEMANTIC-RESEARCH.md](SEMANTIC-RESEARCH.md), а не заменяет граф,
независимый Evaluator или существующую память. Исходный remote HEAD:
`28610c786c5e18ebd5d1aecc744298b21fd56393`, ветка `feat/final-data-research`.

| Проблема | Подтверждение в исходниках | Изменение | Проверяемое свойство |
|---|---|---|---|
| Bootstrap извлекает формат задачи, но не модель величин | Старый TaskDraft: task_kind/target/metric/output | Необязательный для legacy TaskModel; обязательный в новых bootstrap-ответах | C02 получает смысл в том же вызове; C04 сохраняет источник и идентичность |
| Фоновый seed видит только dtype/пропуски/квантили | seed_programs(frame) | Default seed budget = 0; adaptive fallback сохраняет numeric_interactions без неявного переключения | Общий feature research остаётся доступен; фоновые seeds требуют явного opt-in |
| Свободный текст гипотезы не связывает вопрос с исполнением | Старые поля hypothesis/expected_effect | Investigation + проверка видимых ссылок и покрытия входов | Физическая гипотеза о конкретных столбцах не обосновывает автоматический выбор других столбцов |
| Выборка не устанавливает отсутствие редких нарушений | Старый анализ ограничен 20k sample | kind=check: потоковый полный проход fit-partition | Точные counts, отдельные evaluable/missing rows, без holdout labels |
| Сохранённая интерпретация мало влияет на следующее чтение | Семантические записи прежде были преимущественно текстом | Подграф по прошлому вопросу/входам и типизированные scoped revisions | Изменение трактовки меняет выдаваемые величины, отношения и проверки согласованности |

## Владельцы и маршрут

`TaskInterpreter` получает task.md и существующий bounded profile. В том же C02-ответе
модель возвращает `task_model`: entity, observation_grain, prediction_time,
target_meaning/normalization; выбранные quantities, relations и unknowns. Все имена
динамические. Каждый источник задаёт короткую дословную цитату task.md или строку
словаря; `hypothesis` явно не имеет подтверждающей цитаты. Недоступные единицы и
периоды остаются null. Контроллер проверяет цитаты и реальные имена столбцов в рамках
существующего bootstrap repair, затем повторяет проверку при C04. Это проверка
происхождения, не доказательство смысловой интерпретации.

`agent/semantic_planning.py` сохраняет модель в `task/semantic-model.json`. TaskSpec
получает только `semantic_model_artifact_id` и `semantic_model_id`. Hash/identity
проверяются при загрузке. Старые вызовы resolve_spec без task_text/task_model
сохраняют прежний путь; новый runtime всегда передаёт task_text. Mock bootstrap
передаёт явно неизвестную семантику, без выдуманной предметной области.

`prepare_details` замораживает модель рядом с уже scoped памятью. В обычный контекст
идёт semantic_task до 10 kB: определения задачи, выбранные величины/отношения,
вопросы с прикреплёнными независимыми измерениями. Подробности доступны через
read_memory: semantic_model, quantities/index, relations/index и semantics/index.
Исходные определения общие для одной задачи; поздние revisions проходят прежние
work-order/fork/donor cuts. Полная схема остаётся у контроллера. На framing
schema/tools/feedback/transcript теперь резервируется до 32 KiB при сборке контекста;
проверка фактического полного JSON/HTTP-body и общий cap сохранены.

Для subset_count=[subset,superset] контроллер может предложить точный check
subset>superset; для normalized_by=[normalized,denominator] — проверку нулевого
знаменателя. Это общие логические следствия объявленных отношений, а не
датасетные формулы или обязательные фазы. Target не включается в вычисляемые inputs.
Для использования предложенного check модель должна выбрать вопрос и допущения.

## Исследовательское решение

В новых adaptive-контекстах final JSON обязан содержать investigation:
question, basis_ids, columns, interpretation, mechanism, assumptions,
expected_observation, alternative_explanation, decision_rule. Это короткое
проверяемое обязательство до измерения. Свободное совместное изменение признаков,
обработки, модели и параметров сохранено; нет обязательной абляции, новых агентов,
квот на признаки или дополнительного вызова критика. Для tuning/stop допустим
пустой список columns. Для начальной exploratory-гипотезы допустим basis_ids=[].
Начальная подборка offline knowledge использует определение таргета/нормализации
и отношения модели задачи, затем существующие вопросы. Анонимный пример
cross/frequency/group больше не повторяется в каждом обычном контексте; примеры
доступны в tool guide. Это меняет исходные подсказки без дополнительного LLM-вызова.

Computing analyze_data queries тоже прикладывают investigation до вычисления.
Чтение schema/dictionary/report не требует его. Ссылка должна быть показана модели
в проекции, полном разрешённом чтении или текущем диалоге. Raw зависимости
feature_program должны входить в investigation.columns, в том числе через цепочку
производных операций. Для check покрываются также все входы условий и фильтров.

Physical/mathematical интерпретации проверяются на согласованность с объявленными
типами/единицами. Код периода/identifier/category не становится календарным временем
через вычитание. Сложение/вычитание известных разных единиц требуют явного
преобразования; физический логарифм требует безразмерного аргумента. Произведения
разных единиц разрешены. Проверка единиц консервативная, по буквальным обозначениям,
с простой передачей символических единиц по программе; полноценного конвертера
единиц и доказателя физики здесь нет. Статистические/exploratory представления
сохраняют свой статус. Неизвестная семантика создаёт предупреждение, не выдуманный факт.

cyclic_sin/cyclic_cos исполняют sin/cos(2*pi*(x-origin)/period), period>0,
origin по умолчанию 0. Период объявляется в исходной шкале и сравнивается с известным
периодом модели задачи. Эти операции доступны в feature_program и stateless анализе;
program version повышен до feature-program-3. Их геометрия не гарантирует ML-выигрыш.

## Наблюдения и исправления

kind=check принимает до 8 именованных violation-предикатов. Потоковый проход читает
нужные input-столбцы, оставляет fit rows, вычисляет stateless derived, применяет
фильтры и считает нарушения. Для сравнения строка с отсутствующим операндом входит
в missing_rows; is_missing/not_missing оцениваются на всех matched rows. Числа
evaluable/missing дают явный знаменатель. Sample не используется. Действуют прежние
общие analysis CPU/RAM slot и cooperative 120s deadline, дополнительно chunk cap
64 MiB. Сырой target запрещён даже при ссылке только внутри violation; raw
validation/confirmation labels не читаются. Как и существующий CSV-reader, один
декодируемый chunk не прерывается посередине; это не hard process memory watchdog.

Отчёт сохраняет исходный вопрос. Точные check-наблюдения входят в recent analysis
projection и доступны целиком по scoped read. Final decisions сохраняют investigation
в прежних immutable research_decision/research_intent; build_research_memory
связывает его с реальными конфигурациями и Optuna trials. Валидный score означает
измеренную конфигурацию с неустановленным механизмом. Техническая ошибка оставляет
гипотезу непроверенной. Автоматического причинного вердикта и закрытия идеи нет.

semantic_updates.properties дополнен quantity_kind, availability_status, period,
relation_kind. Quantity revision относится к одному столбцу. Relation revision — к
2–6, с фиксированным порядком бинарных normalized_by/subset_count/precedes. Активные
revisions меняют scoped planning view; исходная модель остаётся доступной отдельно.
Конфликты снимают определённость соответствующих свойств и требуют разбора. Ссылки
и даже точный эмпирический результат не делают модельный текст verified.

## Границы результата

Контроллер не доказывает причинность, достаточность цитаты для интерпретации,
качество альтернативного объяснения или способность LLM мыслить. Заданный
expected_observation не означает, что тест статистически различает причины.
Произвольный custom Python остаётся декларацией: покрытие его фактических
зависимостей/физики не сертифицируется. Prediction-time availability здесь
проверяется для заявленных новых программ; это не автоматическая проверка
доступности каждого raw input произвольной модели. Реальные имена/формулы публичных
задач не зашиты. Дополнительные LLM-вызовы для смыслового bootstrap не добавлены;
ошибки ответа используют существующий ограниченный repair и расходуют общий бюджет.

Статические результаты: [VALIDATION-SEMANTIC-PLANNING.json](VALIDATION-SEMANTIC-PLANNING.json).
Новые regression-сценарии написаны, **не запускались**. Обучение, агент, smoke,
платные/live API, установка пакетов и создание platform ZIP не выполнялись.

После этой сессии, в подготовленной среде пользователя:

```bash
python scripts/check_semantic_planning_static.py
python -m unittest tests.test_semantic_planning tests.test_semantic_research tests.test_research_memory tests.test_tool_semantics tests.test_feature_programs tests.test_adaptive_research -v
python scripts/check.py
python scripts/package_solution.py --output dist/solution.zip
python solution.py /absolute/path/to/task.md --env-file .env
```

Последние четыре команды требуют отдельного запуска пользователем: тесты включают
реальные ML/process проверки, упаковщик создаёт ZIP, solution выполняет обучение
и live API по .env. В существующем .env установить RESEARCH_MODE=adaptive и
FEATURE_SEED_BUDGET_FRACTION=0: старое явное значение 0.15 перекрывает новый default.
Остальные бюджеты не увеличивались. Проверка качества требует сравнения при
сопоставимых wall-time/API budget: соответствие замысла исполнению, изменения
решений после новых наблюдений, повторы технических ошибок, стоимость и ML-метрика.
