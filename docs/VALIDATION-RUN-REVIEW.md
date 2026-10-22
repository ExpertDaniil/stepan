# Проверка поиска и журнала · v0.4.3

22 октября 2026 года. База: `ccde4624f9e16a29b1a0cb567618b3b8ebc6f21e`.

`python3 scripts/check.py`: **155 тестов, 358.601 с, OK, exit code 0**.
Сохранены 138 прежних тестов; добавлены 16 проверок поиска/файлов и один сквозной сценарий warmup.
Полный успешный вывод — `validation-run-review-unittest.log`, числа и ограничения — `validation-run-review.json`.
Ранее проходили 154 теста; после последней проверки archived late results выполнен новый полный прогон 155.
Во время финального прогона runtime-файлы не менялись: hashes всех 51 файла проверены до и после.

## Что проверено

- Собственные совместимые результаты наследуются при fork, checkpoint неизменен, GBranchSeed остаётся в typed journal и export.
- Старые, более поздние и sibling-наблюдения не смешиваются: разрешённые стартовые ссылки явные; ошибки не становятся базами.
- Новая ветка может доработать собственную сильную базу; incompatible historical reference не блокирует продвижение.
- Четыре distinct full решения без gain дают плато; три не дают. Суммарные малые improvements сбрасывают streak.
- Повтор seed/solution, точный кеш, неудачное обучение и archived late evaluations не создают ложную серию.
- Та же логика работает для minimize RMSE и maximize classification metric; timer начинается с branch activation/resume.
- Каждый peer получает отношение exploit/explore 3:1; trusted warmup не расходует его исследовательский цикл.
- Обычные warmups двух семейств предшествуют feature-вариантам. Плановый вариант второго peer подтверждает transfer без лишнего API/fit.
- Реальный синтетический end-to-end: два bootstrap API-вызова, четыре warmup, controlled source/pilot, три ветки, четыре seed records, валидный CSV.
- Bounded групповые diagnostics доходят в measured memory; в классификации нет матрицы N×N/построчных labels/неограниченного списка классов.
- Сжатый snapshot большого RandomForest останавливается при byte cap, partial удаляется, предсказания остаются прежними.
- Общая квота snapshots резервируется для параллельных jobs; выключенные веса не мешают публикации CSV.
- Unicode stdout/stderr ограничены именно в байтах; marker входит в лимит, pipe дренируется после усечения.
- Анализатор считает размеры, не загружая pickle/CSV; graph-events.jsonl не становится вторым run.
- Сохранены negative tests: T4/T5 rollback, stale fences, TTL, provider отказ, cleanup process groups, validation/output checks, pinned reconstruction.

## Совместимость и поставка

Все 100 исходных non-G definitions, включая C00–C24, совпадают с базовым commit.
Добавлен GBranchSeed; optional additions к GPlateau, GExperimentMemory, GPolicy и GGraphExport
проверяются producer/consumer, fixtures и runtime. Новых runtime dependencies нет.
`requirements.txt` побайтово равен `requirements.organizers.txt`.

Плоская платформа: 41 корневой файл, 51 runtime-файл, 167814 байт.
SHA-256: `7c3068e7919d425286aeb723a3e340bf2abdcd137d2e113688a9e48a26cb8890`. Исходные Python-модули читаемы и побайтово равны Git-исходникам.
`git diff --check` пройден. Настоящие данные, weights, исходный пользовательский run и ключи в Git не копировались.

## Среда и пределы

Python 3.12.14; SQLite 3.53.1; NumPy 2.3.5; pandas 2.2.3;
SciPy 1.17.0; scikit-learn 1.8.0; joblib 1.5.3.
OpenAI SDK и CatBoost здесь не установлены; зависимости во время работы не устанавливались.

Unit-оценки состояния заданы явно. Сквозные проверки обучают модели и считают метрики на синтетических CSV;
имитируются только ответы LLM. Реальный пользовательский архив анализировался read-only без загрузки pickle,
но его train/test отсутствуют. Новый пользовательский RMSE, официальный контейнер, live API и непрерывные
три часа здесь не проверялись. Отдельного архива классификации на 3 ГБ нет; причина всего его объёма не утверждается.


## Отдельная проверка без сохранённых весов

`MODEL_SNAPSHOT_MAX_BYTES=0 python3 -m unittest tests.test_graph_end_to_end.GraphEndToEndTests.test_real_training_confirms_transfer_forks_and_backtracks -v`
завершился с OK: 1 тест за 95.844 с. В исходном запуске snapshot весов отключён,
затем решение реконструируется из закреплённых code/params/protocol и заново обучается.
Предсказания совпали с опубликованными с rtol=atol=1e-9 в этой же среде.
Вывод сохранён в `validation-run-review-weightless.log`.
