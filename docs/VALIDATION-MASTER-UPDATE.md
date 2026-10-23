# Статическая проверка мастер-апдейта

Дата: 2026-10-23. Исходный commit: b79c7e055d9bcbc417eb7c01c4b9125b3d5bb7f4.
До изменения сверены все 334 исходных Git blob, точное дерево и commit; checkout был чистым.

Выполнено:

```bash
python3 scripts/static_audit.py --official-data-archive ../project_sources/02-data_for_participants-20261021T120712Z-1-001.zip --output docs/validation-master-static.json
git diff --check
```

`static_audit.py` не импортирует модули проекта. Он читает AST, JSON и байты ресурсов;
отчёт содержит SHA256 каждого из 55 runtime-модулей, 99 разобранных Python-файлов,
19 allowlisted текстовых ресурсов и сравнение requirements с официальным архивом.
Количество JSON/методов тестов и точные исходные SHA — в [validation-master-static.json](validation-master-static.json).
Проверены также owned imports, отсутствие коллизий плоских имён и тот факт, что packager
передаёт source content в entries без преобразования, ссылки/словарь schema, хеши knowledge manifest.
Это проверка входов и исходника пути упаковки. ZIP не строился; побайтовое соответствие
фактически созданного ZIP и выполнение временного runtime здесь не проверены.

Чтение сохранённого run f5e641a1 подтвердило 27 запросов / 78885 reported completion tokens,
12 trial одной Optuna-серии, два research decision и окончание серии после последнего LLM-запроса.
Эти числа относятся к историческому запуску, не к проверке нового кода.

По прямому ограничению пользователя не запускались test suite, отдельные regression-сценарии,
обучение, полный агент, live LLM, установка зависимостей. Новые тесты не имеют статуса passed.
В них есть concurrency/transcript/rolling-dispatch проверки без сети, а также маленькие реальные
ML fit для проверки нескольких целей; последние запустятся только при локальной команде пользователя.

Команды unittest/smoke/full run/упаковки и чтения результата приведены в
[MASTER-UPDATE.md](MASTER-UPDATE.md#локальная-проверка-в-существующей-подготовленной-среде).
Эффект на test score, качество гипотез, native estimator/API совместимость и фактическое применение
reasoning-параметров прокси остаются неподтверждёнными.

При финальной ревизии отдельно согласованы RProtocol selection/confirmation fields и
версия evaluator в graph task contract, сохранена совместимость минимального legacy dispatcher,
добавлен явный cooldown после ошибок транспорта. Transcript regression проверяет совпадение
нормализованных assistant/tool IDs, а не равенство исходным provider IDs.
