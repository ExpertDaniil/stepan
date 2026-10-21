# Проверка Research Graph v0.4.0

Дата: 21 октября 2026. Проверенная основа — v0.3.1, commit
3b6d7d84879fffbe76d111d05a9d58816b009e77. Изменения этого commit, включая очистку worker helpers,
сохранены. До реализации отдельно прошли исходные 74 теста v0.3.0; финальный прогон включает все 77 тестов v0.3.1.

`python3 scripts/check.py`: **116 тестов, OK, 143.095 s**. Из них 39 новых проверок графа.
Лог: validation-graph-unittest.log. Машиночитаемые результаты и версии: validation-graph.json.

## Что проверено

| Область | Проверка |
|---|---|
| Совместимость | Все C00–C24 definitions структурно равны исходным; 44 операции и их request/success/error fixtures проходят validator |
| Собственные контексты | Два peer завершаются в обратном порядке; каждый продвигает свою голову; чужая ревизия не инвалидирует задание |
| Origin и fences | Stale/expired proposal не создаёт experiment; late valid result остаётся в origin и может стать global best; old execution fence отвергается |
| Перенос | Контролируемые source/recipient пары; mixed hyperparameters не засчитываются как эффект feature; gate не принимает score из LLM |
| Ветви | Два ребёнка атомарно; лимит откатывает оба; restored heads раздельны; nested fork наследует прежний ban |
| Возврат | G07/G14 выбирают сохранённую альтернативу; контекст берётся из reject-линии; прежний winner остаётся доступен |
| Контекст | Переполнение mandatory части — явная ошибка и освобождение peer; full API request проверяется вместе со schema; локальное отклонение не вызывает transport |
| Cache/seed | Exact dedup добавляет binding; другой seed сохраняет solution и создаёт другой experiment; повтор G04 не перезапускает evaluated модель |
| Транзакции | Инъекция сбоя T4 откатывает native reservation и graph; T5 — selection/CAS; durable key переживает новый ContractBus |
| Восстановление | Сбой rename после durable intent закрывает G12; corrupt pin блокирует поиск; registry rehydrate из SQL locations; unstarted work закрывается; trusted C16 может завершить T5 |
| SQL | FK, composite head/experiment/solution, immutable records, DAG cycle, binary tree |
| Отказы | C14 spawn после baseline завершает оба задания; сохранены CSV/leases и читаемый graph export; прежние timeout/process-group tests проходят |
| ML transform | NumericInteractions ограничен 8 fit-столбцами/36 новыми признаками; обученная линейная модель решает синтетическую квадратичную задачу |
| Настоящий сквозной цикл | Synthetic CSV → реальные fits/метрики → controlled transfer → pilot → fork → plateau → reject → final ranking |
| Реконструкция | Выбранный эксперимент восстановлен из owned-runtime ZIP, реально refit-нут; предсказания совпадают при rtol=atol=1e-9 в той же среде |
| Поставка | Плоский ZIP запускается без исходного репозитория; только solution.py + точный official requirements.txt; повторная сборка воспроизводима |

Искусственно инъецированные scores используются только в state-machine unit fixtures и не считаются
подтверждением качества ML. Сквозной тест использует имитированные LLM-ответы, настоящие процессы обучения,
настоящие predictions/metrics и специально построенные синтетические данные для прохождения веток алгоритма.

## Среда

Python 3.12.14; SQLite 3.53.1; NumPy 2.3.5; pandas 2.2.3; SciPy 1.17.0; scikit-learn 1.8.0;
joblib 1.5.3. Это фактическая среда проверки, а не утверждение о совпадении с каждым official pin.
requirements.txt и requirements.organizers.txt не изменены и побайтово равны.

Не запускались: предоставленный live API (SDK/credentials не использовались), CatBoost, Docker/официальный
образ организаторов, трёхчасовая production-задача и реальная потеря питания FS. Восстановление проверено
инъекцией исключений между шагами и новым ContractBus; это не power-loss certification.
Конкретный score на скрытых задачах из этих инфраструктурных проверок не следует.
