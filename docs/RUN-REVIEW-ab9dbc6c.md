# Разбор запуска ab9dbc6c и исправления согласованности поиска

Источник: `run-20261022T144843Z-ab9dbc6c.zip`. Сохранённый собственный runtime побайтово совпадает с `1103d1f`. Проанализированы события и SQLite; веса не загружались, обучения и тесты в рамках этой работы не запускались.

## Что произошло

Запуск работал **12 мин 33,660 с** при общем бюджете три часа. Остановка — `KeyboardInterrupt` в ожидании train-job, а не плато, квота API, число экспериментов или переход в доводку. Последний проверенный CSV сохранился. Прерванный worker отменён; один следующий эксперимент остался в очереди незавершённого запуска.

Зарегистрированы 65 экспериментов: 63 evaluated, один cancelled, один queued. Процесс обучения начат 64 раза; кроме этого выполнены пять полных refit для публикации CSV. Уникальных оценённых конфигураций 63, включая Dummy; обучаемых семейств пять. Это варианты фиксированного каталога, а не 63 разные ML-архитектуры.

| Семейство | Получили метрику | Лучший RMSE |
|---|---:|---:|
| HistGradientBoosting | 23 | 43,646152 |
| CatBoost | 27 | 43,852276 |
| Extra Trees | 4 | 44,257403 |
| Ridge | 4 | 45,518524 |
| Random Forest | 4 | 50,929536 |
| Dummy | 1 | 56,799432 |

Лучший HGB: max_iter=300, max_leaf_nodes=31, learning_rate=0.1, l2_regularization=1, min_samples_leaf=10, early_stopping=False, target_transform=shifted_log1p, numeric_interactions=False. Это оценка фиксированного локального протокола, не внешний/приватный скор.

По сравнению с лучшим начальным обучаемым решением 44,922615 RMSE снизился до 43,646152: улучшение около **2,84%**. Основной результат появился за первые 2 мин 24 с; затем до прерывания опубликованных улучшений не было.

## Ветви и время

Создано три узла: root, accept(target_transform), reject(target_transform). Это **одна развилка**, без дальнейших переходов. Корень: примерно 65 с и пять оценок. Accept: 11 мин 29 с до прерывания, 58 оценок. Reject создан с собственными измеренными исходными решениями, оставался paused, новых испытаний не получил.

Все 61 решения планировщика: один FORK и 60 CONTINUE. BACKTRACK, FINALIZE, REFINE_BEST не было. Последняя оценка плато: 55 разных попыток без существенного прогресса, 617 с с последнего существенного прогресса, epsilon=0,043646. Таймер 900 с ещё не достигнут. По этим логам нет признаков слишком быстрых переходов по веткам; в данном запуске альтернативная ветвь вообще не исследовалась.

| Операция | Медиана | Диапазон |
|---|---:|---:|
| LLM: исследовательское предложение (16 запросов) | 22,45 с | 10,32–64,87 с |
| Процесс train, включая запуск и предсказание (64) | 4,93 с | 1,40–18,29 с |
| Интервал между началами партий (30 интервалов) | 14,68 с | 0,07–70,69 с |

Интервал партии может включать две последовательные тренировки, параллельное ожидание двух LLM, оценку, публикацию и служебный переход. Это не чистое время размышления отдельной модели. Настройка train-workers=1 при двух исследовательских контекстах объясняет последовательное обучение.

## Подтверждённые проблемы

1. **Повторные предложения.** 18 API-запросов: два служебных и 16 исследовательских. Четыре исследовательских предложения заменены как точные дубли. HGB max_iter=600, learning_rate=0.1 уже дал RMSE 44,191908 против 43,646152 у 300 итераций, но затем предлагался снова. HGB 600/0.05 тоже предложен повторно. Дедупликация не допустила повторного fit, но вызовы LLM потрачены. В позднем контексте старое испытание 600 вытеснено локальными соседями 239/240/241.
2. **Перекос локального перебора.** Выбрано 48 локальных вариантов; 46 получили метрику. Перебор начинал с первого параметра рецепта. Серии часто меняли только число итераций/деревьев; соседние значения 239/240/241 и 319/320 давали много разных скоров при малом разнообразии гипотез. Ни один локальный вариант не включил numeric_interactions.
3. **Серия не использовала обратную связь.** Три продолжения заранее строились от предложения, даже если оно проигрывало измеренной базе; после улучшения список тоже не перестраивался от нового лучшего результата.
4. **Слишком короткая память для длинного поиска.** Восемь подробных фактов с повторяющимися diagnostics не дают устойчивой памяти о старых проигравших конфигурациях. Отдельная компактная сводка старых LLM-предложений полезнее для предотвращения повторов.

Измеренные групповые ошибки показывают трудные State_8 и Commercial; отрицательное смещение само по себе не доказывает недостаточную сложность модели. Систематически увеличивать число итераций вопреки измеренным регрессиям — слабая стратегия. Каталог в этом режиме ограничивает выбор пятью семействами и зарегистрированными преобразованиями. По внешним логам нельзя установить, сколько скрытых рассуждений сделал провайдер, или объявить найденный минимум глобальным оптимумом.

## Внесённые исправления

- Доноры выбираются из принятых идей всей родительской линии с проверкой всех унаследованных запретов. Reject(B) после accept(A) получает допустимый источник A, а не запрещённый источник B.
- Частичный ответ каталожной LLM при доработке своего семейства дополняет измеренную базу и в поиске, и в доводке. При проверке другого семейства сохраняется старт от его дефолтов.
- Таймаут ответа принадлежит C10/Gateway. Срок поручения покрывает оставшееся время запуска, включая ожидание обучения; готовый ответ не протухает из-за работы другого peer.
- У границы фаз применяется прогноз следующей партии и фиксированная временная граница, независимая от меняющегося прогноза refit. Новое обучение требует места до границы поиска для fit и полного refit/публикации; ответ LLM ограничен бюджетом поисковой фазы. Обучение, уже запущенное с допуском, продолжает действовать в рамках своего общего lease. Лучший вариант публикуется после завершённой пачки train, до ожидания следующего ответа/пачки.
- Обычный BACKTRACK делает pause/resume. Архивирование допускается перед созданием полной пары детей и только для уже проверенных/исчерпанных paused-ветвей; непроверенные альтернативы защищены. История и глобальный лучший результат сохраняются.
- В доводке фиксируются зарегистрированные feature-переключатели; остальные параметры, включая enum max_depth и max_features, доступны для подбора.
- Локальные варианты чередуют оси параметров. Мелкие единичные сдвиги крупных целочисленных параметров исключены из обычного поиска и оставлены для финальной доводки. Каждое продолжение серии использует собственный измеренный лучший результат данного семейства на допустимой линии. Локальный поиск сохраняет предпочтение доработки базы 3:1.
- Компактная сводка до 24 испытаний сохраняет старые предложения LLM и лучшие результаты по семействам; остальные подробные факты обходятся без повторяющихся групповых diagnostics. При превышении размера сводка явно сокращается, минимум до восьми записей.
- Плато требует минимум четыре различные завершённые проверки **от каждого peer** после существенного прогресса или возобновления ветви, плюс 900 с без существенного суммарного улучшения. Малые улучшения обновляют best независимо от epsilon. В GPlateau добавлен необязательный счётчик по peer, старые контракты C00–C24 сохранены.

## Границы проверки

По просьбе пользователя тесты и обучение не запускались. Выполнен статический разбор кода и проверка синтаксиса/JSON, согласованности исходных C00–C24, зависимостей и плоской упаковки. Регрессионные сценарии добавлены и скорректированы, но результаты их исполнения неизвестны. Улучшение нового кода на реальных данных не измерено.

85/15 — планирование с резервом и допуском по прогнозу, а не гарантия точного wall-clock соотношения: обучение может длиться дольше прогноза. Глобальный дедлайн и запас на публикацию действуют отдельно; уже проверенный CSV сохраняется. Поведение у границы необходимо измерить последующим запуском в целевой среде.

## Оценённые варианты

Ниже все 63 локальные оценки для проверки разнообразия. Параметры — фактически применённые; при равном семействе это тюнинг одного рецепта. Dummy оставлен как исходная точка.

| № | Семейство | RMSE | Параметры |
|---:|---|---:|---|
| 1 | dummy | 56.799432 | `{"strategy":"mean"}` |
| 2 | catboost | 52.197001 | `{"depth":6,"iterations":800,"l2_leaf_reg":3.0,"learning_rate":0.08,"numeric_interactions":false,"target_transform":"identity"}` |
| 3 | hist_gradient | 48.529673 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":300,"max_leaf_nodes":15,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"identity"}` |
| 4 | catboost | 46.648667 | `{"depth":6,"iterations":800,"l2_leaf_reg":3.0,"learning_rate":0.08,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 5 | hist_gradient | 44.922615 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":300,"max_leaf_nodes":15,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 6 | hist_gradient | 44.191908 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":600,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 7 | catboost | 46.129271 | `{"depth":7,"iterations":800,"l2_leaf_reg":3.0,"learning_rate":0.08,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 8 | hist_gradient | 43.646152 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":300,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 9 | catboost | 45.961040 | `{"depth":7,"iterations":400,"l2_leaf_reg":3.0,"learning_rate":0.08,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 10 | hist_gradient | 43.986360 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":480,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 11 | catboost | 46.177326 | `{"depth":7,"iterations":640,"l2_leaf_reg":3.0,"learning_rate":0.08,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 12 | hist_gradient | 44.899094 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":750,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 13 | catboost | 46.237830 | `{"depth":7,"iterations":1000,"l2_leaf_reg":3.0,"learning_rate":0.08,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 14 | linear | 45.519461 | `{"alpha":1.0,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 15 | catboost | 44.902182 | `{"depth":7,"iterations":800,"l2_leaf_reg":3.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 16 | linear | 45.519429 | `{"alpha":0.5,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 17 | catboost | 44.401677 | `{"depth":7,"iterations":400,"l2_leaf_reg":3.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 18 | linear | 45.518524 | `{"alpha":0.8,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 19 | catboost | 44.556858 | `{"depth":7,"iterations":640,"l2_leaf_reg":3.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 20 | linear | 45.522192 | `{"alpha":1.25,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 21 | catboost | 45.001258 | `{"depth":7,"iterations":1000,"l2_leaf_reg":3.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 22 | hist_gradient | 44.251496 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":150,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 23 | catboost | 44.448995 | `{"depth":7,"iterations":400,"l2_leaf_reg":8.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 24 | hist_gradient | 44.840808 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":75,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 25 | catboost | 44.488528 | `{"depth":7,"iterations":200,"l2_leaf_reg":8.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 26 | hist_gradient | 44.404491 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":120,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 27 | catboost | 44.316309 | `{"depth":7,"iterations":320,"l2_leaf_reg":8.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 28 | hist_gradient | 44.069567 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":188,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 29 | catboost | 44.733539 | `{"depth":7,"iterations":500,"l2_leaf_reg":8.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 30 | random_forest | 51.154082 | `{"max_depth":16,"max_features":"sqrt","min_samples_leaf":2,"n_estimators":100,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 31 | extra_trees | 44.367609 | `{"max_depth":null,"max_features":1.0,"min_samples_leaf":2,"n_estimators":200,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 32 | extra_trees | 44.257403 | `{"max_depth":null,"max_features":1.0,"min_samples_leaf":2,"n_estimators":100,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 33 | random_forest | 51.321811 | `{"max_depth":16,"max_features":"sqrt","min_samples_leaf":2,"n_estimators":50,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 34 | extra_trees | 44.412243 | `{"max_depth":null,"max_features":1.0,"min_samples_leaf":2,"n_estimators":160,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 35 | random_forest | 50.929536 | `{"max_depth":16,"max_features":"sqrt","min_samples_leaf":2,"n_estimators":80,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 36 | extra_trees | 44.339538 | `{"max_depth":null,"max_features":1.0,"min_samples_leaf":2,"n_estimators":250,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 37 | random_forest | 51.216121 | `{"max_depth":16,"max_features":"sqrt","min_samples_leaf":2,"n_estimators":125,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 38 | hist_gradient | 44.296013 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.05,"max_iter":600,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 39 | catboost | 43.852276 | `{"depth":7,"iterations":320,"l2_leaf_reg":4.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 40 | hist_gradient | 44.034844 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.05,"max_iter":300,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 41 | catboost | 45.379200 | `{"depth":7,"iterations":160,"l2_leaf_reg":4.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 42 | hist_gradient | 44.332772 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.05,"max_iter":480,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 43 | catboost | 43.855242 | `{"depth":7,"iterations":256,"l2_leaf_reg":4.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 44 | hist_gradient | 44.473461 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.05,"max_iter":750,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 45 | catboost | 44.170113 | `{"depth":7,"iterations":400,"l2_leaf_reg":4.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 46 | catboost | 44.920671 | `{"depth":7,"iterations":320,"l2_leaf_reg":2.5,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 47 | hist_gradient | 45.331162 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":300,"max_leaf_nodes":63,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 48 | hist_gradient | 43.826598 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":150,"max_leaf_nodes":63,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 49 | catboost | 45.615238 | `{"depth":7,"iterations":160,"l2_leaf_reg":2.5,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 50 | hist_gradient | 44.905953 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":240,"max_leaf_nodes":63,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 51 | catboost | 44.569381 | `{"depth":7,"iterations":256,"l2_leaf_reg":2.5,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 52 | hist_gradient | 46.447484 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":375,"max_leaf_nodes":63,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 53 | catboost | 45.334572 | `{"depth":7,"iterations":400,"l2_leaf_reg":2.5,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 54 | hist_gradient | 43.756170 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":240,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 55 | catboost | 44.891538 | `{"depth":7,"iterations":512,"l2_leaf_reg":4.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 56 | hist_gradient | 44.023595 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":192,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 57 | catboost | 44.130941 | `{"depth":7,"iterations":410,"l2_leaf_reg":4.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 58 | hist_gradient | 43.758756 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":239,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 59 | catboost | 45.405399 | `{"depth":7,"iterations":640,"l2_leaf_reg":4.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 60 | hist_gradient | 43.810507 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":241,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 61 | catboost | 46.600131 | `{"depth":7,"iterations":1024,"l2_leaf_reg":4.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 62 | catboost | 43.852368 | `{"depth":7,"iterations":319,"l2_leaf_reg":4.0,"learning_rate":0.05,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
| 63 | hist_gradient | 44.093391 | `{"early_stopping":false,"l2_regularization":1.0,"learning_rate":0.1,"max_iter":450,"max_leaf_nodes":31,"min_samples_leaf":10,"numeric_interactions":false,"target_transform":"shifted_log1p"}` |
