# Совместная разработка

Начните с `README.md`, `docs/IMPLEMENTATION.md` и согласованных контрактов C00–C24.

## Первый Git

```bash
git init -b main
git add .
git status
git commit -m "Initialize contract-driven AutoML agent skeleton"
```

Перед commit просмотрите список добавленных файлов. `.env`, данные, модели, `runs/`, `dist/` и кэши исключены. Удалённый репозиторий и права доступа команда настраивает отдельно.

## Ветки и владельцы

- A: `feature/runner`, `feature/budget`, `feature/store`, `feature/observability`.
- B: `feature/llm-gateway`, `feature/context`, `feature/researcher`, `feature/task-interpreter`.
- C: `feature/recipes`, `feature/evaluator`, `feature/submission`, `feature/data-tools`.

Ветка соответствует небольшому изменению. После принятия общий `main` остаётся запускаемым. Для GitHub Actions/GitLab CI используйте один provider-neutral шаг `python scripts/check.py`; сервис CI здесь не предполагается.

## Границы модулей

- Researcher возвращает план и артефакты, не изменяет SQLite.
- Только dispatcher-thread записывает SQLite. Tool/context artifacts имеют собственную потокобезопасную регистрацию.
- Все межмодульные переходы проходят `ContractBus.call`. Имя endpoint — логический интерфейс, не сетевой сервис.
- C13 release допустим после подтверждённого завершения дерева процессов. Отмена — C24; терминальное событие — C15.
- После изменения схемы согласуйте producer/consumer, обновите примеры и контрактные тесты.
- Новая метрика требует реализации, направления оптимизации, правил предсказания и осмысленного теста.
- Обучаемая предобработка выполняется внутри pipeline на train-части. Общую валидацию нельзя менять для одного кандидата.
- API-ключи, скрытые test labels и ответы внешнего лидерборда не входят в контекст.

## Параллельный первый цикл

A проверяет резервирование/процессы/дедлайн; B работает через MockTransport и FakeTransport; C проверяет рецепты и публикацию на синтетических задачах. Реальные API-значения и загрузка на платформу добавляются после локального сквозного прогона.
