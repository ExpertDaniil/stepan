"""A bounded factual terminal view; the JSONL record remains authoritative."""
import json
import math


REASONS = {
    'time_reserve': 'достигнут запас времени на окончательное обучение и сохранение',
    'no_affordable_candidates': 'не осталось подходящих вариантов, укладывающихся во время',
    'search_budget_closed': 'поиск закрыт по общему бюджету',
    'llm_call_limit': 'исчерпано число обращений к языковой модели',
    'llm_output_budget': 'исчерпана квота ответов языковой модели',
    'duplicate': 'такая конфигурация уже проверялась',
    'duplicate_or_unaffordable': 'повторная конфигурация или недостаточно времени',
    'stopped': 'исследователь отказался от нового опыта',
    'stopped_proposal': 'исследователь не предложил новый опыт',
    'predicted_duration': 'прогноз обучения и сохранения превышает остаток времени',
    'unaffordable': 'обучение и сохранение не укладываются в остаток времени',
    'next_research_batch_would_cross_phase_boundary': 'следующая партия не помещается в этап поиска',
    'search_time_share_reached': 'время, выделенное на поиск, израсходовано',
    'Measured plateau; resume a saved alternative': 'измерено плато; возвращаем сохранённую альтернативу',
    'Reserve time for a counterfactual trial': 'сохраняем время для проверки альтернативы',
    'No affordable candidate in this branch; try a saved alternative': 'в этой ветви нет подходящего опыта по времени; пробуем сохранённую альтернативу',
    'A controlled component and recipient pilot passed the transfer gate': 'перенос приёма прошёл отдельную измеренную проверку у получателя',
    "Apply the confirmed component to this peer's restored base before tuning": 'проверяем подтверждённый приём на восстановленной собственной базе исследователя',
    'Run one controlled recipient pilot before collective adoption': 'перед общим применением отдельно проверяем приём у получателя',
}


def short(value, limit=320):
    text = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else str(value)
    text = ' '.join(text.split())
    return text if len(text) <= limit else text[:limit-1]+'…'


def number(value):
    return f'{value:.6g}' if isinstance(value, (int, float)) and math.isfinite(value) else short(value)


def seconds(value):
    return f'{value:.0f} с' if isinstance(value, (int, float)) else 'не указано'


def reason(value):
    if isinstance(value, str) and value.startswith('Local catalogue search: '):
        return 'продолжаем локально: '+reason(value.split(': ', 1)[1])
    return REASONS.get(value, short(value)) if isinstance(value, str) else short(value)


class ConciseConsole:
    def __init__(self):
        self.progress_buckets = {}
        self.last_decision = None

    def render(self, r):
        event = r['event']
        get = lambda name, default='—': r.get(name, default)
        eid = get('experiment_id')
        if event == 'run.started':
            return f"Задача: {get('task_path')}. Журнал: {get('run_directory')}."
        if event == 'run.state_changed':
            state = {'initializing': 'подготовка', 'baseline': 'первый проверенный ответ',
                     'searching': 'исследование', 'finalizing': 'окончательное обучение и сохранение'}.get(get('state'))
            return 'Этап: '+state+'.' if state else None
        if event == 'validation.created':
            return (f"Проверка качества: {get('strategy')}, обучение {get('train_rows')} строк, "
                    f"проверка {get('validation_rows')}; метрика {get('metric_id')}. Параметры: {short(get('parameters'))}.")
        if event == 'research.hypothesis_proposed':
            return f"Предложение {get('researcher_id')}, явная гипотеза: {short(get('hypothesis'))} Модель: {get('recipe_id')}; параметры: {short(get('hyperparameters'))}."
        if event == 'research.analysis_completed':
            return (f"Получены дополнительные факты: {get('kind')}, столбцы {short(get('columns'))}, "
                    f"строк {get('rows')}, область {get('scope')}. Отчёт: {get('artifact_id')}.")
        if event == 'research.stop_requested':
            return f"{get('researcher_id')} завершил исследование: {reason(get('reason'))}."
        if event == 'search.local_trial':
            return f"Локальная проверка каталога: {get('researcher_id')}, ветвь {get('branch_id')}, {get('recipe_id')}; параметры: {short(get('hyperparameters'))}."
        if event == 'search.admission':
            verdict = 'допущен' if get('allowed') is True else 'отклонён по времени'
            return (f"Опыт {eid} {verdict}: прогноз {seconds(get('estimated_seconds'))}; "
                    f"осталось {seconds(get('remaining_seconds'))}, запас {seconds(get('reserve_seconds'))}.")
        if event == 'job.started':
            phase = 'окончательное обучение' if get('phase') == 'refit' else 'обучение для проверки'
            return (f"Запущено {phase}: {eid}, {get('recipe_id')}; параметры: {short(get('hyperparameters'))}. "
                    f"Потоки: {get('cpu_threads')}; память: {get('memory_limit_mb')} МиБ; срок: {seconds(get('timeout_seconds'))}.")
        if event == 'worker.model_ready':
            return f"Собрана модель для {eid}: {short(get('model_type'))}; основная модель: {short(get('estimator_type'))}."
        if event in {'job.progress', 'llm.waiting'}:
            elapsed = get('elapsed_seconds', 0)
            bucket = int(elapsed//60) if isinstance(elapsed, (int, float)) else 0
            key = (event, get('job_id') if event == 'job.progress' else get('call_no'))
            if bucket < 1 or bucket <= self.progress_buckets.get(key, 0):
                return None
            self.progress_buckets[key] = bucket
            who = 'Обучение '+eid if event == 'job.progress' else 'Ответ языковой модели'
            return f"{who}: прошло {seconds(elapsed)}, осталось всего {seconds(get('run_seconds_remaining'))}."
        if event == 'llm.call_started':
            purpose = {'task_interpretation': 'понимание условия', 'result_analysis': 'выбор проверки качества',
                       'research_proposal': 'исследовательское предложение'}.get(get('purpose'), short(get('purpose')))
            return f"Обращение к языковой модели №{get('call_no')}: {purpose}; срок ответа {seconds(get('request_timeout_seconds'))}."
        if event == 'experiment.evaluated':
            delta = r.get('improvement_over_baseline')
            comparison = f"; улучшение относительно исходного решения {number(delta)}" if delta is not None else ''
            direction = {'minimize': 'меньше — лучше', 'maximize': 'больше — лучше'}.get(get('direction'), '')
            return f"Оценён {eid}: {get('metric_id')} = {number(get('score'))} ({direction}){comparison}."
        if event == 'job.finished':
            if get('state') == 'succeeded':
                return None  # The independent evaluation follows; worker exit is not a score.
            error = get('error', {}) or {}
            return (f"Опыт {eid} завершился без результата: {get('state')}; {short(error)}. "
                    f"Подробности: {get('stderr_log_path')}.")
        if event == 'submission.published':
            return f"Ответ проверен и сохранён: {get('output_path')}; опыт {eid}, метрика {number(get('score'))}, версия публикации {get('champion_version')}."
        if event == 'submission.refit_deferred':
            return (f"Новый ответ отложен: прогноз окончательного обучения {seconds(get('estimated_seconds'))}, "
                    f"осталось {seconds(get('remaining_seconds'))}. Предыдущий проверенный ответ сохранён.")
        if event == 'search.proposal_replaced':
            return f"Предложение заменено ({get('work_order_id')}): {reason(get('reason'))}."
        if event == 'search.decision':
            action = get('action')
            key = (action, get('branch_id'), get('reason'))
            if action == 'CONTINUE' and (get('reason') == 'Round-robin work on equal peers' or key == self.last_decision):
                return None
            self.last_decision = key
            action_name = {'CONTINUE': 'продолжить', 'BACKTRACK': 'вернуться к альтернативе', 'FORK': 'разделить ветвь',
                           'REFINE_BEST': 'довести лучшее решение', 'FINALIZE': 'завершить поиск',
                           'TRANSFER_COMPONENT': 'проверить перенос приёма'}.get(action, action)
            return f"Решение: {action_name}, ветвь {get('branch_id')}. Основание: {reason(get('reason'))}; {short(get('parameters'))}."
        if event == 'graph.branch_forked':
            return f"Ветвь {get('parent_branch_id')} разделена: с приёмом {get('accept_branch_id')}, без приёма {get('reject_branch_id')}. Идея: {get('idea_id')}."
        if event == 'graph.branch_state_changed':
            state = {'active': 'активна', 'paused': 'приостановлена', 'archived': 'в архиве'}.get(get('state'), get('state'))
            return f"Состояние ветви {get('branch_id')}: {state}. Основание: {reason(get('reason'))}."
        if event == 'search.phase_changed':
            return f"Этап: доводка измеренного лучшего {get('source_experiment_id')}; осталось {seconds(get('remaining_seconds'))}. Основание: {reason(get('reason'))}."
        if event == 'search.finished':
            budget = get('budget', {}) or {}
            return (f"Поиск завершён: {reason(get('stop_reason'))}. Осталось {seconds(budget.get('wall_seconds_remaining'))}; "
                    f"запас {seconds(budget.get('finalization_reserve_seconds'))}.")
        if event == 'run.finished':
            state = {'completed': 'завершён', 'degraded': 'завершён с отдельными ошибками'}.get(get('state'), get('state'))
            return f"Запуск {state}. Сохранённый ответ: {get('output_path')}; {get('metric_id')} = {number(get('score'))}. Журнал: {get('run_directory')}."
        if r.get('level') in {'WARNING', 'ERROR'}:
            identity = f", опыт {eid}" if 'experiment_id' in r else ''
            code = f" [{r['code']}]" if r.get('code') else ''
            return f"{'Ошибка' if r['level'] == 'ERROR' else 'Предупреждение'} {event}{identity}{code}: {short(get('message', get('reason', get('error', 'см. подробный журнал'))))}."
        return None
