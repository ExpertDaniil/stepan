"""Dispatcher-published feedback; API threads never read SQLite or expand lineage."""
from copy import deepcopy
import hashlib
import threading

from agent.contracts import canonical


class FeedbackMailbox:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.lock = threading.RLock()
        self.views = {}

    def publish(self, scope, value):
        digest = hashlib.sha256(canonical(value).encode()).hexdigest()
        with self.lock:
            old = self.views.get(scope)
            if old and old['revision'] == digest:
                return old
            art = self.artifacts.write_json('research/updates-'+digest+'.json', value, 'research_updates')
            view = {'revision': digest, 'artifact_id': art['artifact_id'], **deepcopy(value)}
            self.views[scope] = view
            return deepcopy(view)

    def latest(self, scope):
        with self.lock:
            return deepcopy(self.views.get(scope))


def compact_memory(memory, *, base_experiment_id=None, question='', max_bytes=12000):
    """Bound the entire projection; full immutable documents live in read_memory.

    Rank by base linkage, question-word overlap, then recency. This transparent
    heuristic does not claim semantic retrieval or proof of an interpretation.
    """
    import re
    words = set(re.findall(r'\w+', question.casefold()))
    decisions = memory.get('decisions', [])
    if not words and decisions:
        words = set(re.findall(r'\w+', decisions[-1].get('question', '').casefold()))
    def rank(pair):
        index, record = pair
        related = base_experiment_id is not None and any(
            o.get('experiment_id') == base_experiment_id for o in record.get('observations', []))
        overlap = len(words & set(re.findall(r'\w+', canonical(record).casefold())))
        return (related, overlap, index)
    chosen = [r for _, r in sorted(enumerate(decisions), key=rank, reverse=True)[:4]]
    result = {'instruction': 'Short projection only. Read detail_index with read_memory for exact history. '
        'Model statements are unverified; complete configuration metrics do not establish component effects.',
        'decisions': [], 'open_ideas': [], 'semantic_records': [],
        'omitted_decision_count': memory.get('omitted_decision_count', 0)+len(decisions)-len(chosen),
        'omitted_idea_count': memory.get('omitted_idea_count', 0)+max(0, len(memory.get('open_ideas', []))-8),
        'omitted_semantic_count': memory.get('omitted_semantic_count', 0)+max(0, len(memory.get('semantic_records', []))-8)}
    for r in chosen:
        result['decisions'].append({
            'decision_artifact_id': r['decision_artifact_id'], 'action': r['action'],
            'idea_id': r.get('idea_id'), 'question': r.get('question', '')[:320],
            'decision_reason': r.get('decision_reason', '')[:320],
            'model_interpretations_verified': False,
            'investigation': deepcopy(r.get('investigation')),
            'evidence_ids': r.get('evidence_ids', [])[:12],
            'observations': deepcopy(r.get('observations', [])[-3:]),
            'tuning_series': [{k: deepcopy(v) for k, v in series.items() if k in {
                'series_id', 'artifact_id', 'status', 'stop_reason', 'best_metric', 'best_experiment_id',
                'completed_trials', 'elapsed_seconds'}} for series in r.get('tuning_series', [])[-2:]]})
    for r in sorted(memory.get('open_ideas', []), key=lambda r: (
            len(words & set(re.findall(r'\w+', r.get('question', '').casefold()))),
            r.get('status') in {'running', 'measured'}), reverse=True)[:8]:
        result['open_ideas'].append({k: (v[:320] if isinstance(v, str) else deepcopy(v[:8]) if isinstance(v, list) else v)
                                    for k, v in r.items() if k in {'idea_id', 'question', 'mechanism', 'status',
                                        'evidence_ids', 'knowledge_ids', 'next_test', 'why_deferred'}})
    semantics = memory.get('semantic_records', [])
    selected = sorted(enumerate(semantics), key=lambda p: (
        p[1].get('status') == 'conflicted',
        len(words & set(re.findall(r'\w+', canonical(p[1]).casefold()))), p[0]), reverse=True)[:8]
    result['semantic_records'] = [deepcopy(r) for _, r in selected]
    by_id = {r['record_id']: r for r in semantics}
    conflicts = []
    for r in reversed(semantics):
        other = by_id.get(r.get('conflicts_with'))
        if not other:
            continue
        conflicts.append({'topic': r['content']['topic'], 'status': 'unresolved_contradiction',
            'alternatives': [{'record_id': v['record_id'], 'kind': v['content']['kind'],
                'statement': v['content']['statement'][:160], 'source_decision_artifact_id': v['source_decision_artifact_id'],
                'verification': v['verification']} for v in (other, r)]})
    result['conflicts'] = conflicts[:4]
    result['omitted_conflict_count'] = len(conflicts)-len(result['conflicts'])
    # Prefer semantic evidence to old explanations. Never silently truncate IDs.
    for field in ('decisions', 'open_ideas', 'semantic_records'):
        while len(canonical(result).encode()) > max_bytes and result[field]:
            result[field].pop()
            counter = {'decisions': 'omitted_decision_count', 'open_ideas': 'omitted_idea_count',
                       'semantic_records': 'omitted_semantic_count'}[field]
            result[counter] += 1
    while len(canonical(result).encode()) > max_bytes and result['conflicts']:
        result['conflicts'].pop()
        result['omitted_conflict_count'] += 1
    return result


def compact_updates(value):
    """Keep recent measurements/activity; exact updates can be read by their ID."""
    result = deepcopy({k: v for k, v in value.items() if k != 'solution_catalog'})
    from agent.fit_diagnostics import compact_report
    for record in result.get('fit_diagnostics', []):
        record['report'] = compact_report(record.get('report', {}))
    # Source/parameter documents are read by the gateway from the authorized
    # immutable snapshot, never copied into every API request.
    result.pop('solution_catalog', None)
    original_count = len(result.get('results', []))
    result['results'] = result.get('results', [])[-4:]
    result['omitted_result_count'] = result.get('omitted_result_count', 0)+original_count-len(result['results'])
    for row in result['results']:
        row.pop('effective_parameters', None)
    for series in result.get('series', []):
        series.pop('search_space', None)
    for active in result.get('activity', {}).get('active_searches', []):
        active.pop('fixed_parameters', None)
    comparisons = result.pop('comparison_evidence', {})
    result['comparison_details'] = {'omitted_comparisons': len(comparisons.get('comparisons', [])),
                                    'reference_id': value['artifact_id'], 'section': 'updates'}
    result['details'] = {'reference_id': value['artifact_id'], 'section': 'updates', 'tool': 'read_memory'}
    for field in ('admission_feedback', 'series', 'pending_experiments'):
        while len(canonical(result).encode()) > 12000 and len(result.get(field, [])) > 1:
            result[field].pop(0)
            result['details_omitted'] = True
    return result


def frame_updates(view, context):
    """Hold one dialogue's question and research base fixed while updates arrive.

    Background trials that change neither the best solution nor this branch's
    controller feedback leave the previous per-turn view unchanged; they stay
    readable through ``details``. Two references are named separately: the
    research base (effect of the change) and the current best (utility).
    """
    import json
    view = deepcopy(view)
    # Background trial IDs/states churn after every fit. They stay in the exact
    # mailbox details but cannot by themselves refresh the researcher's feed.
    view['pending_experiments'] = [r for r in view.get('pending_experiments', [])
                                   if r.get('execution_kind') != 'background_tuning']
    frame = context.get('_research_frame')
    if frame is None:
        user = json.loads(context['messages'][1]['content'])
        projection = user.get('profile_summary', user)
        advice = projection.get('error_research') or {}
        recommendation = advice.get('recommendation') or {}
        leader = advice.get('leader') or {}
        base = recommendation.get('base_experiment_id') or (user.get('base_result') or {}).get('experiment_id')
        frame = {'question': recommendation.get('question'), 'research_base_experiment_id': base,
                 'research_base_metric': (leader.get('metric') if leader.get('experiment_id') == base else
                                          (user.get('base_result') or {}).get('metric'))}
        context['_research_frame'] = frame
    best = view.get('best_solution') or {}
    relevant = canonical([best.get('experiment_id'), best.get('metric'), view.get('admission_feedback', []),
                          view.get('candidate_results', []),
                          view.get('pending_experiments', []), [r.get('artifact_id') for r in view.get('fit_diagnostics', [])],
                          [{k: s.get(k) for k in ('series_id', 'status', 'best_experiment_id')} for s in view.get('series', [])]])
    previous = context.get('_feedback_turn')
    if previous and previous['relevant'] == relevant:
        result = deepcopy(previous['view'])
        result['changed_since_previous_turn'] = False
        result['changes'] = []
        return result
    changes = []
    if previous:
        old = previous['best']
        if old.get('experiment_id') != best.get('experiment_id'):
            changes.append({'kind': 'best_solution', 'from': {k: old.get(k) for k in ('experiment_id', 'metric')},
                            'to': {k: best.get(k) for k in ('experiment_id', 'metric')}})
        seen = {canonical(r) for r in previous['view'].get('admission_feedback', [])}
        changes.extend({'kind': 'controller_feedback', 'status': r.get('status'), 'code': r.get('code')}
                       for r in view.get('admission_feedback', []) if canonical(r) not in seen)
        for field, kind in (('candidate_results', 'candidate_result'), ('pending_experiments', 'pending_research'),
                            ('fit_diagnostics', 'fit_diagnostics'), ('series', 'tuning_series')):
            if canonical(previous['view'].get(field, [])) != canonical(view.get(field, [])):
                changes.append({'kind': kind})
    result = deepcopy(view)
    result['research_frame'] = {**frame,
        'utility_reference': {k: best.get(k) for k in ('experiment_id', 'recipe_id', 'metric')} if best else None,
        'instruction': 'Your question and research base stay fixed for this dialogue. Judge a change by its effect against '
            'the research base and its utility against the current best. A better tuned best changes only the utility '
            'reference; change direction only when new measurements contradict your hypothesis.'}
    result['changed_since_previous_turn'] = True
    result['changes'] = changes
    context['_feedback_turn'] = {'relevant': relevant, 'view': deepcopy(result), 'best': deepcopy(best)}
    return result
