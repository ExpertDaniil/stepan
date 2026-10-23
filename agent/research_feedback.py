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
    # Prefer semantic evidence to old explanations. Never silently truncate IDs.
    for field in ('decisions', 'open_ideas', 'semantic_records'):
        while len(canonical(result).encode()) > max_bytes and result[field]:
            result[field].pop()
            counter = {'decisions': 'omitted_decision_count', 'open_ideas': 'omitted_idea_count',
                       'semantic_records': 'omitted_semantic_count'}[field]
            result[counter] += 1
    return result


def compact_updates(value):
    """Keep recent measurements/activity; exact updates can be read by their ID."""
    result = deepcopy(value)
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
