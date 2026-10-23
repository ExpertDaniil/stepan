"""Dispatcher-owned activity publication and admission-time overlap review."""
from copy import deepcopy


def domain_overlap(left, right):
    """Intersection / smaller domain, an overlap heuristic rather than a probability."""
    if left['type'] != right['type']:
        return 0.0
    if left['type'] == 'categorical':
        # Keep JSON type identity: bool is not an integer category.
        from agent.contracts import canonical
        a, b = ({canonical(x) for x in d['choices']} for d in (left, right))
        return len(a & b)/max(1, min(len(a), len(b)))
    lo, hi = max(left['low'], right['low']), min(left['high'], right['high'])
    if hi < lo:
        return 0.0
    width = min(left['high']-left['low'], right['high']-right['low'])
    if width == 0:
        return 1.0
    if left['type'] == 'int':
        # Different grids are deliberately not conflated.
        if left.get('step', 1) != right.get('step', 1):
            return 0.0
        step = left.get('step', 1)
        if (left['low']-right['low']) % step:
            return 0.0
        return (max(0, (hi-lo)//step)+1)/(width//step+1)
    if left.get('step') != right.get('step') or left.get('log', False) != right.get('log', False):
        return 0.0
    return (hi-lo)/width


def overlapping_spaces(a, b):
    # Distinct axes remain distinct questions. This would not merge the historical
    # six-axis HGB search with the five-axis HGB search that actually improved it.
    return bool(a and set(a) == set(b) and all(domain_overlap(a[k], b[k]) >= .8 for k in a))


class ResearchCoordination:
    def __init__(self, controller):
        self.c = controller
        self.reasoning = {}

    def dispatched(self, ticket):
        work = ticket['work_order']
        self.reasoning[work['work_order_id']] = {k: work[k] for k in
            ('work_order_id', 'researcher_id', 'branch_id', 'base_experiment_id')}

    def released(self, ticket):
        self.reasoning.pop(ticket['work_order']['work_order_id'], None)

    def snapshot(self, branch):
        optimizer = self.c.local_search.optuna
        active = []
        if optimizer is not None:
            for (scope, peer), session in optimizer.active.items():
                if scope == branch and session['status'] == 'running':
                    active.append({'series_id': session['series_id'], 'study_id': session['study_id'],
                        'researcher_id': peer, 'source_plan_id': session['source_plan_id'],
                        'recipe_id': session['anchor']['recipe_id'], 'search_space': deepcopy(session['space']),
                        'fixed_parameters': deepcopy(session['fixed_parameters']),
                        'proposed_trials': len(session['trials']),
                        'completed_trials': sum(t['finished'] and t.get('primary_metric') is not None for t in session['trials'])})
        return {'reasoning': [deepcopy(v) for v in self.reasoning.values() if v['branch_id'] == branch],
                'active_searches': active,
                'instruction': 'Pending searches have no assumed score. Coordinate complementary work. '
                    'A materially overlapping concurrent tune requires coordination.related_series_ids '
                    'and a concrete reason after observing that series in feedback.'}

    def review(self, plan, intent, branch):
        optimizer = self.c.local_search.optuna
        if intent['action'] != 'tune' or optimizer is None:
            return None
        from agent.tuning_space import normalize_space
        restrictions = self.c.graph.branch(branch)['constraints'] if self.c.graph else None
        space = normalize_space(self.c.catalog, plan, intent['tuning'], restrictions)
        proposed_scope = optimizer._scope(plan, space, branch)
        observed = {sid for d in intent.get('delivered_feedback', []) for sid in d.get('active_series_ids', [])}
        acknowledgement = intent.get('coordination', {})
        for (scope, peer), session in list(optimizer.active.items()):
            if scope != branch or peer == plan['researcher_id'] or not optimizer.has_series(peer, branch):
                continue
            if proposed_scope['pipeline'] != session['scope']['pipeline'] or not overlapping_spaces(space, session['space']):
                continue
            sid = session['series_id']
            if sid in observed and sid in acknowledgement.get('related_series_ids', []) and acknowledgement.get('reason', '').strip():
                self.c.log.event('research.overlap_acknowledged', series_id=sid, plan_id=plan['plan_id'],
                                 reason=acknowledgement['reason'])
                continue
            return {'series_id': sid, 'researcher_id': peer, 'search_space': session['space'],
                    'proposed_trials': len(session['trials']),
                    'completed_trials': sum(t['finished'] for t in session['trials']),
                    'observed_before_response': sid in observed,
                    'reason': 'Materially overlapping concurrent search lacks an informed explicit justification.'}
        return None
