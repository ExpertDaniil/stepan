"""Dispatcher-built research orientation from comparable measured configurations.

This is an agenda and a leaderboard, not a semantic judge of model reasoning.
It never reads rows/labels or promotes model-authored conclusions to evidence.
"""
from copy import deepcopy

from agent.contracts import canonical


def build_landscape(store, artifacts, experiment_ids):
    measured, constants = [], []
    reference = None
    for eid in dict.fromkeys(experiment_ids):
        exp = store.get(eid)
        ev = store.experiment_evaluation(eid)
        if exp['state'] != 'evaluated' or not ev or not ev['valid'] or ev['violations']:
            continue
        metric = ev['primary_metric']
        scope = tuple(metric[k] for k in ('dataset_id', 'validation_id', 'fidelity', 'metric_id', 'direction'))
        if reference is None:
            reference = scope
        if scope != reference:
            continue
        plan = exp['plan']
        parameters = plan['model']['hyperparameters']
        record = {'experiment_id': eid, 'recipe_id': plan['recipe_id'], 'family': plan['model']['family'],
                  'primary_metric': deepcopy(metric), 'code_artifact_id': plan['code_artifact_id'],
                  'parameters': deepcopy(parameters), 'feature_steps': deepcopy(plan['feature_steps'])}
        if len(canonical(record).encode()) > 3000:
            record.pop('parameters')
            record['feature_steps'] = [{'step': str(s.get('step', 'custom'))[:100]} for s in plan['feature_steps'][:6]]
            record['details'] = 'Full source/parameters are frozen under experiment_id.'
        (constants if plan['model']['family'] == 'dummy' else measured).append(record)
    direction = reference[-1] if reference else 'minimize'
    sign = 1 if direction == 'minimize' else -1
    measured.sort(key=lambda r: sign*r['primary_metric']['value'])
    families = {}
    for record in measured:
        families.setdefault(record['family'], record)
    programs = sum(bool(store.get(r['experiment_id'])['plan']['model']['hyperparameters'].get('feature_program'))
                   for r in measured)
    return {'scope': 'visible_lineage_same_selection_protocol',
            'measured_learned_configurations': len(measured), 'constant_references': len(constants),
            'owned_feature_program_evaluations': programs,
            'strongest_measured_learned': measured[0] if measured else None,
            'best_per_family': list(families.values())[:6],
            'omitted_families': max(0, len(families)-6),
            'constant_reference_policy': 'Constant mean/majority is an emergency reference, not a tuning direction. '
                'Research catalog excludes Dummy. A custom constant-like result is diagnosed from predictions, not source guesses.',
            'next_question': ('First establish an affordable learned model before optimizing an intercept.' if not measured else
                'Use the strongest measured learned result as a comparison, even if your own head is weaker. '
                'Explain why your next representation/model could improve its errors or answer an unresolved data question.'),
            'interpretation': 'Scores measure complete configurations. Custom family groups implementations; '
                'finite trials do not exhaust a family or prove a feature mechanism.'}
