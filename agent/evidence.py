"""Measured comparison scopes; model prose never creates a scientific fact."""
from copy import deepcopy
import hashlib

from agent.contracts import canonical
from agent.errors import AgentError


def _measurement(store, eid):
    exp = store.get(eid)
    ev = store.experiment_evaluation(eid)
    if exp['state'] != 'evaluated' or not ev or not ev['valid'] or ev['violations']:
        raise ValueError('Experiment has no trusted completed evaluation: '+eid)
    return exp['plan'], ev['primary_metric']


def _changes(a, b, prefix='parameter'):
    result = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            result.append(prefix+':'+key)
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            result.extend(_changes(a[key], b[key], prefix+':'+key))
        elif canonical(a[key]) != canonical(b[key]):
            result.append(prefix+':'+key)
    return result


def compare(store, artifacts, left_id, right_id):
    result = {'experiment_ids': [left_id, right_id], 'comparable': False,
              'supported_scope': 'unverified', 'component_effect_identified': False}
    try:
        left, lm = _measurement(store, left_id)
        right, rm = _measurement(store, right_id)
    except (ValueError, AgentError) as error:
        return {**result, 'reason': str(error)}
    keys = ('dataset_id', 'validation_id', 'fidelity', 'metric_id', 'direction')
    if any(lm[k] != rm[k] for k in keys) or any(left[k] != right[k] for k in ('dataset_id', 'task_spec_id', 'validation_id', 'seed', 'fidelity')):
        return {**result, 'reason': 'Different data/task/metric/split/seed/fidelity.'}
    changes = _changes(left['model']['hyperparameters'], right['model']['hyperparameters'])
    source_changed = artifacts.meta(left['code_artifact_id'])['sha256'] != artifacts.meta(right['code_artifact_id'])['sha256']
    family_changed = (left['recipe_id'], left['model']['family']) != (right['recipe_id'], right['model']['family'])
    manifest_changed = canonical(left['feature_steps']) != canonical(right['feature_steps'])
    # Comparing source strings alone cannot establish identical runtime dependencies.
    execution_ids = ['execution-'+eid for eid in (left_id, right_id)]
    runtime_equal = False
    if all(eid in artifacts.entries for eid in execution_ids):
        executions = [artifacts.read_json(eid) for eid in execution_ids]
        runtime_equal = bool(executions[0].get('runtime_sha256')) and executions[0]['runtime_sha256'] == executions[1].get('runtime_sha256')
    from agent.catalog import factory_source
    owned = (not left['recipe_id'].startswith('python-') and not family_changed and
             artifacts.meta(left['code_artifact_id'])['sha256'] == hashlib.sha256(factory_source(left['recipe_id']).encode()).hexdigest())
    # A custom manifest may be inaccurate. A changed feature source cannot be isolated
    # by counting its declared feature steps; retain complete-configuration scope.
    controlled = runtime_equal and owned and not source_changed and not family_changed and len(changes) == 1
    improvement = (lm['value']-rm['value']) * (1 if lm['direction'] == 'minimize' else -1)
    return {**result, 'comparable': True, 'metric_id': lm['metric_id'], 'direction': lm['direction'],
        'scores': [lm['value'], rm['value']], 'right_improvement': improvement,
        'changed_parameters': changes, 'source_changed': source_changed, 'family_changed': family_changed,
        'feature_manifest_changed': manifest_changed, 'frozen_runtime_equal': runtime_equal,
        'supported_scope': 'single_parameter_contrast' if controlled else 'complete_configuration_comparison',
        'contrast_parameter': changes[0] if controlled else None,
        'limitations': 'One adaptively used split; no statistical significance, global mechanism attribution '
                      'or model-family exhaustion follows from this comparison.'}


def assess_claims(store, artifacts, claims, visible_ids):
    assessments = []
    visible = set(visible_ids)
    for claim in claims:
        ids = list(dict.fromkeys(claim['evidence_ids']))
        assessment = {'model_claim': deepcopy(claim), 'status': 'unverified',
                      'model_statement_verified': False, 'authoritative_statement': None}
        if not set(ids) <= visible:
            assessment['reason'] = 'Evidence was not available in the assigned context/updates.'
        elif claim['scope'] == 'hypothesis':
            assessment['reason'] = 'Unmeasured hypothesis retained as a question.'
        elif claim['scope'] == 'family':
            assessment['reason'] = 'Finite local trials or a budget limit cannot establish a model-family ceiling.'
        elif len(ids) != 2:
            assessment['reason'] = 'A comparison requires two distinct trusted experiment IDs.'
        else:
            comparison = compare(store, artifacts, *ids)
            assessment['comparison'] = comparison
            if not comparison['comparable']:
                assessment['reason'] = comparison['reason']
            elif claim['scope'] == 'configuration':
                assessment.update(status='measured_configuration_comparison', authoritative_statement={
                    'scores': comparison['scores'], 'right_improvement': comparison['right_improvement']})
            elif (comparison['supported_scope'] == 'single_parameter_contrast'
                  and claim['component'] == comparison['contrast_parameter']):
                assessment.update(status='limited_parameter_contrast', authoritative_statement={
                    'parameter': comparison['contrast_parameter'], 'scores': comparison['scores'],
                    'scope': 'This exact fixed pipeline/protocol; not a general component mechanism.'})
            else:
                assessment['reason'] = 'Joint/unverified source or parameter changes do not identify the requested component effect.'
        assessments.append(assessment)
    return assessments


def comparison_context(store, artifacts, experiment_ids):
    measured = []
    for eid in dict.fromkeys(experiment_ids):
        try:
            plan, metric = _measurement(store, eid)
        except (AgentError, ValueError):
            continue
        measured.append((metric['value']*(1 if metric['direction'] == 'minimize' else -1), eid, plan['recipe_id']))
    measured.sort()
    if not measured:
        return {'comparisons': [], 'scope': 'No measured comparison available.'}
    best = measured[0][1]
    candidates = [eid for _, eid, _ in measured[1:3]]
    custom = next((eid for _, eid, recipe in measured if recipe == 'python-pipeline-v1' and eid != best), None)
    if custom and custom not in candidates:
        candidates.append(custom)
    return {'comparisons': [compare(store, artifacts, best, eid) for eid in candidates],
        'scope': 'Controller-computed differences of complete configurations. Free-form model explanations '
                 'remain unverified. A budget stop, marginal correlation or one joint regression cannot close '
                 'a component/family question. No mandatory one-change experiment sequence is imposed.'}
