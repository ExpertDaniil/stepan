"""Dispatcher-owned, branch-visible archive of measured feature representations."""
from copy import deepcopy
import hashlib

from agent.contracts import canonical
from agent.feature_schema import FACTORY_SOURCE


def archive(store, artifacts, experiment_ids, limit=4):
    representations = {}
    for eid in dict.fromkeys(experiment_ids):
        exp = store.get(eid)
        plan = exp['plan']
        if plan['recipe_id'] != 'python-pipeline-v1':
            continue
        ev = store.experiment_evaluation(eid)
        if exp['state'] != 'evaluated' or not ev or not ev['valid'] or ev['violations']:
            continue
        params = plan['model']['hyperparameters']
        code = artifacts.meta(plan['code_artifact_id'])['sha256']
        program = params.get('feature_program')
        identity = canonical([code, program, plan['feature_steps']])
        score = ev['primary_metric']['value']
        record = {'representation_id': 'features-'+hashlib.sha256(identity.encode()).hexdigest()[:20],
            'experiment_id': eid, 'validation_id': plan['validation_id'],
            'primary_metric': deepcopy(ev['primary_metric']), 'code_artifact_id': plan['code_artifact_id'],
            'feature_steps': deepcopy(plan['feature_steps']), 'parameters': deepcopy(params),
            'implementation': ('owned_program' if program and code == hashlib.sha256(FACTORY_SOURCE.encode()).hexdigest()
                               else 'custom_source_declared_features'),
            'claim_scope': 'Complete configuration measured; feature/component causality not established.'}
        if record['implementation'] == 'owned_program':
            from agent.feature_schema import operation_facts
            record['execution_specification'] = operation_facts(program)
        else:
            record['feature_declaration_verified'] = False
        old = representations.get(identity)
        better = old is None or (score < old['primary_metric']['value'] if ev['primary_metric']['direction'] == 'minimize'
                                else score > old['primary_metric']['value'])
        if better:
            representations[identity] = record
    records = list(representations.values())
    records.sort(key=lambda x: x['primary_metric']['value']*(1 if x['primary_metric']['direction'] == 'minimize' else -1))
    # Include a recent alternative as well as best representations; no causal diversity claim.
    selected = records[:max(0, limit-1)]
    if records:
        recent = next(reversed(representations.values()))
        if recent not in selected:
            selected.append(recent)
        elif len(records) >= limit:
            selected.append(records[limit-1])
    omitted_details = 0
    for record in selected:
        if len(canonical(record).encode()) > 6000:
            record.pop('parameters')
            if len(canonical(record).encode()) > 6000:
                record['feature_steps'] = [{'step': str(s.get('step', 'custom'))[:100]}
                                           for s in record['feature_steps'][:8]]
            record['details'] = 'Exact program/model in the frozen execution artifact for experiment_id.'
            omitted_details += 1
    return {'representations': selected, 'omitted_representation_count': len(records)-len(selected),
        'omitted_parameter_details': omitted_details,
        'instruction': 'Evolve executable representations. Reuse actual programs, compare compatible measurements, '
            'and retain alternatives. Catalogue target transforms alone do not constitute this archive.'}


def interface():
    return {'action': 'features', 'compiler': 'owned inductive fit/transform pipeline',
        'operations': {
            'rowwise': ['missing', 'log1p', 'signed_log1p', 'square', 'sum', 'difference', 'product', 'ratio'],
            'categorical': ['frequency', 'cross'],
            'grouped': ['group_mean', 'group_std', 'group_rank', 'relative_to_group'],
            'datetime': ['month', 'dayofweek', 'hour']},
        'geometry': 'cyclic_sin/cyclic_cos require an explicit positive period and optional origin, in the input scale.',
        'semantics': 'Named operations execute in order, may consume earlier generated columns, and preserve originals. '
            'Group statistics/frequencies/rank reference distributions fit on fit rows only, never on prediction batches. '
            'Unknown groups yield missing, unseen frequencies zero, division by zero yields missing. '
            'Choose a question/mechanism from semantic_task before choosing an operator. '
            'Use named inputs; column order is not relevance. Each features action supplies an investigation covering its raw dependencies.',
        'output': 'action=features; recipe_id is an owned learned base; hyperparameters are that base parameters; '
            'feature_program=[{op,name,columns, optional group_by}]; code/tuning=null; feature_steps=[]. '
            'Use action=pipeline for transformations outside this language; custom tuning remains available.',
        'detail_hint': 'Use read_memory section=tool_guide for examples. Select actual inputs from semantic_task and analysis, not from a default example program.'}
