"""One final comparison on labels withheld from adaptive scores and diagnostics.

All trials predict the entire frozen validation partition. Research sees scores
only on selection_rows. Confirmation reuses saved predictions after API dispatch
is closed, so it needs no extra fitting and must never guide subsequent search.
"""
from copy import deepcopy
import json
import math

from agent.artifacts import file_sha256
from agent.errors import AgentError
from agent.predictions import checked_matrix, read_training, targets, validation_columns
from agent.store import comparable


def partition(frame, valid, proposal, spec):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    if len(valid) < 40:
        return list(valid), [], 'fewer_than_40_validation_rows'
    strategy = proposal['strategy']
    seed = (proposal['seed']+104729) % (2**32-1)
    try:
        if strategy == 'temporal_holdout':
            column = proposal['parameters']['column']
            values = frame.loc[valid, column]
            if not pd.api.types.is_numeric_dtype(values):
                values = pd.to_datetime(values, format='ISO8601', utc=True, errors='raise')
            periods = sorted(values.unique().tolist())
            if len(periods) < 2:
                return list(valid), [], 'one_validation_period'
            cutoff = periods[max(1, int(len(periods)*.75))]
            a = [i for i in valid if values.loc[i] < cutoff]
            b = [i for i in valid if values.loc[i] >= cutoff]
        elif strategy == 'group_holdout':
            groups = frame.loc[valid, proposal['parameters']['column']].fillna('__MISSING__').astype(str)
            left, right = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed).split(valid, groups=groups))
            a, b = np.asarray(valid)[left].tolist(), np.asarray(valid)[right].tolist()
        else:
            stratify = frame.loc[valid, spec['target_column']].astype(str) if spec['task_kind'] != 'regression' else None
            a, b = train_test_split(valid, test_size=.25, random_state=seed, stratify=stratify)
        if not a or not b:
            return list(valid), [], 'empty_confirmation_partition'
        return sorted(map(int, a)), sorted(map(int, b)), None
    except (ValueError, TypeError) as error:
        return list(valid), [], 'partition_unavailable:'+type(error).__name__


def finalize(controller):
    if getattr(controller, 'confirmation_attempted', False):
        return
    controller.confirmation_attempted = True
    controller.researcher.gateway.cancelled.set()
    controller.budget.search_stopped = True
    protocol = controller.evaluator.protocols[controller.validation['validation_id']]
    indices = protocol.get('confirmation_indices', [])
    if not indices:
        controller.log.event('validation.confirmation_skipped', reason=protocol.get('confirmation_unavailable_reason', 'disabled'))
        return
    # Confirmation is paid from its explicit reserve, inside the finalization
    # budget. Requiring additional time beyond the whole reserve skipped it at
    # precisely the normal search/finalization boundary.
    if controller.budget.remaining() <= controller.budget.output_reserve()+1:
        controller.log.event('validation.confirmation_skipped', reason='finalization_time_reserve')
        return
    best = controller.final_candidate()
    if best is None:
        return
    reference = controller.store.experiment_evaluation(best)['primary_metric']
    candidates = []
    for row in controller.store.db.execute("SELECT experiment_id FROM experiments WHERE run_id=? AND state='evaluated'", (controller.store.run_id,)):
        eid = row[0]
        ev = controller.store.experiment_evaluation(eid)
        if eid not in controller.failed_refits and ev and ev['valid'] and not ev['violations'] and comparable(reference, ev['primary_metric']):
            champion = controller.store.champion()
            ready = champion and champion['experiment_id'] == eid
            cost = controller.budget.duration.estimate(controller.store.get(eid)['plan'], 'refit')+controller.budget.output_reserve()
            if not ready and cost > controller.budget.remaining():continue
            candidates.append((eid, ev['primary_metric']['value']))
    candidates.sort(key=lambda pair: pair[1], reverse=reference['direction'] == 'maximize')
    finalists = [eid for eid, _ in candidates[:2]]
    champion = controller.store.champion()
    if champion and champion['experiment_id'] not in finalists:
        finalists = [best, champion['experiment_id']]
    if len(finalists) < 2:
        controller.log.event('validation.confirmation_skipped', reason='fewer_than_two_finalists')
        return
    from agent.evaluation import score_predictions
    import pandas as pd
    import numpy as np
    from agent.data_transport import transport
    data = transport(controller.artifacts).read(controller.spec['train_artifact_id'], controller.spec,
                                              columns=targets(controller.spec), rows=indices)
    names = targets(controller.spec)
    truth = data[names if len(names) > 1 else names[0]]
    columns = validation_columns(controller.spec)
    kind = controller.spec['metric']['prediction_kind']
    scores = []
    for eid in finalists:
        prediction_id = None
        for row in controller.store.db.execute('SELECT record FROM job_events ORDER BY rowid DESC'):
            event = json.loads(row[0])
            if event['experiment_id'] != eid or event['state'] != 'succeeded':
                continue
            prediction_id = next((a for a in event['output_artifact_ids']
                if controller.artifacts.meta(a)['kind'] == 'validation_predictions'), None)
            if prediction_id:
                break
        if not prediction_id:
            raise AgentError('CONFIRMATION_UNAVAILABLE', 'Saved validation predictions are unavailable for '+eid)
        path = controller.artifacts.resolve(prediction_id)
        if file_sha256(path) != controller.artifacts.meta(prediction_id)['sha256']:
            raise AgentError('CANDIDATE_CHANGED', 'Confirmation predictions changed after evaluation')
        frame = pd.read_csv(path, dtype={c: str for c in columns} if kind == 'labels' else None, keep_default_na=False)
        if list(frame.columns) != ['__row_index', *columns] or frame['__row_index'].tolist() != protocol['valid_indices']:
            raise AgentError('INVALID_PREDICTIONS', 'Confirmation row order/columns changed')
        values = frame.set_index('__row_index').loc[indices, columns].to_numpy()
        matrix = checked_matrix(values, len(indices), len(columns), kind, labels=controller.spec['allowed_labels'])
        value = float(score_predictions(controller.spec, truth, matrix))
        if not math.isfinite(value):
            raise AgentError('METRIC_FAILED', 'Confirmation score is not finite')
        scores.append({'experiment_id': eid, 'prediction_artifact_id': prediction_id, 'value': value})
    reverse = reference['direction'] == 'maximize'
    # Ties retain primary ranking. No arbitrary epsilon or significance claim.
    chosen = sorted(scores, key=lambda s: s['value'], reverse=reverse)[0]['experiment_id']
    report = {'validation_id': controller.validation['validation_id'], 'rows': len(indices),
        'metric_id': reference['metric_id'], 'direction': reference['direction'], 'finalists': scores,
        'chosen_experiment_id': chosen, 'prior_champion_experiment_id': champion['experiment_id'] if champion else None,
        'expected_champion_version': champion['champion_version'] if champion else 0,
        'limitations': 'One comparison of adaptively chosen finalists on one withheld partition. '
            'No repeated CV, causal attribution, statistical significance or unlimited independent tests are claimed. '
            'No further research requests are permitted after this comparison.'}
    art = controller.artifacts.write_json('validation/final-confirmation.json', report, 'final_confirmation')
    controller.submission.final_confirmation = {'artifact_id': art['artifact_id'], **deepcopy(report)}
    controller.confirmed_candidate = chosen
    controller.log.event('validation.finalists_confirmed', confirmation_artifact_id=art['artifact_id'],
        chosen_experiment_id=chosen, finalists=scores, rows=len(indices), further_research_allowed=False)
