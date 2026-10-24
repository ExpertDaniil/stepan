"""Budgeted out-of-fold error discovery confined to the trusted fit partition.

Diagnostic jobs use normal process/CPU/RSS/lease guards, never create candidate
scores, and never read selection/confirmation target rows. Reports are bounded.
"""
from copy import deepcopy
import json
from pathlib import Path
import time
import uuid

from agent.errors import AgentError


def inner_folds(frame, spec, proposal, seed):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
    strategy = proposal['strategy']
    if len(frame) < 30:
        raise ValueError('Too few fit rows for useful OOF diagnostics')
    if strategy == 'group_holdout':
        values = frame[proposal['parameters']['column']].fillna('__MISSING__').astype(str)
        count = min(3, values.nunique())
        if count < 2:
            raise ValueError('Too few distinct fit groups')
        splits = GroupKFold(count).split(frame, groups=values)
    elif strategy == 'temporal_holdout':
        values = frame[proposal['parameters']['column']]
        if not pd.api.types.is_numeric_dtype(values):
            values = pd.to_datetime(values, format='ISO8601', utc=True, errors='raise')
        periods = sorted(values.unique())
        count = min(3, len(periods)-1)
        if count < 2:
            raise ValueError('Too few distinct fit periods')
        splits = [(np.flatnonzero(values.isin([periods[i] for i in a])),
                   np.flatnonzero(values.isin([periods[i] for i in b])))
                  for a, b in TimeSeriesSplit(count).split(periods)]
    elif spec['task_kind'] != 'regression':
        y = frame[spec['target_column']].astype(str)
        count = min(3, int(y.value_counts().min()))
        if count < 2:
            raise ValueError('A class has too few fit examples for stratified OOF')
        splits = StratifiedKFold(count, shuffle=True, random_state=seed).split(frame, y)
    else:
        splits = KFold(3, shuffle=True, random_state=seed).split(frame)
    return [{'fit': list(map(int, a)), 'predict': list(map(int, b))} for a, b in splits]


def execute_oof(module, task, params, fit_x, fit_y, job, guard, limits):
    import numpy as np
    import pandas as pd
    from agent.ml import enforce_threads, validate_estimator, SequentialTargets
    from agent.predictions import predict, targets, validation_columns
    started = time.monotonic()
    records = []
    with limits:
        for number, split in enumerate(job['diagnostic']['folds']):
            model = validate_estimator(enforce_threads(module.make_pipeline(task, dict(params)), job['cpu_threads']))
            if len(targets(task)) > 1:
                model = SequentialTargets(model)
            model.fit(fit_x.iloc[split['fit']], fit_y.iloc[split['fit']])
            values = predict(model, fit_x.iloc[split['predict']], task, 'train')
            frame = pd.DataFrame(values, columns=validation_columns(task))
            frame.insert(0, '__row_index', fit_x.index.to_numpy()[split['predict']])
            frame.insert(1, '__fold', number)
            records.append(frame)
            print('AUTOML_WORKER '+json.dumps({'stage': 'oof_fold_completed', 'fold': number,
                'fit_rows': len(split['fit']), 'predicted_rows': len(frame)}), flush=True)
            del model
    if guard:
        guard.active = False
    output = pd.concat(records, ignore_index=True).sort_values('__row_index')
    if output['__row_index'].duplicated().any():
        raise ValueError('OOF rows must be predicted exactly once')
    output.to_csv(job['prediction_path'], index=False)
    return {'success': True, 'fit_seconds': time.monotonic()-started,
            'rows': len(output), 'diagnostic_only': True, 'folds': len(records),
            'sample_rows': len(fit_x), 'model_path': None}


def error_profile(data, truth, prediction, fold, task, *, deadline=None):
    """Scan eligible columns, rank supported patterns; no causal/underfit verdict."""
    import numpy as np
    import pandas as pd
    from agent.predictions import targets
    regression = task['task_kind'] == 'regression'
    y = np.asarray(truth, dtype=float if regression else str)
    p = np.asarray(prediction, dtype=float if regression else str)
    if regression:
        if y.ndim == 1:
            p = p.reshape(-1)
        residual = p-y
        signed = residual.mean(axis=1) if residual.ndim > 1 else residual
        error = np.abs(residual).mean(axis=1) if residual.ndim > 1 else np.abs(residual)
    else:
        signed = None
        error = (p.reshape(-1) != y.reshape(-1)).astype(float)
    baseline = float(error.mean())
    excluded = set(task['excluded_features']) | set(targets(task))
    eligible = [c for c in data if c not in excluded]
    groups, associations, scanned = [], [], 0
    minimum = max(10, int(len(error)*.02))
    fold = np.asarray(fold)
    for column in eligible:
        if deadline is not None and time.monotonic() >= deadline:
            break
        series = data[column]
        masks = [('missing', series.isna().to_numpy())]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
            cuts = sorted(set(numeric.dropna().quantile([.25, .5, .75]).tolist()))
            if cuts:
                buckets = np.searchsorted(cuts, numeric.fillna(0).to_numpy(), side='right')
                masks += [('quantile_'+str(i), (buckets == i) & numeric.notna().to_numpy()) for i in range(len(cuts)+1)]
            valid = numeric.notna().to_numpy()
            if valid.sum() >= minimum and numeric[valid].nunique() > 1:
                ranks = numeric[valid].rank().to_numpy()
                err_ranks = pd.Series(error[valid]).rank().to_numpy()
                correlation = np.corrcoef(ranks, err_ranks)[0, 1] if np.std(err_ranks) else 0.0
                associations.append({'feature': str(column)[:100], 'absolute_error_spearman': float(correlation)})
        else:
            values = series.dropna().astype(str)
            if values.nunique() > 1:
                actual = series.astype(str).to_numpy()
                masks += [(str(value)[:80], (actual == value) & series.notna().to_numpy())
                          for value in values.value_counts().index[:5]]
        scanned += 1
        for label, mask in masks:
            rows = int(mask.sum())
            if rows < minimum or rows == len(error):
                continue
            lift = float(error[mask].mean()-baseline)
            per_fold = [float(error[mask & (fold == f)].mean()-error[fold == f].mean())
                        for f in np.unique(fold) if (mask & (fold == f)).sum() >= 3]
            record = {'feature': str(column)[:100], 'group': label, 'rows': rows,
                      'mean_error': float(error[mask].mean()), 'error_delta': lift,
                      'folds_with_support': len(per_fold),
                      'folds_with_higher_error': sum(value > 0 for value in per_fold)}
            if signed is not None:
                record['prediction_bias'] = float(signed[mask].mean())
            groups.append(record)
    return {'scope': 'fit_only_inner_oof', 'predicted_rows': len(error), 'mean_error': baseline,
            'eligible_columns': len(eligible), 'scanned_columns': scanned,
            'omitted_columns': len(eligible)-scanned,
            'hard_groups': sorted(groups, key=lambda r: (-r['error_delta'], -r['rows']))[:8],
            'error_associations': sorted(associations, key=lambda r: -abs(r['absolute_error_spearman']))[:8],
            'interpretation': 'Exploratory associations on a bounded fit sample. Selection/confirmation labels were not used. '
                'High group error does not prove a cause, underfitting, or a useful new feature. Validate proposed remedies normally.'}


class FitDiagnostics:
    def __init__(self, controller):
        self.c = controller
        self.attempted = set()
        self.records = []
        self.spent = 0.0

    def visible(self, ids):
        return deepcopy([r for r in self.records if r['experiment_id'] in set(ids)][-2:])

    def ensure(self, ids):
        c = self.c
        cfg = c.config
        if (not cfg.research_oof_enabled or len(self.attempted) >= cfg.research_oof_jobs or c.runner.active()
                or not c.store.champion() or not c.budget.snapshot()['search_allowed']):
            return
        remaining = min(cfg.time_budget_seconds*cfg.research_oof_budget_fraction-self.spent,
                        c.budget.remaining()-c.budget.reserve_seconds()-2, c.budget.phase_remaining()-2)
        timeout = int(min(cfg.research_oof_timeout_seconds, remaining))
        if timeout < 5:
            return
        candidates = []
        for eid in ids:
            exp, ev = c.store.get(eid), c.store.experiment_evaluation(eid)
            if exp['state'] == 'evaluated' and ev and ev['valid'] and not ev['violations'] and exp['plan']['model']['family'] != 'dummy':
                m = ev['primary_metric']
                if m['validation_id'] == c.validation['validation_id']:
                    candidates.append((m['value']*(1 if m['direction'] == 'minimize' else -1), eid))
        if not candidates:
            return
        eid = min(candidates, key=lambda candidate: candidate[0])[1]
        if eid in self.attempted:
            return
        experiment = c.store.get(eid)
        estimate = 3*c.budget.duration.estimate(experiment['plan'])
        if estimate > timeout or not c.budget.can_train(experiment['plan'])['allowed']:
            return
        self.attempted.add(eid)
        started = time.monotonic()
        lease = job = None
        owner = 'fit-diagnostics-'+uuid.uuid4().hex[:12]
        resources = c.resources(timeout)
        try:
            import numpy as np
            import pandas as pd
            from agent.data_transport import transport
            from agent.predictions import targets, validation_columns, checked_matrix
            indices = c.artifacts.read_json(c.validation['train_rows_artifact_id'])
            rng = np.random.default_rng(cfg.seed)
            selected = sorted(rng.choice(indices, min(len(indices), cfg.research_oof_rows), replace=False).tolist())
            proposal = c.evaluator.protocols[c.validation['validation_id']]['proposal']
            columns = targets(c.spec)
            if proposal['strategy'] in {'group_holdout', 'temporal_holdout'}:
                columns = list(dict.fromkeys([*columns, proposal['parameters']['column']]))
            table = transport(c.artifacts).read(c.spec['train_artifact_id'], c.spec, columns=columns, rows=selected)
            folds = inner_folds(table, c.spec, proposal, cfg.seed)
            train_art = c.artifacts.write_json('diagnostics/'+owner+'.fit.json', selected, 'diagnostic_fit_rows')
            valid_art = c.artifacts.write_json('diagnostics/'+owner+'.unused.json', [], 'diagnostic_unused_rows')
            resources['timeout_seconds'] = min(timeout, int(remaining-(time.monotonic()-started)))
            if resources['timeout_seconds'] < 5:
                return
            allocation = c.lease(owner, 'train', resources)
            if not allocation['granted']:
                return
            lease = allocation['lease']
            payload = {'lease': lease, 'lease_owner_id': owner,
                       'train_rows_artifact_id': train_art['artifact_id'], 'valid_rows_artifact_id': valid_art['artifact_id'],
                       'diagnostic': {'folds': folds, 'scope': 'fit_only_inner_oof'}}
            job = c.runner.start_diagnostics(payload, c.spec, experiment)
            c.log.event('diagnostics.oof_started', experiment_id=eid, job_id=job.job_id,
                        sampled_fit_rows=len(selected), complete_fit_rows=len(indices), folds=len(folds), timeout_seconds=timeout)
            event = c.wait_jobs([job])[job.job_id]
            if event['state'] != 'succeeded':
                raise ValueError('Optional OOF worker ended '+event['state'])
            raw = pd.read_csv(job.prediction_path, dtype={n: str for n in validation_columns(c.spec)}
                              if c.spec['metric']['prediction_kind'] == 'labels' else None, keep_default_na=False)
            actual_rows = raw['__row_index'].tolist()
            expected = sorted(selected[position] for f in folds for position in f['predict'])
            if actual_rows != expected or len(set(actual_rows)) != len(actual_rows):
                raise ValueError('Diagnostic row coverage differs from trusted inner folds')
            all_data = transport(c.artifacts).read(c.spec['train_artifact_id'], c.spec, rows=actual_rows)
            names = targets(c.spec)
            y = all_data[names if len(names) > 1 else names[0]]
            kind = c.spec['metric']['prediction_kind']
            pred = checked_matrix(raw[validation_columns(c.spec)].to_numpy(), len(raw), len(validation_columns(c.spec)),
                                  kind, labels=c.spec['allowed_labels'])
            if kind == 'probabilities':
                pred = np.asarray(c.spec['allowed_labels'])[pred.argmax(axis=1)]
            report = error_profile(all_data, y, pred, raw['__fold'], c.spec,
                                   deadline=min(started+timeout, time.monotonic()+15))
            report.update(experiment_id=eid, validation_id=c.validation['validation_id'], sampled_fit_rows=len(selected),
                          complete_fit_rows=len(indices), sampled=len(selected) < len(indices),
                          folds=len(folds), split_strategy=proposal['strategy'],
                          warmup_rows_without_oof=len(selected)-len(raw), score_authoritative=False)
            art = c.artifacts.write_json('diagnostics/'+owner+'.json', report, 'fit_diagnostics')
            self.records.append({'artifact_id': art['artifact_id'], 'experiment_id': eid, 'report': report})
            c.log.event('diagnostics.oof_completed', artifact_id=art['artifact_id'], experiment_id=eid,
                        predicted_rows=len(raw), scanned_columns=report['scanned_columns'], primary_score_changed=False)
        except (AgentError, ValueError, OSError, KeyError) as error:
            c.log.event('diagnostics.oof_unavailable', experiment_id=eid, reason=str(error)[:600], retained_output=True)
        finally:
            if job and not job.reported:
                c.cancel_job(job, 'Diagnostic cleanup', 'cancelled')
                c.wait_jobs([job])
            elif lease and job is None:
                c.call('C13', {'operation': 'release', 'owner_id': owner, 'phase': 'train', 'requested': resources,
                       'lease_id': lease['lease_id'], 'fencing_token': lease['fencing_token']}, c.budget.lease)
            self.spent += time.monotonic()-started
