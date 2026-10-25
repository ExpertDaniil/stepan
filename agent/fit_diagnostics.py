"""Budgeted out-of-fold error discovery confined to the trusted fit partition.

Diagnostic jobs use normal process/CPU/RSS/lease guards, never create candidate
scores, and never read selection/confirmation target rows. Reports are bounded.
"""
from copy import deepcopy
import json
from pathlib import Path
import time
import uuid
import hashlib

from agent.errors import AgentError


def diverse_groups(groups, limit=12):
    """Mix measured cost, per-row severity and substantial rare concentrations."""
    cost = sorted(groups, key=lambda r: (-r['priority_cost'], -r['error_delta']))
    severity = sorted((r for r in groups if r['error_delta'] > 0),
                      key=lambda r: (-r['error_delta'], -r['priority_cost']))
    rare = sorted((r for r in groups if r['support_fraction'] <= .02
                   and r['priority_fraction'] >= .05), key=lambda r: -r['priority_cost'])
    chosen = []
    for rank in range(len(groups)):
        for sequence in (cost, severity, rare):
            if rank < len(sequence) and sequence[rank] not in chosen:
                chosen.append(sequence[rank])
                if len(chosen) == limit:return chosen
    return chosen


def compact_group(group):
    result = deepcopy(group)
    aliases = result.get('related_groups', [])
    result['related_groups'] = aliases[:6]
    result['omitted_related_groups'] = result.get('omitted_related_groups', 0)+max(0, len(aliases)-6)
    return result


def compact_report(report, limit=4):
    """The full computed groups stay in the immutable scoped diagnostic artifact."""
    result = deepcopy({k:v for k,v in report.items() if k != 'diagnostic_groups'})
    for field in ('hard_groups', 'error_associations'):
        values = result.get(field, [])
        result['omitted_'+field] = result.get('omitted_'+field, 0)+max(0, len(values)-limit)
        result[field] = [compact_group(g) for g in values[:limit]] if field == 'hard_groups' else values[:limit]
    return result


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
    from agent.supervised import bind_validation_policy
    from agent.predictions import predict, targets, validation_columns
    started = time.monotonic()
    records = []
    with limits:
        for number, split in enumerate(job['diagnostic']['folds']):
            model = validate_estimator(enforce_threads(bind_validation_policy(module.make_pipeline(task, dict(params)),
                job.get('training_policy')), job['cpu_threads']))
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


def error_values(truth, prediction, task):
    import numpy as np
    y = np.asarray(truth, dtype=float if task['task_kind'] == 'regression' else str)
    raw = np.asarray(prediction)
    if task['task_kind'] == 'regression':
        p = raw.astype(float).reshape(y.shape)
        residual = p-y
        return y, p, np.abs(residual), residual**2
    p = (np.asarray(task['allowed_labels'])[raw.argmax(axis=1)]
         if task.get('metric', {}).get('prediction_kind') == 'probabilities' else raw.reshape(-1).astype(str))
    return y.reshape(-1), p, (p != y.reshape(-1)).astype(float), None


def error_profile(data, truth, prediction, fold, task, *, deadline=None):
    """Cost of measured misses on a uniform FIT sample; never expected remedy gain."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    from agent.predictions import targets
    y, p, absolute, squared = error_values(truth, prediction, task)
    regression = squared is not None
    error = absolute.mean(axis=1) if absolute.ndim > 1 else absolute
    metric = task.get('metric', {}).get('metric_id', 'rmse' if regression else 'f1_macro')
    mass = (squared.mean(axis=1) if squared.ndim > 1 else squared) if regression and metric != 'mae' else error
    baseline = float(mass.mean()); fold = np.asarray(fold)
    excluded = set(task.get('excluded_features', [])) | set(targets(task))
    eligible = [c for c in data if c not in excluded]
    groups, associations, scanned = [], [], 0
    memberships = {}
    labels = task.get('allowed_labels', sorted(set(y))) if not regression else []
    weights = {str(label): 1/max(1, int((y == label).sum())) for label in labels}
    total_cost = float(mass.sum()) if regression else sum(weights.get(str(a), 0) for a, b in zip(y, p) if a != b)
    def add(column, label, mask, definition, applicable=True):
        rows = int(mask.sum())
        if not rows or rows == len(error):return
        membership = np.packbits(mask).tobytes()
        alias = {'feature': str(column), 'group': str(label)[:100], 'definition': definition,
                 'test_applicable': applicable}
        if membership in memberships:
            existing = memberships[membership]
            existing['related_groups'].append(alias)
            if applicable and not existing['test_applicable']:
                existing.update(alias)
            return
        supported = [f for f in np.unique(fold) if (mask & (fold == f)).any()]
        deltas = [float(mass[mask & (fold == f)].mean()-mass[fold == f].mean()) for f in supported]
        record = {'feature': str(column), 'group': str(label)[:100], 'definition': definition,
            'test_applicable': applicable, 'rows': rows, 'support_fraction': rows/len(error),
            'support_uncertain': rows <= 10 or len(supported) < 2, 'mean_error': float(error[mask].mean()),
            'error_delta': float(mass[mask].mean()-baseline), 'folds_with_support': len(supported),
            'folds_with_higher_error': sum(d > 0 for d in deltas),
            'per_fold': [{'fold': int(f), 'rows': int((mask & (fold == f)).sum()), 'error_delta': d}
                         for f, d in zip(supported, deltas)]}
        if regression:
            part = squared[mask]
            record.update(sse=float(part.sum()), rmse=float(np.sqrt(part.mean())),
                sse_fraction=float(part.sum()/squared.sum()) if squared.sum() else 0.,
                prediction_bias=float((p-y)[mask].mean()), priority_cost=float(mass[mask].sum()))
        else:
            pr, re, f1, support = precision_recall_fscore_support(y[mask], p[mask], labels=labels, zero_division=0)
            class_metrics = [
                {'label': str(l), 'rows': int(n), 'precision': float(a), 'recall': float(b), 'f1': float(c)}
                for l, a, b, c, n in zip(labels, pr, re, f1, support)]
            record.update(macro_f1=float(f1.mean()), class_metrics=sorted(class_metrics, key=lambda r:r['f1'])[:6],
                omitted_classes=max(0,len(labels)-6),
                priority_cost=float(sum(weights.get(str(a), 0) for a, b in zip(y[mask], p[mask]) if a != b)))
        record['priority_fraction'] = record['priority_cost']/total_cost if total_cost else 0.
        record['row_membership_sha256'] = hashlib.sha256(json.dumps(sorted(map(str, data.index[mask]))).encode()).hexdigest()
        record['related_groups'] = [alias]
        memberships[membership] = record
        groups.append(record)
    for column in eligible:
        if deadline is not None and time.monotonic() >= deadline:break
        series = data[column]
        add(column, 'missing', series.isna().to_numpy(), {'kind': 'missing', 'column': str(column)})
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)
            cuts = sorted(set(numeric.dropna().quantile([.25, .5, .75]).tolist()))
            for i in range(len(cuts)+1) if cuts else []:
                lower, upper = cuts[i-1] if i else None, cuts[i] if i < len(cuts) else None
                mask = numeric.notna().to_numpy()
                if lower is not None:mask &= numeric.ge(lower).to_numpy()
                if upper is not None:mask &= numeric.lt(upper).to_numpy()
                add(column, 'quantile_'+str(i), mask, {'kind': 'range', 'column': str(column), 'lower': lower, 'upper': upper})
            valid = numeric.notna().to_numpy()
            if valid.sum() >= 3 and numeric[valid].nunique() > 1:
                ranks = numeric[valid].rank().to_numpy(); errors = pd.Series(mass[valid]).rank().to_numpy()
                corr = np.corrcoef(ranks, errors)[0, 1] if np.std(errors) else 0.
                associations.append({'feature': str(column), 'error_mass_spearman': float(corr)})
        else:
            actual = series.astype(str)
            # Cost-first category discovery includes rare groups without oversampling.
            costs = pd.Series(mass, index=series.index).groupby(actual).sum().sort_values(ascending=False)
            for value in costs.index[:32]:
                add(column, value, (actual.eq(value) & series.notna()).to_numpy(),
                    {'kind': 'equals', 'column': str(column), 'value': str(value)})
        scanned += 1
    if not regression:
        for label in labels:add('target_class', label, y == label, {'kind': 'target_class', 'label': str(label)}, False)
    result = {'scope': 'fit_only_inner_oof', 'metric_id': metric, 'predicted_rows': len(error),
        'mean_error': float(error.mean()), 'eligible_columns': len(eligible), 'scanned_columns': scanned,
        'omitted_columns': len(eligible)-scanned, 'groups_overlap': True,
        'diagnostic_groups': groups,
        'hard_groups': diverse_groups(groups),
        'omitted_groups': max(0, len(groups)-12),
        'duplicate_groups_merged': sum(len(g['related_groups'])-1 for g in groups),
        'error_associations': sorted(associations, key=lambda r: -abs(r['error_mass_spearman']))[:8],
        'priority_measure': ('absolute_error_mass' if metric == 'mae' else 'squared_error_mass') if regression else 'class_normalized_misses_proxy',
        'interpretation': 'Discovery on the reported uniform fit sample only; overlapping shares are not additive or population estimates. '
            'High cost is not proven fixability or expected gain. Small support remains uncertain. '
            'Target classes/worst residual rows are post-hoc; deployment rules require available inputs/predictions.'}
    if regression:
        total = float(squared.sum()); row_sse = squared.sum(axis=1) if squared.ndim > 1 else squared
        result.update(sse=total, rmse=float(np.sqrt(squared.mean())), prediction_bias=float((p-y).mean()),
            error_concentration=[{'worst_fraction': q, 'rows': max(1, int(np.ceil(len(error)*q))),
                'sse_fraction': float(np.sort(row_sse)[::-1][:max(1, int(np.ceil(len(error)*q)))].sum()/total) if total else 0.,
                'test_applicable': False} for q in (.01, .05, .1)])
    else:
        pr, re, f1, support = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
        classes = [{'label': str(l), 'rows': int(n), 'precision': float(a), 'recall': float(b), 'f1': float(c)}
                   for l, a, b, c, n in zip(labels, pr, re, f1, support)]
        pairs = pd.DataFrame({'truth':y, 'prediction':p}).value_counts().head(16)
        result['class_summary'] = {'macro_f1': float(f1.mean()), 'class_count': len(labels),
            'classes': sorted(classes,key=lambda r:r['f1'])[:12], 'omitted_classes': max(0,len(labels)-12),
            'confusion': [{'truth':str(a), 'prediction':str(b), 'rows':int(n)} for (a,b),n in pairs.items()],
            'confusion_is_bounded': True}
        if task.get('metric'):
            from agent.evaluation import score_predictions
            result['primary_metric_value'] = float(score_predictions(task, y, np.asarray(prediction)))
    return result


class FitDiagnostics:
    """Dispatcher-owned renewable queue; at most one diagnostic learner at once."""
    def __init__(self, controller):
        import threading
        self.c = controller
        self.attempted, self.records = set(), []
        self.spent = 0.
        self.pending, self.running, self.retry_at = [], None, {}
        self.failures = {}
        self.lock = threading.RLock()

    def postpone_failure(self, eid, error):
        import hashlib
        signature = hashlib.sha256((type(error).__name__+':'+str(error)).encode()).hexdigest()
        previous, count = self.failures.get(eid, (None, 0))
        count = count+1 if previous == signature else 1
        self.failures[eid] = (signature, count)
        self.retry_at[eid] = time.monotonic()+min(900, 30*2**min(count-1, 5))

    def visible(self, ids):
        with self.lock:
            return deepcopy([r for r in self.records if r['experiment_id'] in set(ids)][-4:])

    def remaining(self):
        c = self.c
        total = c.config.time_budget_seconds*c.config.research_oof_budget_fraction
        progress = min(1., max(0., (time.monotonic()-c.budget.started)/c.config.time_budget_seconds))
        # Reserve half of the allowance for later representations.
        unlocked = total*(.5+.5*progress)
        return min(unlocked-self.spent,
                   c.budget.remaining()-c.budget.reserve_seconds()-2, c.budget.phase_remaining()-2)

    def ensure(self, ids):
        c, cfg = self.c, self.c.config
        if not cfg.research_oof_enabled or cfg.research_oof_jobs == 0 or not c.store.champion():return
        candidates = []
        for eid in ids:
            exp, ev = c.store.get(eid), c.store.experiment_evaluation(eid)
            if (exp['state'] == 'evaluated' and ev and ev['valid'] and not ev['violations']
                    and exp['plan']['model']['family'] != 'dummy'
                    and ev['primary_metric']['validation_id'] == c.validation['validation_id']):
                m = ev['primary_metric']
                candidates.append((m['value']*(1 if m['direction'] == 'minimize' else -1), eid))
        if not candidates or self.remaining() < 5 or not c.budget.snapshot()['search_allowed']:return
        eid = min(candidates)[1]
        if eid not in self.pending and (not self.running or self.running['eid'] != eid):
            reports = [r for r in self.records if r['experiment_id'] == eid]
            previous = self.records[-1] if self.records else None
            plan = c.store.get(eid)['plan']
            refresh = not previous
            if previous and previous['experiment_id'] != eid:
                old = c.store.get(previous['experiment_id'])['plan']
                old_metric = c.store.experiment_evaluation(previous['experiment_id'])['primary_metric']
                metric = c.store.experiment_evaluation(eid)['primary_metric']
                refresh = (old['code_artifact_id'] != plan['code_artifact_id'] or old['recipe_id'] != plan['recipe_id']
                    or old.get('feature_steps') != plan.get('feature_steps')
                    or abs(old_metric['value']-metric['value']) >= .01*max(abs(old_metric['value']), 1e-12)
                    or time.monotonic()-previous['completed_monotonic'] >= 600)
            if reports:refresh = time.monotonic()-reports[-1]['completed_monotonic'] >= 600
            if refresh:
                # A pending older leader is replaceable; completed reports and
                # an already running diagnostic retain their exact identities.
                self.pending[:] = [eid]
                c.log.event('diagnostics.oof_queued', experiment_id=eid, reason='new_representation_progress_or_stale')
        self.dispatch()

    def dispatch(self):
        c, cfg = self.c, self.c.config
        if self.running or not self.pending or self.remaining() < 5 or not c.budget.snapshot()['search_allowed']:return
        eid = next((e for e in self.pending if self.retry_at.get(e, 0) <= time.monotonic()), None)
        if eid is None:return
        snapshot = c.budget.snapshot()
        reserve_cpu = int(not c.budget.analysis_active and c.budget.cpu_capacity > 1)
        cpu = min(max(1, (c.budget.cpu_capacity-reserve_cpu)//max(1, cfg.max_workers)), snapshot['cpu_threads_available']-reserve_cpu)
        memory = min(cfg.train_memory_mb, (c.budget.memory_capacity-c.budget.analysis_memory)//max(1, cfg.max_workers),
                     snapshot['memory_mb_available']-(0 if c.budget.analysis_active else c.budget.analysis_memory))
        if cpu < 1 or memory < 256:return
        owner = 'fit-diagnostics-'+uuid.uuid4().hex[:12]
        resources = {'cpu_threads': cpu, 'memory_limit_mb': memory,
                     'timeout_seconds': int(min(cfg.research_oof_timeout_seconds, self.remaining()))}
        lease = job = None
        started = time.monotonic()
        try:
            allocation = c.lease(owner, 'train', resources)
            if not allocation['granted']:return
            lease = allocation['lease']
            import numpy as np
            from agent.data_transport import transport
            from agent.predictions import targets
            indices = c.artifacts.read_json(c.validation['train_rows_artifact_id'])
            selected = sorted(np.random.default_rng(cfg.seed).choice(indices, min(len(indices), cfg.research_oof_rows), replace=False).tolist())
            proposal = c.evaluator.protocols[c.validation['validation_id']]['proposal']
            columns = targets(c.spec)
            if proposal['strategy'] in {'group_holdout', 'temporal_holdout'}:
                columns = list(dict.fromkeys([*columns, proposal['parameters']['column']]))
            table = transport(c.artifacts).read(c.spec['train_artifact_id'], c.spec, columns=columns, rows=selected)
            folds = inner_folds(table, c.spec, proposal, cfg.seed)
            fit = c.artifacts.write_json('diagnostics/'+owner+'.fit.json', selected, 'diagnostic_fit_rows')
            unused = c.artifacts.write_json('diagnostics/'+owner+'.unused.json', [], 'diagnostic_unused_rows')
            job = c.runner.start_diagnostics({'lease': lease, 'lease_owner_id': owner,
                'train_rows_artifact_id': fit['artifact_id'], 'valid_rows_artifact_id': unused['artifact_id'],
                'diagnostic': {'folds': folds, 'scope': 'fit_only_inner_oof'}}, c.spec, c.store.get(eid))
            self.running = {'job': job, 'eid': eid, 'selected': selected, 'folds': folds, 'proposal': proposal,
                            'started': started, 'complete_rows': len(indices), 'owner': owner}
            self.pending.remove(eid);self.attempted.add(eid)
            c.log.event('diagnostics.oof_started', experiment_id=eid, job_id=job.job_id,
                sampled_fit_rows=len(selected), complete_fit_rows=len(indices), folds=len(folds),
                cpu_threads=cpu, timeout_seconds=lease['timeout_seconds'])
        except (AgentError, ValueError, OSError, KeyError) as error:
            self.postpone_failure(eid, error)
            c.log.event('diagnostics.oof_unavailable', experiment_id=eid, reason=str(error)[:600], retryable=True)
        finally:
            if lease and job is None:
                c.call('C13', {'operation': 'release', 'owner_id': owner, 'phase': 'train', 'requested': resources,
                    'lease_id': lease['lease_id'], 'fencing_token': lease['fencing_token']}, c.budget.lease)
                self.spent += time.monotonic()-started

    def poll(self):
        if not self.running:
            self.dispatch();return
        c, state = self.c, self.running
        job = state['job']
        if c.runner.poll(job) is None:c._heartbeat_and_limits(job)
        if c.runner.poll(job) is None:return
        # Clear before wait_jobs: that routine may itself service diagnostics.
        self.running = None
        try:
            event = c.wait_jobs([job])[job.job_id]
            if event['state'] != 'succeeded':raise ValueError('OOF worker ended '+event['state'])
            self.complete(state, event)
        except (AgentError, ValueError, OSError, KeyError) as error:
            self.postpone_failure(state['eid'], error)
            if state['eid'] not in self.pending:self.pending.append(state['eid'])
            c.log.event('diagnostics.oof_unavailable', experiment_id=state['eid'], reason=str(error)[:600], retryable=True)
        finally:self.spent += time.monotonic()-state['started']
        c.publish_research_updates()

    def complete(self, state, event):
        import pandas as pd
        from agent.data_transport import transport
        from agent.predictions import targets, validation_columns, checked_matrix
        c = self.c
        job, eid, selected, folds = state['job'], state['eid'], state['selected'], state['folds']
        raw = pd.read_csv(job.prediction_path, dtype={n: str for n in validation_columns(c.spec)}
                          if c.spec['metric']['prediction_kind'] == 'labels' else None, keep_default_na=False)
        rows = raw['__row_index'].tolist()
        expected = sorted(selected[i] for f in folds for i in f['predict'])
        if rows != expected or len(set(rows)) != len(rows):raise ValueError('OOF coverage differs from trusted fit folds')
        data = transport(c.artifacts).read(c.spec['train_artifact_id'], c.spec, rows=rows)
        names = targets(c.spec);y = data[names if len(names) > 1 else names[0]]
        pred = checked_matrix(raw[validation_columns(c.spec)].to_numpy(), len(rows), len(validation_columns(c.spec)),
                              c.spec['metric']['prediction_kind'], labels=c.spec['allowed_labels'])
        report = error_profile(data, y, pred, raw['__fold'], c.spec, deadline=time.monotonic()+10)
        prediction_id = next(a for a in event['output_artifact_ids'] if c.artifacts.meta(a)['kind'] == 'fit_oof_predictions')
        plan = c.store.get(eid)['plan']
        identity = {'experiment_id': eid, 'dataset_id': c.spec['dataset_id'], 'task_spec_id': c.spec['task_spec_id'],
            'validation_id': c.validation['validation_id'], 'protocol_sha256': c.validation['protocol_sha256'],
            'train_sha256': c.artifacts.meta(c.spec['train_artifact_id'])['sha256'],
            'code_sha256': c.artifacts.meta(plan['code_artifact_id'])['sha256'],
            'parameters': deepcopy(plan['model']['hyperparameters']), 'seed': c.config.seed,
            'sample_rows': selected, 'predicted_row_ids': rows, 'folds': folds,
            'predictions_artifact_id': prediction_id, 'predictions_sha256': c.artifacts.meta(prediction_id)['sha256']}
        provenance = c.artifacts.write_json('diagnostics/'+state['owner']+'.identity.json', identity, 'fit_oof_identity')
        report.update(experiment_id=eid, validation_id=c.validation['validation_id'],
            sampled_fit_rows=len(selected), complete_fit_rows=state['complete_rows'], sampled=len(selected)<state['complete_rows'],
            sampling='uniform_without_replacement', folds=len(folds), split_strategy=state['proposal']['strategy'],
            warmup_rows_without_oof=len(selected)-len(rows), coverage_fraction=len(rows)/max(1, state['complete_rows']),
            identity_artifact_id=provenance['artifact_id'], score_authoritative=False)
        art = c.artifacts.write_json('diagnostics/'+state['owner']+'.json', report, 'fit_diagnostics')
        record = {'artifact_id': art['artifact_id'], 'experiment_id': eid, 'report': report,
                  'identity_artifact_id': provenance['artifact_id'], 'completed_monotonic': time.monotonic()}
        with self.lock:self.records.append(record)
        analysis = getattr(c.context, 'analysis', None)
        if analysis is not None:
            # Copy-on-write publication: model threads read immutable metadata,
            # never SQLite, a mutable dispatcher collection or held-out labels.
            analysis.oof_sources = {**getattr(analysis, 'oof_sources', {}), eid: {
                **record, 'identity': deepcopy(identity),
                'identity_sha256': provenance['sha256']}}
        if getattr(c, 'error_research', None):c.error_research.ingest(record)
        c.log.event('diagnostics.oof_completed', artifact_id=art['artifact_id'], experiment_id=eid,
                    predicted_rows=len(raw), primary_score_changed=False)

    def shutdown(self):
        self.pending.clear()
        if self.running:
            state = self.running;self.running = None
            self.c.cancel_job(state['job'], 'Diagnostic finalization barrier')
            self.c.wait_jobs([state['job']])
            self.spent += time.monotonic()-state['started']
