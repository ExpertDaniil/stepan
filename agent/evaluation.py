import hashlib
import math
import uuid

from agent.contracts import canonical
from agent.errors import AgentError
from tools.data import read_table
from agent.predictions import checked_matrix, read_training, targets, validation_columns
from agent.data_transport import transport


def score_predictions(spec, truth, prediction):
    import numpy as np
    from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error,
                                 mean_squared_error, r2_score, roc_auc_score)
    key = spec['metric']['metric_id']
    if spec['task_kind'] == 'regression':
        y = np.asarray(truth, dtype=float)
        p = prediction.reshape(-1) if y.ndim == 1 else prediction
        return {'rmse': lambda: math.sqrt(mean_squared_error(y, p)),
                'mae': lambda: mean_absolute_error(y, p), 'r2': lambda: r2_score(y, p)}[key]()
    y = np.asarray(truth, dtype=str)
    if spec['metric']['prediction_kind'] == 'probabilities':
        if key == 'log_loss':
            return log_loss(y, prediction, labels=spec['allowed_labels'])
        positive = spec['metric']['parameters']['positive_label']
        return roc_auc_score(y == positive, prediction[:, spec['allowed_labels'].index(positive)])
    p = prediction.reshape(-1).astype(str)
    return (accuracy_score(y, p) if key == 'accuracy' else f1_score(y, p,
        average='macro' if key == 'f1_macro' else 'weighted', labels=spec['allowed_labels'], zero_division=0))


def error_diagnostics(data, task, fit_indices, valid_indices, y, prediction, columns=None, *, max_groups=4):
    """Selection-only aggregates. Column groups overlap; never add their shares."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import precision_recall_fscore_support
    regression = task['task_kind'] == 'regression'
    truth = np.asarray(y, dtype=float if regression else str)
    predicted = np.asarray(prediction, dtype=float if regression else str)
    if regression and truth.ndim != predicted.ndim:
        predicted = predicted.reshape(truth.shape)
    error = np.abs(truth-predicted) if regression else (truth != predicted).astype(float)
    squared = (truth-predicted)**2 if regression else None
    row_error = error.mean(axis=1) if error.ndim > 1 else error
    row_sse = squared.sum(axis=1) if regression and squared.ndim > 1 else squared
    total_sse = float(np.sum(squared)) if regression else None
    result = {'scope': 'selection_validation_aggregates', 'rows': len(truth),
              'groups_overlap': True, 'aggregation': 'different feature partitions overlap; shares are not additive across features'}
    if regression:
        result.update(sse=total_sse, rmse=float(np.sqrt(np.mean(squared))) if len(truth) else None,
                      target_count=truth.shape[1] if truth.ndim > 1 else 1)
        if len(truth):
            ordered = np.sort(row_sse)[::-1]
            result['error_concentration'] = [{'worst_fraction': f, 'rows': max(1, int(np.ceil(len(truth)*f))),
                'sse_fraction': float(ordered[:max(1, int(np.ceil(len(truth)*f)))].sum()/total_sse) if total_sse else 0.0}
                for f in (.01, .05, .1)]
    else:
        labels = task.get('allowed_labels') or sorted(set(truth) | set(predicted))
        precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=labels, zero_division=0)
        classes = [{'label': str(label)[:80], 'rows': int(n), 'precision': float(p), 'recall': float(r), 'f1': float(f)}
                   for label, p, r, f, n in zip(labels, precision, recall, f1, support)]
        result['class_summary'] = {'class_count': len(labels), 'macro_f1': float(np.mean(f1)),
            'worst_classes': sorted(classes, key=lambda c: (c['f1'], -c['rows']))[:6],
            'omitted_classes': max(0, len(labels)-6)}
    excluded = set(task['excluded_features']) | set(targets(task))
    candidates = list(columns) if columns is not None else [c for c in data if c not in excluded]
    if columns is None:
        def priority(column):
            fit, valid = data[column].iloc[fit_indices], data[column].iloc[valid_indices]
            missing_shift = abs(float(fit.isna().mean())-float(valid.isna().mean()))
            categorical = not pd.api.types.is_numeric_dtype(fit)
            new = (float((~valid.astype(str).isin(set(fit.dropna().astype(str)))).mean()) if categorical else 0.0)
            return (-missing_shift-new, 0 if categorical and 2 <= fit.nunique() <= 32 else 1, str(column))
        candidates.sort(key=priority)
    groups = []
    for column in candidates[:12 if columns is not None else 6]:
        if column in targets(task) or column not in data:
            continue
        fit, valid = data[column].iloc[fit_indices], data[column].iloc[valid_indices]
        missing = valid.isna().to_numpy()
        masks = [('missing', missing)]
        if pd.api.types.is_numeric_dtype(fit) and not pd.api.types.is_bool_dtype(fit):
            finite = pd.to_numeric(fit, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            cuts = sorted(set(map(float, finite.quantile([1/3, 2/3]).dropna().tolist())))
            vals = pd.to_numeric(valid, errors='coerce').to_numpy(dtype=float)
            buckets = np.searchsorted(cuts, vals, side='right')
            masks += [('bin_'+str(i), (buckets == i) & np.isfinite(vals)) for i in range(len(cuts)+1)]
        else:
            actual = valid.fillna('__MISSING__').astype(str).to_numpy()
            known = set(fit.dropna().astype(str))
            top = fit.dropna().astype(str).value_counts().index[:8]
            masks += [(str(label)[:80], (actual == label) & ~missing) for label in top]
            masks += [('unseen_in_fit', ~np.isin(actual, list(known)) & ~missing)]
        for label, mask in masks:
            count = int(mask.sum())
            if count < 5:
                continue
            record = {'feature': str(column)[:80], 'group': label, 'rows': count,
                      'support_fraction': float(count/max(1, len(truth))), 'mean_error': float(row_error[mask].mean())}
            if regression:
                contribution = float(np.sum(squared[mask]))
                record.update(rmse=float(np.sqrt(np.mean(squared[mask]))), sse=contribution,
                    sse_fraction=contribution/total_sse if total_sse else 0.0,
                    bias=float(np.mean(predicted[mask]-truth[mask])))
            else:
                _, _, f1, _ = precision_recall_fscore_support(truth[mask], predicted[mask], labels=labels, zero_division=0)
                record.update(misclassification_rate=float(row_error[mask].mean()), macro_f1=float(np.mean(f1)))
            groups.append(record)
    result['hard_groups'] = sorted(groups, key=lambda g: g.get('sse', g['mean_error']), reverse=True)[:max_groups]
    result['omitted_groups'] = max(0, len(groups)-max_groups)
    result['minimum_group_support'] = 5
    return result


class Evaluator:
    def __init__(self, spec, artifacts, store, log):
        self.spec, self.artifacts, self.store, self.log = spec, artifacts, store, log
        self.protocols = {}
        self.baseline_value = None
        self.diagnostic_sources = {}
        import threading
        self.diagnostic_lock = threading.RLock()

    def create_protocol(self, payload):
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split, GroupShuffleSplit
        frame = transport(self.artifacts).read(self.spec['train_artifact_id'], self.spec)
        proposal = payload["proposal"]
        params = proposal["parameters"]
        fraction = params.get("valid_fraction", 0.2)
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0.05 <= fraction <= 0.5:
            raise AgentError("INVALID_SPLIT", "valid_fraction must be between 0.05 and 0.5")
        indices = np.arange(len(frame))
        fold_plan = None
        strategy = proposal["strategy"]
        y = frame[self.spec["target_column"]]
        try:
            duplicate_groups = None
            if params.get('duplicate_groups') and strategy in {'stratified_holdout', 'random_holdout'}:
                features = [c for c in frame.columns if c not in targets(self.spec) and c not in self.spec['excluded_features']]
                duplicate_groups = pd.util.hash_pandas_object(frame[features].astype(str), index=False).to_numpy()
            if strategy == "stratified_holdout":
                if self.spec["task_kind"] == "regression":
                    raise AgentError("INVALID_SPLIT", "Stratification requires classification")
                if duplicate_groups is not None:
                    from sklearn.model_selection import StratifiedGroupKFold
                    splitter = StratifiedGroupKFold(max(2, round(1/fraction)), shuffle=True, random_state=proposal["seed"])
                    a, b = next(splitter.split(indices, y.astype(str), duplicate_groups))
                else:
                    a, b = train_test_split(indices, test_size=fraction, random_state=proposal["seed"], stratify=y.astype(str))
            elif strategy == "random_holdout":
                if duplicate_groups is not None:
                    a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=proposal["seed"]).split(
                        indices, groups=duplicate_groups))
                else:
                    a, b = train_test_split(indices, test_size=fraction, random_state=proposal["seed"])
            elif strategy == "group_holdout":
                column = params.get("column")
                if column not in frame or column in targets(self.spec):
                    raise AgentError("INVALID_SPLIT", "Invalid group column")
                a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=proposal["seed"]).split(frame, groups=frame[column].fillna("__MISSING__").astype(str)))
            elif strategy == "temporal_holdout":
                column = params.get("column")
                if column not in frame or column in targets(self.spec) or frame[column].isna().any():
                    raise AgentError("INVALID_SPLIT", "Temporal column must be present and complete")
                # Whole periods stay together. Never split a timestamp boundary.
                ordered = frame[column]
                if not pd.api.types.is_numeric_dtype(ordered):
                    try:
                        ordered = pd.to_datetime(ordered, format="ISO8601", utc=True, errors="raise")
                    except (ValueError, TypeError) as exc:
                        raise AgentError("INVALID_SPLIT", "Temporal column must be numeric or unambiguous ISO-8601") from exc
                periods = sorted(ordered.unique().tolist())
                if len(periods) < 2:
                    raise AgentError("INVALID_SPLIT", "At least two periods required")
                # Rolling origin: at least two validation periods when three exist,
                # split into two consecutive blocks. Each block is predicted by a
                # model fitted on every earlier row, as the deployment would be.
                cut = max(1, min(int(len(periods)*(1-fraction)), len(periods)-2))
                blocks = [list(block) for block in np.array_split(np.asarray(periods[cut:], dtype=object), min(2, len(periods)-cut))]
                cutoff = periods[cut]
                a, b = indices[(ordered < cutoff).to_numpy()], indices[(ordered >= cutoff).to_numpy()]
                values = ordered.to_numpy()
                fold_plan = [{'valid_periods': [str(v) for v in block],
                              'fit': indices[values < block[0]].tolist(),
                              'valid': indices[np.isin(values, np.asarray(block, dtype=values.dtype))].tolist()}
                             for block in blocks]
            else:
                raise AgentError("INVALID_SPLIT", f"Unsupported validation strategy: {strategy}")
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError("INVALID_SPLIT", str(exc)) from exc
        a, b = sorted(map(int, a)), sorted(map(int, b))
        if not a or not b or set(a) & set(b) or len(a) + len(b) != len(frame):
            raise AgentError("INVALID_SPLIT", "Invalid row partition")
        from agent.confirmation import partition
        from agent.semantic_planning import validation_scenario
        scenario = validation_scenario(proposal, self.spec, self.artifacts)
        if fold_plan:
            # Every rolling block is a scored deployment period; none is withheld.
            selection, confirmation, confirmation_reason = list(b), [], 'rolling_temporal_folds_score_every_period'
        else:
            selection, confirmation, confirmation_reason = partition(frame, b, proposal, self.spec)
        protocol = {"dataset_id": payload["dataset_id"], "metric": self.spec["metric"], "proposal": proposal,
                    "train_indices": a, "valid_indices": b, "evaluator_version": "sklearn-metrics-v2",
                    'selection_indices': selection, 'confirmation_indices': confirmation, 'scenario': scenario,
                    'confirmation_unavailable_reason': confirmation_reason,
                    'fold_plan': ([{'valid_periods': f['valid_periods'], 'fit_rows': len(f['fit']), 'valid_rows': len(f['valid'])}
                                   for f in fold_plan] if fold_plan else None),
                    'prediction_contract': {'target_columns': targets(self.spec), 'allowed_labels': self.spec['allowed_labels'],
                                            'prediction_kind': self.spec['metric']['prediction_kind']}}
        sha = hashlib.sha256(canonical(protocol).encode()).hexdigest()
        ident = "validation-" + sha[:16]
        train_art = self.artifacts.write_json(f"validation/{ident}.train.json", a, "split_train")
        valid_art = self.artifacts.write_json(f"validation/{ident}.valid.json", b, "split_valid")
        self.artifacts.write_json(f"validation/{ident}.protocol.json", protocol, "validation_protocol")
        if fold_plan:
            if sorted(i for f in fold_plan for i in f['valid']) != b:
                raise AgentError("INVALID_SPLIT", "Rolling folds must partition the validation rows")
            self.artifacts.write_json(f"validation/{ident}.folds.json", fold_plan, "split_folds")
        self.protocols[ident] = protocol
        self.log.event("validation.created", validation_id=ident, strategy=strategy, parameters=params,
                       metric_id=self.spec["metric"]["metric_id"], direction=self.spec["metric"]["direction"],
                       train_rows=len(a), validation_rows=len(b), protocol_sha256=sha, reason=proposal["reason"])
        self.log.event('validation.confirmation_reserved', validation_id=ident, selection_rows=len(selection),
                       confirmation_rows=len(confirmation), unavailable_reason=confirmation_reason,
                       labels_visibility='evaluator_only_until_final_research_barrier')
        return {"validation_id": ident, "train_rows_artifact_id": train_art["artifact_id"],
                "valid_rows_artifact_id": valid_art["artifact_id"], "labels_visibility": "evaluator_only",
                "metric": self.spec["metric"], "protocol_sha256": sha}

    def evaluate(self, payload):
        import time
        started = time.monotonic()
        import numpy as np
        import pandas as pd
        from sklearn.metrics import mean_absolute_error
        if (payload['task_spec_id'] != self.spec['task_spec_id'] or payload["validation_id"] not in self.protocols
                or payload["dataset_id"] != self.spec["dataset_id"]):
            raise AgentError("STALE_VALIDATION", "Unknown evaluation protocol")
        protocol = self.protocols[payload["validation_id"]]
        path = self.artifacts.resolve(payload["predictions_artifact_id"])
        columns = validation_columns(self.spec)
        kind = self.spec['metric']['prediction_kind']
        frame = pd.read_csv(path, dtype={c: str for c in columns} if kind == 'labels' else None, keep_default_na=False)
        if list(frame.columns) != ["__row_index", *columns] or frame["__row_index"].tolist() != protocol["valid_indices"]:
            raise AgentError("INVALID_PREDICTIONS", "Validation row order or columns do not match")
        data = transport(self.artifacts).read(self.spec['train_artifact_id'], self.spec)
        target_columns = targets(self.spec)
        selection = protocol.get('selection_indices', protocol['valid_indices'])
        y = data[target_columns if len(target_columns) > 1 else target_columns[0]].iloc[selection]
        checked_matrix(frame[columns].to_numpy(), len(frame), len(columns), kind, labels=self.spec['allowed_labels'])
        prediction = checked_matrix(frame.set_index('__row_index').loc[selection, columns].to_numpy(),
                                    len(selection), len(columns), kind, labels=self.spec['allowed_labels'])
        key = self.spec["metric"]["metric_id"]
        try:
            value = score_predictions(self.spec, y, prediction)
        except (ValueError, TypeError) as error:
            raise AgentError('METRIC_FAILED', str(error)) from error
        if self.spec['task_kind'] == 'regression' and len(target_columns) == 1:
            prediction = prediction.reshape(-1)
        elif self.spec['task_kind'] != 'regression':
            y = y.astype(str)
            prediction = (np.asarray(self.spec['allowed_labels'])[prediction.argmax(axis=1)]
                          if kind == 'probabilities' else prediction.reshape(-1))
        if not math.isfinite(float(value)):
            raise AgentError("METRIC_FAILED", "Metric is not finite")
        experiment = self.store.get(payload["experiment_id"])
        if experiment["plan"]["model"]["family"] == "dummy" and self.baseline_value is None:
            self.baseline_value = float(value)
        direction = self.spec["metric"]["direction"]
        improvement = ((self.baseline_value-float(value)) if direction == "minimize" else (float(value)-self.baseline_value)) if self.baseline_value is not None else None
        diagnostics = {"rows": len(selection), 'prediction_rows': len(frame), "metric_id": key, "direction": direction, "score": float(value),
                       "baseline_score": self.baseline_value, "improvement_over_baseline": improvement,
                       "improvement_percent": 100*improvement/abs(self.baseline_value) if self.baseline_value and improvement is not None else None}
        if self.spec["task_kind"] == "regression":
            errors = np.abs(np.asarray(y, dtype=float)-prediction)
            diagnostics.update(mae=float(mean_absolute_error(y, prediction)), target_mean=float(np.mean(np.asarray(y, dtype=float))),
                               target_std=float(np.std(np.asarray(y, dtype=float))), absolute_error_p50=float(np.quantile(errors, 0.5)),
                               absolute_error_p90=float(np.quantile(errors, 0.9)))
        # Describe the actual output; source inspection cannot prove that custom code learns.
        if self.spec['task_kind'] == 'regression':
            spread = np.std(np.asarray(prediction, dtype=float), axis=0)
            scale = np.maximum(1.0, np.mean(np.abs(np.asarray(prediction, dtype=float)), axis=0))
            constant_like = bool(np.all(spread <= 1e-12*scale))
            diagnostics['prediction_behavior'] = {'constant_like': constant_like,
                'prediction_std': np.atleast_1d(spread).astype(float).tolist(),
                'note': 'Constant-like predictions are observed, not proof of a code defect; compare learned alternatives.' if constant_like else 'Predictions vary across validation inputs.'}
        elif self.spec['metric']['prediction_kind'] != 'probabilities':
            values, counts = np.unique(prediction, return_counts=True)
            diagnostics['prediction_behavior'] = {'constant_like': len(values) == 1,
                'predicted_class_count': len(values), 'largest_predicted_class_fraction': float(counts.max()/len(prediction))}
        diagnostics.update(error_diagnostics(data, self.spec, protocol["train_indices"], selection, y, prediction))
        feature_path = path.with_name('features.json')
        with self.artifacts.lock:
            feature_art = next((entry['meta'] for entry in self.artifacts.entries.values()
                if entry['meta']['kind'] == 'feature_execution' and entry['path'] == str(feature_path)), None)
        if feature_art:
            diagnostics['feature_execution_artifact_id'] = feature_art['artifact_id']
            diagnostics['execution_report_scope'] = 'same_job_as_evaluated_predictions; custom semantics unverified'
        diag = self.artifacts.write_json("diagnostics/" + payload["experiment_id"] + ".json", diagnostics, "diagnostics")
        record = {"evaluation_id": "eval-" + uuid.uuid4().hex[:12], "experiment_id": payload["experiment_id"], "valid": True,
                  "primary_metric": {"metric_id": key, "direction": self.spec["metric"]["direction"], "value": float(value),
                                     "validation_id": payload["validation_id"], "dataset_id": payload["dataset_id"], "fidelity": payload["fidelity"]},
                  "diagnostics_artifact_id": diag["artifact_id"], "violations": []}
        with self.diagnostic_lock:
            self.diagnostic_sources[payload['experiment_id']] = {
                'predictions_artifact_id': payload['predictions_artifact_id'],
                'validation_id': payload['validation_id'], 'diagnostics_artifact_id': diag['artifact_id'],
                'selection_sse': diagnostics.get('sse'),
                'predictions_sha256': self.artifacts.meta(payload['predictions_artifact_id'])['sha256']}
        self.store.save_evaluation(record)
        if getattr(self, 'timing_observer', None):self.timing_observer(time.monotonic()-started)
        self.log.event("experiment.evaluated", experiment_id=payload["experiment_id"], recipe_id=experiment["plan"]["recipe_id"],
                       hyperparameters=experiment["plan"]["model"]["hyperparameters"], validation_id=payload["validation_id"],
                       diagnostics_artifact_id=diag["artifact_id"], **diagnostics)
        return {"evaluation": record}

    def diagnose(self, query, check_time):
        """No SQLite, raw labels, confirmation rows, source edits or model execution."""
        import numpy as np
        import pandas as pd
        from tools.analysis import apply_filters, filter_columns
        with self.diagnostic_lock:
            source = dict(self.diagnostic_sources.get(query['experiment_id'], {}))
        if not source:
            raise AgentError('INVALID_ANALYSIS', 'No trusted evaluated predictions for experiment_id')
        protocol = self.protocols[source['validation_id']]
        selection = protocol.get('selection_indices', protocol['valid_indices'])
        requested = list(query['columns'])
        derived = query.get('derived', [])
        generated = {s['name'] for s in derived}
        needed = list(dict.fromkeys([*requested, *filter_columns(query.get('filter'), query.get('filters')),
            *(c for step in derived for c in step['columns'])]))
        needed = [c for c in needed if c not in generated]
        if not requested:
            raise AgentError('INVALID_ANALYSIS', 'Error diagnostics need explicit feature columns')
        check_time()
        full = transport(self.artifacts).read(self.spec['train_artifact_id'], self.spec,
            columns=list(dict.fromkeys([*needed, *targets(self.spec)])))
        if int(full.memory_usage(deep=True).sum()) > 256*1024*1024:
            raise AgentError('ANALYSIS_MEMORY_LIMIT', 'Diagnostic projection exceeds 256 MiB; select fewer columns')
        # Feature bin definitions use fit only. Derived operators are stateless.
        if derived:
            from agent.feature_engineering import FeatureProgram
            for step in derived:
                check_time()
                full[step['name']] = FeatureProgram._apply(full, step, {})
        selected = full.iloc[selection]
        mask = apply_filters(selected, query.get('filter'), query.get('filters')).to_numpy()
        positions = np.flatnonzero(mask)
        population = len(positions)
        if population > 20000:
            positions = np.sort(np.random.default_rng(42).choice(positions, 20000, replace=False))
        rows = np.asarray(selection)[positions].tolist()
        if len(rows) < 5:
            return {'scope': 'selection_validation_aggregates', 'validation_id': source['validation_id'],
                'matched_selection_rows': population, 'insufficient_group_support': True, 'minimum_group_support': 5}
        columns = validation_columns(self.spec)
        kind = self.spec['metric']['prediction_kind']
        from agent.artifacts import file_sha256
        prediction_path = self.artifacts.resolve(source['predictions_artifact_id'])
        expected_sha = source.get('predictions_sha256', self.artifacts.meta(source['predictions_artifact_id'])['sha256'])
        if file_sha256(prediction_path) != expected_sha:
            raise AgentError('INVALID_PREDICTIONS', 'Trusted saved predictions changed after evaluation')
        predictions = pd.read_csv(prediction_path,
            dtype={c: str for c in columns} if kind == 'labels' else None, keep_default_na=False)
        prediction = predictions.set_index('__row_index').loc[rows, columns].to_numpy()
        truth = full[targets(self.spec) if len(targets(self.spec)) > 1 else targets(self.spec)[0]].iloc[rows]
        try:
            metric = score_predictions(self.spec, truth, prediction)
        except ValueError:
            metric = None  # E.g. ROC AUC is undefined for a one-class slice.
        if self.spec['task_kind'] != 'regression':
            prediction = (np.asarray(self.spec['allowed_labels'])[prediction.argmax(axis=1)]
                          if kind == 'probabilities' else prediction.reshape(-1).astype(str))
        elif len(targets(self.spec)) == 1:
            prediction = prediction.reshape(-1)
        check_time()
        result = error_diagnostics(full, self.spec, protocol['train_indices'], rows, truth, prediction, requested, max_groups=12)
        result.update(experiment_id=query['experiment_id'], evidence_ids=[source['diagnostics_artifact_id']],
            validation_id=source['validation_id'], query=query, metric_id=self.spec['metric']['metric_id'],
            slice_metric=float(metric) if metric is not None and math.isfinite(metric) else None,
            selection_rows=len(selection), matched_selection_rows=population, analyzed_rows=len(rows),
            sampled=population > len(rows), sample_rule='uniform_after_feature_filter',
            sse_denominator='analyzed_filtered_rows; not a population estimate',
            confirmation_labels_used=False)
        if source.get('selection_sse') is not None:
            denominator = source['selection_sse']
            result['full_selection_sse'] = denominator
            result['sse_fraction_of_full_selection'] = result['sse']/denominator if denominator else 0.0
            for group in result['hard_groups']:
                group['full_selection_sse_fraction'] = group['sse']/denominator if denominator else 0.0
        return result
