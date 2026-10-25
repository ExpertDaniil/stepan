"""Opt-in supervised representations, fitted inside every enclosing estimator fit.

No global OOF cache. fit_transform excludes each training row's fold; transform
uses only fitted history. Temporal lookup is strictly earlier than prediction time.
"""
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from agent.feature_engineering import _keys


def time_values(values):
    if pd.api.types.is_numeric_dtype(values):
        result = pd.to_numeric(values, errors='raise').to_numpy(dtype=float)
    else:
        result = pd.to_datetime(values, format='ISO8601', utc=True, errors='raise').astype('int64').to_numpy()
    if values.isna().any() or not np.isfinite(result).all():
        raise ValueError('Temporal representation requires complete ordered time values')
    return result


def inner_splits(X, y, *, classification=False, folds=4, seed=42, time_column=None, group_column=None):
    from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
    if time_column:
        t = time_values(X[time_column])
        periods = np.unique(t)
        for block in np.array_split(periods, min(folds+1, len(periods)))[1:]:
            fit, valid = np.flatnonzero(t < block.min()), np.flatnonzero(np.isin(t, block))
            if len(fit) and len(valid):
                yield fit, valid
    elif group_column:
        g = _keys(X[group_column])
        count = min(folds, g.nunique())
        if count >= 2:
            yield from GroupKFold(count).split(X, y, groups=g)
    else:
        count = min(folds, len(X))
        if count < 2:
            return
        least = pd.Series(y).value_counts().min() if classification else 0
        if classification and least >= 2:
            yield from StratifiedKFold(min(count, int(least)), shuffle=True, random_state=seed).split(X, y)
        else:
            yield from KFold(count, shuffle=True, random_state=seed).split(X)


class TargetStatistics(TransformerMixin, BaseEstimator):
    def __init__(self, columns, classification=False, classes=None, smoothing=20.,
                 time_column=None, group_column=None, folds=4, seed=42,
                 max_groups=20000, max_cells=2000000, name='target_stats'):
        self.columns, self.classification, self.classes, self.smoothing = columns, classification, classes, smoothing
        self.time_column, self.group_column, self.folds, self.seed = time_column, group_column, folds, seed
        self.max_groups, self.max_cells, self.name = max_groups, max_cells, name

    def _keys(self, X):
        parts = [_keys(X[c]) for c in self.columns]
        result = parts[0].str.len().astype(str)+':'+parts[0]
        for part in parts[1:]:
            result = result+part.str.len().astype(str)+':'+part
        return result.to_numpy()

    def _target(self, y):
        values = np.asarray(y)
        if values.ndim != 1 or pd.isna(values).any():
            raise ValueError('Target statistics require one complete target')
        if self.classification:
            if not self.classes or len(set(self.classes)) != len(self.classes) or len(self.classes) > 64:
                raise ValueError('Classification statistics require explicit fixed class order, at most 64 classes')
            values = values.astype(str)
            if not set(values) <= set(self.classes):
                raise ValueError('Target outside the declared class order')
            return (values[:, None] == np.asarray(self.classes)[None, :]).astype(float)
        values = values.astype(float)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite target')
        return values[:, None]

    def _aggregate(self, keys, y, times=None):
        prior = y.mean(axis=0) if len(y) else self.default_
        table = {}
        for key, positions in pd.Series(np.arange(len(keys))).groupby(keys, sort=False):
            positions = positions.to_numpy()
            if times is None:
                table[key] = (len(positions), y[positions].sum(axis=0))
            else:
                t, inverse = np.unique(times[positions], return_inverse=True)
                counts = np.bincount(inverse)
                sums = np.zeros((len(t), y.shape[1]))
                np.add.at(sums, inverse, y[positions])
                table[key] = (t, np.cumsum(counts), np.cumsum(sums, axis=0))
        if len(table) > self.max_groups:
            raise ValueError('Target-statistics group limit exceeded')
        return prior, table

    def fit(self, X, y):
        if not 1 <= len(self.columns) <= 4 or len(set(self.columns)) != len(self.columns) or not set(self.columns) <= set(X):
            raise ValueError('Name one to four distinct available key columns')
        if not np.isfinite(self.smoothing) or self.smoothing <= 0 or not 2 <= self.folds <= 10:
            raise ValueError('Positive smoothing and two to ten inner folds required')
        self.input_columns_ = list(X.columns)
        self.default_ = np.full(len(self.classes), 1/len(self.classes)) if self.classification and self.classes else np.zeros(1)
        values = self._target(y)
        if len(X)*(values.shape[1]+1) > self.max_cells:
            raise ValueError('Target-statistics memory cell limit exceeded')
        keys = self._keys(X)
        times = time_values(X[self.time_column]) if self.time_column else None
        self.prior_, self.statistics_ = self._aggregate(keys, values, times)
        if times is not None:
            _, global_table = self._aggregate(np.full(len(X), 'all'), values, times)
            self.history_ = global_table['all']
        names = [self.name+'__'+str(i) for i in range(values.shape[1])]+[self.name+'__support']
        if set(names) & set(X):
            raise ValueError('Target-statistics outputs would overwrite an input')
        self.output_columns_ = names
        self.report_ = {'operator': 'cross_fitted_target_statistics', 'columns': list(self.columns),
            'class_order': list(self.classes or []), 'time_column': self.time_column, 'group_column': self.group_column,
            'smoothing': self.smoothing, 'fit_rows': len(X), 'groups': len(self.statistics_),
            'training': 'strictly_past' if self.time_column else 'inner_out_of_fold',
            'inference': 'fitted_history_only; no targets from predicted batch', 'causal_entity_identity': 'not_established'}
        return self

    def _encode(self, X, prior, table):
        keys = self._keys(X)
        output = np.zeros((len(X), len(self.default_)+1))
        output[:, :-1] = prior
        if self.time_column:
            times = time_values(X[self.time_column])
            gt, gn, gs = self.history_
            loc = np.searchsorted(gt, times, side='left')-1
            available = loc >= 0
            output[:, :-1] = self.default_
            output[available, :-1] = gs[loc[available]]/gn[loc[available], None]
        for key, positions in pd.Series(np.arange(len(keys))).groupby(keys, sort=False):
            positions = positions.to_numpy()
            stats = table.get(key)
            if stats is None:
                continue
            if self.time_column:
                t, counts, sums = stats
                where = np.searchsorted(t, times[positions], side='left')-1
                mask = where >= 0
                positions, where = positions[mask], where[mask]
                n, total = counts[where], sums[where]
            else:
                n, total = stats
            prior_rows = output[positions, :-1]
            output[positions, :-1] = (total+self.smoothing*prior_rows)/(np.asarray(n)[..., None]+self.smoothing)
            output[positions, -1] = np.log1p(n)
        return output

    def _append(self, X, encoded):
        if set(X.columns) != set(self.input_columns_):
            raise ValueError('Target-statistics input schema changed')
        result = X.loc[:, self.input_columns_].copy()
        for i,c in enumerate(self.output_columns_):
            result[c] = encoded[:, i]
        return result

    def transform(self, X):
        return self._append(X, self._encode(X, self.prior_, self.statistics_))

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        if self.time_column:
            return self.transform(X)
        encoded = np.zeros((len(X), len(self.default_)+1))
        encoded[:, :-1] = self.default_
        values, keys = self._target(y), self._keys(X)
        for a,b in inner_splits(X, y, classification=self.classification, folds=self.folds,
                                seed=self.seed, group_column=self.group_column):
            prior, table = self._aggregate(keys[a], values[a])
            encoded[b] = self._encode(X.iloc[b], prior, table)
        return self._append(X, encoded)


def target_statistics(task_spec, columns, **options):
    """Executor supplies the frozen split semantics, including during final refit."""
    context = task_spec.get('_fit_context', {})
    strategy, column = context.get('strategy'), context.get('column')
    if not context:
        raise ValueError('Owned target statistics require executor-provided fit context')
    forbidden = set(task_spec.get('target_columns', [task_spec['target_column']])) | set(task_spec['excluded_features'])
    if set(columns) & forbidden:
        raise ValueError('Target/identifier key is forbidden')
    if set(options) & {'classification', 'classes', 'time_column', 'group_column'}:
        raise ValueError('Split and class semantics belong to the executor')
    return TargetStatistics(columns, classification=task_spec['task_kind'] != 'regression',
        classes=task_spec.get('allowed_labels'), time_column=column if strategy == 'temporal_holdout' else None,
        group_column=column if strategy == 'group_holdout' else None, **options)


class MetricDecisionClassifier(BaseEstimator):
    """Inner-fit class decisions, separate from probability calibration."""
    def __init__(self, estimator, classes, seed=42, time_column=None, group_column=None):
        self.estimator, self.classes, self.seed = estimator, classes, seed
        self.time_column, self.group_column = time_column, group_column

    def _probabilities(self, model, X):
        raw = np.asarray(model.predict_proba(X), dtype=float)
        result = np.zeros((len(X), len(self.classes_)))
        for i,c in enumerate(model.classes_):
            result[:, list(self.classes_).index(str(c))] = raw[:, i]
        if not np.isfinite(result).all() or (result < 0).any() or not np.allclose(result.sum(axis=1), 1):
            raise ValueError('Invalid inner class probabilities')
        return result

    def fit(self, X, y):
        from sklearn.base import clone
        from sklearn.metrics import f1_score
        self.classes_ = np.asarray(self.classes, dtype=str)
        if len(set(self.classes_)) != len(self.classes_) or len(self.classes_) < 2:
            raise ValueError('Explicit class order required')
        truth = np.asarray(y, dtype=str)
        self.weights_ = np.ones(len(self.classes_))
        splits = list(inner_splits(X, truth, classification=True, seed=self.seed,
                                  time_column=self.time_column, group_column=self.group_column))
        self.decision_report_ = {'scope': 'inner_fit_only', 'probabilities_calibrated': False,
                                 'accepted': False, 'class_order': self.classes_.tolist()}
        if splits:
            a,b = splits[-1]
            if len(np.unique(truth[a])) >= 2 and len(b) >= 40:
                pilot = clone(self.estimator).fit(X.iloc[a], truth[a])
                p = self._probabilities(pilot, X.iloc[b])
                # Fixed disjoint subsets: optimize then confirm inside outer fit.
                left, right = np.arange(len(b))[::2], np.arange(len(b))[1::2]
                def score(weights, rows):
                    return f1_score(truth[b][rows], self.classes_[np.argmax(p[rows]*weights, axis=1)],
                                    labels=self.classes_, average='macro', zero_division=0)
                weights = self.weights_.copy()
                initial = score(weights, left)
                for _ in range(2):
                    changed = False
                    for c,label in enumerate(self.classes_):
                        if min(np.sum(truth[b][left] == label), np.sum(truth[b][right] == label)) < 3:
                            continue
                        best_score, best = score(weights, left), weights[c]
                        for value in (.5, .75, 1., 1.5, 2.):
                            candidate = weights.copy(); candidate[c] = value
                            candidate_score = score(candidate, left)
                            if candidate_score > best_score+1e-12:
                                best_score, best = candidate_score, value
                        changed |= best != weights[c]
                        weights[c] = best
                    if not changed:
                        break
                before, after = score(self.weights_, right), score(weights, right)
                accepted = after >= before+.002
                if accepted:
                    self.weights_ = weights
                self.decision_report_.update(accepted=bool(accepted), inner_fit_rows=len(a),
                    tuning_rows=len(left), confirmation_rows=len(right), before=before, after=after,
                    tuning_before=initial, limitation='One inner holdout; outer validation remains untouched.')
        self.model_ = clone(self.estimator).fit(X, truth)
        self.decision_report_['weights'] = self.weights_.tolist()
        return self

    def predict_proba(self, X):
        return self._probabilities(self.model_, X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X)*self.weights_, axis=1)]


def metric_decision(task_spec, estimator):
    if (task_spec['task_kind'] == 'regression' or task_spec['metric']['metric_id'] != 'f1_macro'
            or task_spec['metric']['prediction_kind'] != 'labels'
            or task_spec['output'].get('prediction_kind', 'labels') != 'labels'):
        raise ValueError('Class decision tuning is available only for Macro F1 label tasks')
    context = task_spec.get('_fit_context')
    if not context:
        raise ValueError('Metric decisions require executor-provided fit context')
    return MetricDecisionClassifier(estimator, task_spec['allowed_labels'],
        time_column=context.get('column') if context['strategy'] == 'temporal_holdout' else None,
        group_column=context.get('column') if context['strategy'] == 'group_holdout' else None)
