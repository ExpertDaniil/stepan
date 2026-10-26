"""Controller-owned strong tabular learner, fitted only on the rows it receives.

One estimator, three measured mechanisms; each is kept only when its own inner
check (built from the fit rows alone) supports it:

* LightGBM with inner early stopping and numerically guarded multiclass settings.
* Entity history: under a temporal protocol, a key whose rows recur across periods
  is discovered from the fit rows; strictly earlier target values of the same key
  become features of one joint model, where missing history marks a first
  appearance. (A separate base model for first appearances was measured worse on
  later periods: it cannot tell new keys from recurring ones.)
* Seed bagging of the final fit: deployment periods are new, so prediction
  variance across seeds is a first-order error term.
* Class decisions: class weighting for class-balanced metrics and per-class log
  offsets tuned for the declared label metric on one inner half, confirmed on the other.

No dataset, column or label name is referenced. Held-out labels never enter fit.
"""
from copy import deepcopy
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

HISTORY_FEATURES = ('last', 'mean', 'count', 'gap')
ENTITY_MIN_MULTI_PERIOD = 0.05   # Some keys recur; the inner check decides usefulness.
ENTITY_MAX_WITHIN_SHARE = 0.6    # The key explains at least 40% of target variance.
ENTITY_MAX_DUPLICATE = 0.1       # A key is almost unique inside one period.
ENTITY_MAX_COLUMNS = 5
ENTITY_SECONDS = 30.0
HISTORY_MIN_GAIN = 0.01          # Relative inner loss reduction needed to use history.
DECISION_MIN_GAIN = 0.002        # Confirmed inner gain of the declared label metric needed to keep offsets.
OFFSET_GRID = (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0)


MAX_NATIVE_CATEGORIES = 1000      # Above this, a fit-only frequency replaces native levels.
DATE_PATTERN = r'^\d{4}-\d{2}-\d{2}'


def _text(values):
    return values.astype('object').where(values.notna(), None).map(lambda v: None if v is None else str(v))


class TableEncoder:
    """Fit-only column encodings: native categories, frequencies or date parts."""
    def fit(self, X):
        self.columns_ = list(X.columns)
        self.categories_, self.frequencies_, self.dates_ = {}, {}, []
        for column in X.columns:
            values = X[column]
            if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
                continue
            text = _text(values).dropna()
            sample = text.head(500)
            if len(sample) and sample.str.match(DATE_PATTERN).all() and \
                    pd.to_datetime(text, format='ISO8601', errors='coerce', utc=True).notna().mean() >= 0.99:
                self.dates_.append(column)
                continue
            levels = text.unique()
            if len(levels) > MAX_NATIVE_CATEGORIES:
                self.frequencies_[column] = text.value_counts(normalize=True).to_dict()
            else:
                self.categories_[column] = sorted(levels.tolist())
        return self

    def transform(self, X):
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError('Strong model input lacks fitted columns: '+', '.join(map(str, missing[:5])))
        result = {}
        for column in self.columns_:
            values = X[column]
            if column in self.categories_:
                result[column] = pd.Categorical(_text(values), categories=self.categories_[column])
            elif column in self.frequencies_:
                result[column] = _text(values).map(self.frequencies_[column]).fillna(0.0).astype(float)
            elif column in self.dates_:
                stamp = pd.to_datetime(_text(values), format='ISO8601', errors='coerce', utc=True)
                result[column] = (stamp.astype('int64')/1e9).where(stamp.notna()).astype(float)
                for part in ('year', 'month', 'day', 'dayofweek', 'hour'):
                    result[column+'__'+part] = getattr(stamp.dt, part).astype(float)
            else:
                result[column] = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).astype(float)
        return pd.DataFrame(result, index=X.index)


def ordered_values(values):
    """Period values as floats (numbers or ISO dates); None when not orderable."""
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        return pd.to_numeric(values, errors='coerce').astype(float)
    parsed = pd.to_datetime(values, format='ISO8601', errors='coerce', utc=True)
    if parsed.notna().mean() < 0.99:
        return None
    return pd.Series(parsed.astype('int64'), index=values.index).where(parsed.notna()).astype(float)


def _codes(values):
    return pd.factorize(values.astype('object').where(values.notna(), '\x00missing'), sort=False)[0].astype(np.int64)


def _combine(left, right):
    return pd.factorize(left*(int(right.max())+2)+right, sort=False)[0].astype(np.int64)


def key_statistics(key, period, target):
    """Recurrence, within-period uniqueness and within-key target variance share."""
    pairs = key*(int(period.max())+2)+period
    unique_pairs, pair_counts = np.unique(pairs, return_counts=True)
    duplicate = float((pair_counts > 1).mean())
    pair_keys = unique_pairs//(int(period.max())+2)
    periods_per_key = np.bincount(pair_keys)
    present = periods_per_key[periods_per_key > 0]
    multi = float((present > 1).mean())
    counts = np.bincount(key).astype(float)
    sums = np.bincount(key, weights=target)
    squares = np.bincount(key, weights=target*target)
    total = float(np.sum((target-target.mean())**2))
    used = counts > 0
    within = float(np.sum(squares[used]-sums[used]**2/counts[used])/total) if total > 0 else 1.0
    return {'multi_period': multi, 'within_share': within, 'duplicate': duplicate, 'keys': int(used.sum()),
            'score': multi*(1-within)*(1-duplicate)}


def discover_entity(X, y, period_column, seconds=ENTITY_SECONDS):
    """Greedy key search on fit rows; returns the accepted key with evidence, or None."""
    period = ordered_values(X[period_column])
    target = np.asarray(y, dtype=float)
    if period is None or period.isna().any() or not np.isfinite(target).all() or target.std() == 0:
        return None
    period_codes = pd.factorize(period, sort=True)[0].astype(np.int64)
    if period_codes.max() < 1:
        return None
    started = time.monotonic()
    codes = {}
    for column in X.columns:
        if column == period_column:
            continue
        values = X[column]
        unique = values.nunique(dropna=True)
        if unique < 2 or unique >= 0.95*len(values):
            continue
        codes[column] = _codes(values)
    chosen, key, best = [], np.zeros(len(X), dtype=np.int64), None
    while len(chosen) < ENTITY_MAX_COLUMNS and time.monotonic()-started < seconds:
        step = None
        for column, column_codes in codes.items():
            if column in chosen:
                continue
            candidate = _combine(key, column_codes)
            stats = key_statistics(candidate, period_codes, target)
            if step is None or stats['score'] > step[2]['score']:
                step = (column, candidate, stats)
        if step is None or (best is not None and step[2]['score'] <= best['score']+0.002):
            break
        chosen.append(step[0])
        key, best = step[1], step[2]
    if (best is None or best['multi_period'] < ENTITY_MIN_MULTI_PERIOD or best['within_share'] > ENTITY_MAX_WITHIN_SHARE
            or best['duplicate'] > ENTITY_MAX_DUPLICATE or best['keys'] < 10):
        return None
    return {'columns': chosen, 'period_column': period_column, **{k: round(v, 6) if isinstance(v, float) else v
            for k, v in best.items()}}


class EntityHistory:
    """Strictly earlier target history of the same key; fitted rows only."""
    def __init__(self, columns, period_column, prefix='__history_'):
        self.columns, self.period_column, self.prefix = list(columns), period_column, prefix

    def _keys(self, X):
        from agent.feature_engineering import _keys
        parts = [_keys(X[c]) for c in self.columns]
        key = parts[0].str.len().astype(str)+':'+parts[0]
        for part in parts[1:]:
            key = key+part.str.len().astype(str)+':'+part
        return key

    def fit(self, X, y):
        frame = pd.DataFrame({'key': self._keys(X).to_numpy(), 'period': ordered_values(X[self.period_column]).to_numpy(),
                              'y': np.asarray(y, dtype=float)})
        grouped = frame.groupby(['key', 'period'], sort=True).y.agg(['sum', 'count', 'mean']).reset_index()
        grouped['cum_sum'] = grouped.groupby('key')['sum'].cumsum()
        grouped['cum_count'] = grouped.groupby('key')['count'].cumsum()
        self.table_ = pd.DataFrame({'key': grouped['key'], 'period': grouped['period'].astype(float),
            'last': grouped['mean'], 'mean': grouped['cum_sum']/grouped['cum_count'],
            'count': grouped['cum_count'].astype(float)}).sort_values('period', kind='mergesort')
        return self

    def transform(self, X):
        period = ordered_values(X[self.period_column])
        if period is None or period.isna().any():
            raise ValueError('Entity history needs complete ordered period values')
        query = pd.DataFrame({'key': self._keys(X).to_numpy(), 'period': period.to_numpy(), 'row': np.arange(len(X))})
        query = query.sort_values('period', kind='mergesort')
        table = self.table_.rename(columns={'period': 'history_period'})
        merged = pd.merge_asof(query, table, left_on='period', right_on='history_period', by='key',
                               allow_exact_matches=False, direction='backward').sort_values('row')
        result = pd.DataFrame(index=X.index)
        result[self.prefix+'last'] = merged['last'].to_numpy()
        result[self.prefix+'mean'] = merged['mean'].to_numpy()
        result[self.prefix+'count'] = merged['count'].fillna(0.0).to_numpy()
        result[self.prefix+'gap'] = (merged['period']-merged['history_period']).to_numpy()
        return result


def _lgbm_parameters(params, regression, threads, seed, classes=None):
    options = {'n_estimators': 5000, 'learning_rate': params['learning_rate'], 'num_leaves': params['num_leaves'],
               'min_child_samples': params['min_child_samples'], 'colsample_bytree': params['colsample_bytree'],
               'reg_lambda': params['reg_lambda'], 'subsample': 0.8, 'subsample_freq': 1,
               'n_jobs': max(1, int(threads)), 'random_state': int(seed), 'verbose': -1}
    if not regression:
        # Duplicate rows with conflicting labels drive multiclass hessians to zero;
        # a minimum hessian keeps leaf values bounded instead of diverging.
        options.update(min_sum_hessian_in_leaf=1.0)
    return options


def _fit_lgbm(regression, options, X, y, weight=None, eval_set=None, eval_weight=None, rounds=None):
    import lightgbm as lgb
    model_type = lgb.LGBMRegressor if regression else lgb.LGBMClassifier
    options = dict(options)
    if rounds is not None:
        options['n_estimators'] = max(1, int(rounds))
    model = model_type(**options)
    if eval_set is None:
        model.fit(X, y, sample_weight=weight)
        return model, options['n_estimators']
    model.fit(X, y, sample_weight=weight, eval_set=[eval_set], eval_sample_weight=[eval_weight] if eval_weight is not None else None,
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model, max(1, int(model.best_iteration_ or options['n_estimators']))


def _loss(objective, truth, prediction):
    """Inner selection loss on the scale the learner was given (log scale under a log transform)."""
    from agent.metrics import ABSOLUTE
    if objective == ABSOLUTE:
        return float(np.mean(np.abs(truth-prediction)))
    return float(np.sqrt(np.mean((truth-prediction)**2)))


def _row_groups(X):
    """Exact duplicate feature rows share a group, so inner checks cannot memorize them."""
    return pd.util.hash_pandas_object(X.astype(str), index=False).to_numpy()


def _inner_split(X, y, policy, regression, seed):
    """One inner holdout mirroring the frozen outer protocol, from fit rows only."""
    strategy = (policy or {}).get('strategy')
    column = (policy or {}).get('column')
    if strategy == 'temporal_holdout' and column in X.columns:
        period = ordered_values(X[column])
        if period is not None and not period.isna().any() and period.nunique() >= 2:
            last = period.max()
            valid = np.flatnonzero((period == last).to_numpy())
            fit = np.flatnonzero((period < last).to_numpy())
            if len(fit) and len(valid):
                return fit, valid
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    groups = (X[column].astype(str).to_numpy() if strategy == 'group_holdout' and column in X.columns else _row_groups(X))
    if len(np.unique(groups)) < 6:
        groups = np.arange(len(X))
    if regression:
        splitter = GroupKFold(6)
        fit, valid = next(splitter.split(X, y, groups))
    else:
        labels = np.asarray(y).astype(str)
        least = pd.Series(labels).value_counts().min()
        if least >= 2:
            fit, valid = next(StratifiedGroupKFold(6, shuffle=True, random_state=seed).split(X, labels, groups))
        else:
            fit, valid = next(GroupKFold(6).split(X, labels, groups))
    return np.asarray(fit), np.asarray(valid)


def _tune_offsets(probabilities, truth, classes, scorer):
    """Coordinate search of per-class log offsets maximizing scorer(truth, labels)."""
    logp = np.log(np.clip(probabilities, 1e-12, 1))
    offsets = np.zeros(len(classes))
    def score(values):
        return float(scorer(truth, classes[(logp+values).argmax(axis=1)]))
    best = score(offsets)
    for _ in range(3):
        changed = False
        for index in range(len(classes)):
            for value in OFFSET_GRID:
                trial = offsets.copy()
                trial[index] = value
                candidate = score(trial)
                if candidate > best+1e-12:
                    best, offsets, changed = candidate, trial, True
        if not changed:
            break
    return offsets, best, score


class StrongTabularModel(BaseEstimator):
    def __init__(self, task_spec=None, parameters=None, cpu_threads=1, seed=42, validation_policy=None):
        self.task_spec, self.parameters, self.cpu_threads, self.seed = task_spec, parameters, cpu_threads, seed
        self.validation_policy = validation_policy

    # ---- shared -------------------------------------------------------------
    def _settings(self):
        regression = self.task_spec['task_kind'] == 'regression'
        defaults = {'learning_rate': 0.05, 'num_leaves': 63 if regression else 31, 'min_child_samples': 20 if regression else 30,
                    'colsample_bytree': 0.7, 'reg_lambda': 1.0 if regression else 5.0, 'class_weight_power': 0.5,
                    'bags': 3}
        return regression, {**defaults, **(self.parameters or {})}

    def _bagged(self, options, X, y, rounds, weight=None):
        """Final models on every fit row with the inner round count, one per seed."""
        regression = self.task_spec['task_kind'] == 'regression'
        models = []
        for bag in range(max(1, int(self.bags_))):
            seeded = {**options, 'random_state': int(self.seed)+1009*bag}
            models.append(_fit_lgbm(regression, seeded, X, y, weight=weight, rounds=rounds)[0])
        return models

    def fit(self, X, y):
        started = time.monotonic()
        regression, params = self._settings()
        self.columns_ = list(X.columns)
        self.bags_ = max(1, int(params['bags']))
        self.encoder_ = TableEncoder().fit(X)
        self.report_ = {'operator': 'strong_tabular_model', 'fit_rows': len(X), 'policy': deepcopy(self.validation_policy)}
        if regression:
            self._fit_regression(X, np.asarray(y, dtype=float), params)
        else:
            self._fit_classification(X, np.asarray(y).astype(str), params)
        self.report_['fit_seconds'] = round(time.monotonic()-started, 3)
        return self

    # ---- regression ---------------------------------------------------------
    def _history_frame(self, history, encoded, X):
        return pd.concat([encoded, history.transform(X)], axis=1)

    def _fit_regression(self, X, y, params):
        from agent.metrics import ABSOLUTE, regression_objective
        metric = regression_objective(self.task_spec)
        options = _lgbm_parameters(params, True, self.cpu_threads, self.seed)
        if metric == ABSOLUTE:
            options['objective'] = 'l1'
        encoded = self.encoder_.transform(X)
        fit, valid = _inner_split(X, y, self.validation_policy, True, self.seed)
        base, base_rounds = _fit_lgbm(True, options, encoded.iloc[fit], y[fit], eval_set=(encoded.iloc[valid], y[valid]))
        base_prediction = base.predict(encoded.iloc[valid])
        base_loss = _loss(metric, y[valid], base_prediction)
        self.report_.update(inner_fit_rows=len(fit), inner_valid_rows=len(valid), base_rounds=base_rounds, inner_base_loss=base_loss)
        self.history_ = None
        policy = self.validation_policy or {}
        entity = None
        if policy.get('strategy') == 'temporal_holdout' and policy.get('column') in X.columns:
            # The key is a structural fact of the model's own fit rows; history values
            # for inner-valid rows still come only from strictly earlier periods.
            entity = discover_entity(X, y, policy['column'])
        self.report_['entity'] = entity
        self.mode_ = 'base'
        if entity:
            inner_history = EntityHistory(entity['columns'], entity['period_column']).fit(X.iloc[fit], y[fit])
            fit_frame = self._history_frame(inner_history, encoded.iloc[fit], X.iloc[fit])
            valid_frame = self._history_frame(inner_history, encoded.iloc[valid], X.iloc[valid])
            covered_fit = fit_frame['__history_count'].to_numpy() > 0
            covered_valid = valid_frame['__history_count'].to_numpy() > 0
            if covered_fit.sum() >= 200 and covered_valid.sum() >= 50:
                joint, joint_rounds = _fit_lgbm(True, options, fit_frame, y[fit], eval_set=(valid_frame, y[valid]))
                joint_loss = _loss(metric, y[valid], joint.predict(valid_frame))
                choice = 'joint' if joint_loss < base_loss*(1-HISTORY_MIN_GAIN) else 'base'
                self.mode_ = choice
                self.report_.update(inner_losses={'base': base_loss, 'joint': joint_loss}, history_mode=choice,
                                    joint_rounds=joint_rounds, inner_valid_covered_share=float(covered_valid.mean()))
                if choice == 'joint':
                    self.history_ = EntityHistory(entity['columns'], entity['period_column']).fit(X, y)
                    self.bag_models_ = self._bagged(options, self._history_frame(self.history_, encoded, X), y,
                                                round(joint_rounds*1.1))
        if self.mode_ == 'base':
            self.bag_models_ = self._bagged(options, encoded, y, round(base_rounds*1.1))
        self.report_['bags'] = len(self.bag_models_)

    # ---- classification -----------------------------------------------------
    def _weights(self, labels, power):
        counts = pd.Series(labels).value_counts()
        weights = (len(labels)/(len(counts)*counts))**float(power)
        return pd.Series(labels).map(weights).to_numpy(dtype=float)

    def _fit_classification(self, X, labels, params):
        from agent.metrics import class_balanced, label_score, tunes_decisions
        self.classes_ = np.asarray(sorted(set(labels)), dtype=str)
        options = _lgbm_parameters(params, False, self.cpu_threads, self.seed)
        encoded = self.encoder_.transform(X)
        power = float(params['class_weight_power']) if class_balanced(self.task_spec) else 0.0
        weight = self._weights(labels, power)
        fit, valid = _inner_split(X, labels, self.validation_policy, False, self.seed)
        model, rounds = _fit_lgbm(False, options, encoded.iloc[fit], labels[fit], weight=weight[fit],
                                  eval_set=(encoded.iloc[valid], labels[valid]), eval_weight=weight[valid])
        self.offsets_ = np.zeros(len(self.classes_))
        self.report_.update(inner_fit_rows=len(fit), inner_valid_rows=len(valid), rounds=rounds, class_weight_power=power)
        if tunes_decisions(self.task_spec) and len(valid) >= 40:
            # Decisions are tuned for the declared label metric on one inner half and
            # kept only when they also improve it on the other half.
            metric = self.task_spec['metric']
            vocabulary = [str(label) for label in (self.task_spec.get('allowed_labels') or self.classes_)]
            def scorer(truth, predicted):
                return label_score(metric, vocabulary, truth, predicted)
            probabilities = self._aligned(model, encoded.iloc[valid])
            truth = labels[valid]
            half = np.random.RandomState(self.seed).rand(len(valid)) < 0.5
            offsets, tuned, _ = _tune_offsets(probabilities[half], truth[half], self.classes_, scorer)
            logp = np.log(np.clip(probabilities[~half], 1e-12, 1))
            def confirm(values):
                return scorer(truth[~half], self.classes_[(logp+values).argmax(axis=1)])
            before, after = confirm(np.zeros(len(self.classes_))), confirm(offsets)
            accepted = after >= before+DECISION_MIN_GAIN
            if accepted:
                self.offsets_, _, _ = _tune_offsets(probabilities, truth, self.classes_, scorer)
            self.report_['decision'] = {'metric_id': metric['metric_id'], 'tuning_half_score': tuned, 'confirm_before': before,
                                        'confirm_after': after, 'accepted': bool(accepted), 'offsets': self.offsets_.tolist()}
        self.bag_models_ = self._bagged(options, encoded, labels, round(rounds*1.1), weight=weight)
        self.report_['bags'] = len(self.bag_models_)

    def _aligned(self, model, frame):
        raw = np.asarray(model.predict_proba(frame), dtype=float)
        result = np.zeros((len(frame), len(self.classes_)))
        for index, label in enumerate(model.classes_):
            result[:, int(np.flatnonzero(self.classes_ == str(label))[0])] = raw[:, index]
        return result

    # ---- inference ----------------------------------------------------------
    def predict_proba(self, X):
        if self.task_spec['task_kind'] == 'regression':
            raise TypeError('Regression has no class probabilities')
        encoded = self.encoder_.transform(X)
        return np.mean([self._aligned(model, encoded) for model in self.bag_models_], axis=0)

    def predict(self, X):
        if self.task_spec['task_kind'] != 'regression':
            logp = np.log(np.clip(self.predict_proba(X), 1e-12, 1))
            return self.classes_[(logp+self.offsets_).argmax(axis=1)]
        frame = self.encoder_.transform(X)
        if self.mode_ == 'joint':
            frame = self._history_frame(self.history_, frame, X)
        return np.mean([np.asarray(model.predict(frame), dtype=float) for model in self.bag_models_], axis=0)


def make_strong_model(task_spec, params):
    options = dict(params)
    threads = int(options.pop('_cpu_threads', 1))
    seed = int(options.pop('_seed', 42))
    return StrongTabularModel(task_spec=task_spec, parameters=options, cpu_threads=threads, seed=seed)
