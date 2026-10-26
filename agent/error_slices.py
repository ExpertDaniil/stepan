"""Error slices: where the leader's fit-only OOF loss concentrates and whether it is fixable.

Single-column groups rank large populations first ("State X: 74% of SSE" when it
holds 84% of rows). Here a shallow tree over raw inputs and model/entity meta
finds rule-defined slices, ranked by EXCESS loss (loss share minus row share).
Each slice is then tested, cross-fitted over the OOF folds:

* fixable    – a frozen correction (agent.slice_actions) improves the declared
               metric on held-out folds; the remedy is ready to execute;
* explainable – the slice residual is predictable from inputs (feature work);
* noise      – neither: do not spend research on it.

Inputs are fit-partition rows only (inner OOF); no selection/confirmation label
is read. Parameters of a remedy are estimated on all OOF slice rows and frozen;
the Evaluator later measures the corrected pipeline on selection rows.
"""
from copy import deepcopy
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd

from agent import metrics as registry
from agent import slice_actions as actions

VERSION = 'error-slices-1'
MAX_DEPTH = 3
MIN_LEAF_FRACTION = 0.01
MIN_LEAF_ROWS = 20
MIN_EXCESS = 0.02            # loss share minus row share
MIN_LIFT = 1.2
MAX_SLICES = 5
OVERLAP_LIMIT = 0.7          # Jaccard; overlapping candidates keep the larger excess
MIN_RELATIVE_GAIN = 0.001    # remedy must improve the declared metric by 0.1% of its value
EXPLAINABLE_R2 = 0.05
EXPLAINABLE_AUC = 0.65
WEIGHTS = tuple(round(0.1*i, 1) for i in range(1, 11))
TOP_CATEGORIES = 50
EXAMPLES = 8


def _jsonable(value):
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


# --------------------------------------------------------------------------- loss

class Problem:
    """Aligned truth/prediction arrays and the declared metric's per-row loss."""
    def __init__(self, spec, truth, prediction):
        self.spec = spec
        self.regression = spec['task_kind'] == 'regression'
        self.kind = spec['metric']['prediction_kind']
        self.labels = [str(label) for label in spec.get('allowed_labels') or []]
        if self.regression:
            self.y = np.asarray(truth, dtype=float).reshape(-1)
            self.p = np.asarray(prediction, dtype=float).reshape(len(self.y), -1)[:, 0]
            objective = registry.regression_objective(spec)
            self.scale = 'log1p' if objective == registry.LOG_SQUARED else 'identity'
            self.absolute = objective == registry.ABSOLUTE
            fy, fp = actions.forward(self.y, self.scale), actions.forward(self.p, self.scale)
            self.loss = np.abs(fy-fp) if self.absolute else (fy-fp)**2
            self.predicted = None
            self.probabilities = None
        else:
            self.y = np.asarray(truth).reshape(-1).astype(str)
            self.scale = 'identity'
            if self.kind == 'probabilities':
                self.probabilities = np.asarray(prediction, dtype=float)
                self.predicted = np.asarray(self.labels)[self.probabilities.argmax(axis=1)]
                index = np.asarray([self.labels.index(v) for v in self.y])
                self.loss = -np.log(np.clip(self.probabilities[np.arange(len(self.y)), index], 1e-15, 1))
            else:
                self.probabilities = None
                self.predicted = np.asarray(prediction).reshape(-1).astype(str)
                miss = (self.predicted != self.y).astype(float)
                if registry.class_balanced(spec):
                    counts = pd.Series(self.y).value_counts()
                    weight = pd.Series(self.y).map(len(self.y)/(len(counts)*counts)).to_numpy(dtype=float)
                    self.loss = miss*weight
                else:
                    self.loss = miss

    def score(self, rows=None, prediction=None):
        rows = np.arange(len(self.y)) if rows is None else rows
        if self.regression:
            values = self.p if prediction is None else prediction
            return registry.score(self.spec, self.y[rows], np.asarray(values)[rows].reshape(-1, 1))
        if self.kind == 'probabilities':
            values = self.probabilities if prediction is None else prediction
            return registry.score(self.spec, self.y[rows], np.asarray(values)[rows])
        values = self.predicted if prediction is None else prediction
        return registry.score(self.spec, self.y[rows], np.asarray(values)[rows])

    def gain(self, before, after):
        """Positive when ``after`` is better under the declared direction."""
        return (before-after) if self.spec['metric']['direction'] == 'minimize' else (after-before)


# --------------------------------------------------------------------------- mining

def _design(data, meta, excluded):
    """Numeric design matrix for the miner, plus how to translate splits back to rules."""
    columns, encoders = {}, {}
    for column in data.columns:
        if column in excluded:
            continue
        values = data[column]
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            columns[column] = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            continue
        keys = values.astype('object').where(values.notna(), None).map(lambda v: None if v is None else str(v))
        if keys.nunique(dropna=False) > max(TOP_CATEGORIES*4, 0.5*len(keys)):
            continue  # identifier-like text cannot define a reusable slice
        encoders[column] = keys
    for column in meta.columns:
        values = meta[column]
        if pd.api.types.is_numeric_dtype(values):
            columns[column] = values.to_numpy(dtype=float)
        else:
            encoders[column] = values.astype('object').map(lambda v: None if v is None else str(v))
    return columns, encoders


def _encode_categories(encoders, loss):
    """Order categories by mean loss so one threshold separates hard from easy levels."""
    numeric, translations = {}, {}
    for column, keys in encoders.items():
        frequency = keys.value_counts(dropna=False)
        kept = set(frequency.index[:TOP_CATEGORIES])
        grouped = keys.map(lambda k: k if k in kept else '\x00other')
        order = pd.Series(loss).groupby(grouped.fillna('\x00missing').to_numpy()).mean().sort_values()
        rank = {key: float(i) for i, key in enumerate(order.index)}
        numeric[column] = grouped.fillna('\x00missing').map(rank).to_numpy(dtype=float)
        translations[column] = [None if key == '\x00missing' else key for key in order.index]
    return numeric, translations


def _side_condition(column, threshold, left, translations, missing_left):
    if column in translations:
        ordered = translations[column]
        cut = int(math.floor(threshold))
        chosen = ordered[:cut+1] if left else ordered[cut+1:]
        other = ordered[cut+1:] if left else ordered[:cut+1]
        if '\x00other' in chosen:
            return {'column': column, 'op': 'not_in', 'value': [v for v in other if v != '\x00other'][:actions.MAX_SET_VALUES]}
        return {'column': column, 'op': 'in', 'value': [v for v in chosen if v != '\x00other'][:actions.MAX_SET_VALUES]}
    condition = {'column': column, 'op': 'le' if left else 'gt', 'value': float(threshold)}
    if (missing_left and left) or (not missing_left and not left):
        condition['missing'] = True
    return condition


def mine(problem, data, meta, fold, excluded, seed=42, deadline=None):
    from sklearn.tree import DecisionTreeRegressor
    numeric, encoders = _design(data, meta, excluded)
    encoded, translations = _encode_categories(encoders, problem.loss)
    names = list(numeric)+list(encoded)
    if not names:
        return []
    matrix = np.column_stack([numeric[n] for n in numeric]+[encoded[n] for n in encoded])
    n = len(problem.loss)
    leaf = max(MIN_LEAF_ROWS, int(math.ceil(MIN_LEAF_FRACTION*n)))
    if n < 2*leaf or float(problem.loss.sum()) <= 0:
        return []
    tree = DecisionTreeRegressor(max_depth=MAX_DEPTH, min_samples_leaf=leaf, random_state=seed).fit(matrix, problem.loss)
    structure = tree.tree_
    # Membership is recomputed from the translated rule, exactly as a deployed
    # SliceCorrectionModel will evaluate it, never from the tree's own routing.
    frame = actions.rule_frame(data, meta)
    missing_left = getattr(structure, 'missing_go_to_left', np.zeros(structure.node_count, dtype=bool))
    conditions = {0: []}
    for node in range(structure.node_count):
        left, right = structure.children_left[node], structure.children_right[node]
        if left == -1:
            continue
        column = names[structure.feature[node]]
        threshold = structure.threshold[node]
        conditions[left] = simplify(conditions[node]+[_side_condition(column, threshold, True, translations, bool(missing_left[node]))])
        conditions[right] = simplify(conditions[node]+[_side_condition(column, threshold, False, translations, bool(missing_left[node]))])
    total, folds = float(problem.loss.sum()), np.asarray(fold)
    candidates = []
    for node, rule in conditions.items():
        if node == 0:
            continue
        mask = actions.rule_mask(frame, rule)
        rows = int(mask.sum())
        if rows < leaf:
            continue
        row_share, loss_share = rows/n, float(problem.loss[mask].sum())/total
        excess, lift = loss_share-row_share, loss_share/max(row_share, 1e-12)
        if excess < MIN_EXCESS or lift < MIN_LIFT:
            continue
        per_fold = []
        for f in np.unique(folds):
            inside = mask & (folds == f)
            if inside.any() and problem.loss[folds == f].sum() > 0:
                per_fold.append(float(problem.loss[inside].sum()/problem.loss[folds == f].sum())/(inside.sum()/(folds == f).sum()))
        candidates.append({'node': int(node), 'depth': len(rule), 'conditions': rule, 'mask': mask, 'rows': rows,
                           'row_share': row_share, 'loss_share': loss_share, 'excess': excess, 'lift': lift,
                           'fold_lifts': per_fold})
    candidates.sort(key=lambda c: (-c['excess'], c['depth']))
    chosen = []
    for candidate in candidates:
        # Overlap coefficient: a slice nested in (or containing) a chosen one adds no new region.
        if all((c['mask'] & candidate['mask']).sum()/min(c['mask'].sum(), candidate['mask'].sum()) <= OVERLAP_LIMIT for c in chosen):
            chosen.append(candidate)
        if len(chosen) == MAX_SLICES:
            break
    return chosen


def simplify(conditions):
    """Merge numeric bounds per column; a missing value satisfies the merged rule only if every bound allowed it."""
    merged, order = {}, []
    result = []
    for condition in conditions:
        if condition['op'] not in actions.NUMERIC_OPS:
            result.append(condition)
            continue
        column = condition['column']
        if column not in merged:
            merged[column] = {'le': None, 'gt': None, 'missing': True}
            order.append(column)
        bounds = merged[column]
        if condition['op'] == 'le':
            bounds['le'] = condition['value'] if bounds['le'] is None else min(bounds['le'], condition['value'])
        else:
            bounds['gt'] = condition['value'] if bounds['gt'] is None else max(bounds['gt'], condition['value'])
        bounds['missing'] = bounds['missing'] and bool(condition.get('missing', False))
    for column in order:
        bounds = merged[column]
        for op in ('gt', 'le'):
            if bounds[op] is not None:
                condition = {'column': column, 'op': op, 'value': float(bounds[op])}
                if bounds['missing']:
                    condition['missing'] = True
                result.append(condition)
    return result


# --------------------------------------------------------------------------- remedies

def _fold_pairs(fold):
    folds = np.asarray(fold)
    values = np.unique(folds)
    if len(values) < 2:
        return []
    return [(np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in values]


def _fit_regression(problem, action, rows, history_last):
    """Frozen params for ``action`` estimated on ``rows`` (slice rows of the estimation folds)."""
    fy = actions.forward(problem.y[rows], problem.scale)
    fp = actions.forward(problem.p[rows], problem.scale)
    if action == 'shift':
        residual = fy-fp
        value = float(np.median(residual) if problem.absolute else np.mean(residual))
        return {'value': value} if math.isfinite(value) and value != 0 else None
    def loss(values):
        return float(np.mean(np.abs(fy-values)) if problem.absolute else np.mean((fy-values)**2))
    if action == 'shrink':
        center = float(np.median(fy) if problem.absolute else np.mean(fy))
        best = min(WEIGHTS, key=lambda w: loss((1-w)*fp+w*center))
        return {'weight': best, 'center': center} if loss((1-best)*fp+best*center) < loss(fp) else None
    if action == 'blend_history':
        last = np.asarray(history_last, dtype=float)[rows]
        usable = np.isfinite(last)
        if usable.sum() < 10:
            return None
        fh = actions.forward(last, problem.scale)
        def blended(w):
            values = fp.copy()
            values[usable] = (1-w)*fp[usable]+w*fh[usable]
            return values
        best = min(WEIGHTS, key=lambda w: loss(blended(w)))
        return {'weight': best} if loss(blended(best)) < loss(fp) else None
    raise ValueError(action)


def _fit_labels(problem, mask, rows):
    """Greedy predicted-class remapping inside the slice, kept only if the declared metric improves."""
    inside = rows[mask[rows]]
    if not len(inside):
        return None
    mapping = {}
    current = problem.predicted.copy()
    base = problem.score(rows, current)
    for label in pd.Series(problem.predicted[inside]).value_counts().index:
        hits = inside[problem.predicted[inside] == label]
        counts = pd.Series(problem.y[hits]).value_counts()
        target = str(counts.index[0])
        if target == label:
            continue
        trial = current.copy()
        trial[hits] = target
        score = problem.score(rows, trial)
        if problem.gain(base, score) > 0:
            mapping[str(label)], current, base = target, trial, score
    return {'mapping': mapping} if mapping else None


def _fit_probabilities(problem, mask, rows):
    inside = rows[mask[rows]]
    if len(inside) < 10:
        return None
    observed = np.asarray([(problem.y[inside] == label).mean() for label in problem.labels])
    predicted = problem.probabilities[inside].mean(axis=0)
    ratios = {}
    for label, o, p in zip(problem.labels, observed, predicted):
        if p > 1e-6:
            ratio = float(np.clip(o/p, *actions.RATIO_BOUNDS)) if o > 0 else actions.RATIO_BOUNDS[0]
            if abs(ratio-1) > 1e-3:
                ratios[label] = round(ratio, 6)
    return {'ratios': ratios} if ratios else None


def _apply(problem, action, params, mask, history_last):
    if problem.regression:
        return actions.apply_regression(problem.p, mask, action, params, problem.scale, history_last)
    if action == 'remap':
        return actions.apply_labels(problem.predicted, mask, params)
    return actions.apply_probabilities(problem.probabilities, mask, params, problem.labels)


def _base_output(problem):
    return problem.p if problem.regression else problem.probabilities if problem.kind == 'probabilities' else problem.predicted


def evaluate_remedy(problem, action, mask, fold, history_last=None):
    """Cross-fitted effect of one frozen correction on the declared metric over all OOF rows."""
    pairs = _fold_pairs(fold)
    if not pairs:
        return None
    corrected = np.array(_base_output(problem), copy=True)
    fold_gains = []
    for estimate, apply in pairs:
        rows = estimate[mask[estimate]]
        if problem.regression:
            params = _fit_regression(problem, action, rows, history_last) if len(rows) >= 10 else None
        elif action == 'remap':
            params = _fit_labels(problem, mask, estimate)
        else:
            params = _fit_probabilities(problem, mask, estimate)
        local = np.zeros(len(mask), dtype=bool)
        local[apply] = mask[apply]
        if params is not None:
            updated = _apply(problem, action, params, local, history_last)
            corrected[local] = updated[local]
        try:
            fold_gains.append(problem.gain(problem.score(apply), problem.score(apply, corrected)))
        except ValueError:
            fold_gains.append(0.0)
    base, after = problem.score(), problem.score(prediction=corrected)
    all_rows = np.arange(len(mask))
    if problem.regression:
        final = _fit_regression(problem, action, all_rows[mask], history_last)
    elif action == 'remap':
        final = _fit_labels(problem, mask, all_rows)
    else:
        final = _fit_probabilities(problem, mask, all_rows)
    gain = problem.gain(base, after)
    required = max(2, math.ceil(2*len(pairs)/3)) if len(pairs) > 2 else len(pairs)
    improved = sum(g > 0 for g in fold_gains)
    return {'action': action, 'params': final, 'oof_metric_before': float(base), 'oof_metric_after': float(after),
            'oof_gain': float(gain), 'relative_gain': float(gain/max(abs(base), 1e-12)),
            'folds_improved': int(improved), 'folds': len(pairs),
            'robust': bool(final is not None and gain > 0 and gain/max(abs(base), 1e-12) >= MIN_RELATIVE_GAIN and improved >= required)}


def explainability(problem, data, meta, mask, fold, excluded, seed=42):
    """Cross-fitted predictability of the slice residual (regression) or error (classification)."""
    import lightgbm as lgb
    rows = np.flatnonzero(mask)
    if len(rows) < 60:
        return None
    frame = data.iloc[rows].drop(columns=[c for c in data.columns if c in excluded]).reset_index(drop=True)
    for column in frame.columns:
        if not (pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])):
            frame[column] = frame[column].astype('object').where(frame[column].notna(), None).astype(str).astype('category')
    for column in meta.columns:
        values = meta[column].iloc[rows].reset_index(drop=True)
        frame[column] = values.astype(str).astype('category') if not pd.api.types.is_numeric_dtype(values) else values
    if problem.regression:
        target = (actions.forward(problem.y, problem.scale)-actions.forward(problem.p, problem.scale))[rows]
        model_type = lgb.LGBMRegressor
    else:
        target = (problem.predicted != problem.y)[rows].astype(int)
        if target.min() == target.max():
            return None
        model_type = lgb.LGBMClassifier
    options = dict(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=max(10, len(rows)//50),
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=2, verbose=-1)
    folds = np.asarray(fold)[rows]
    predicted = np.full(len(rows), np.nan)
    for value in np.unique(folds):
        train, test = folds != value, folds == value
        if train.sum() < 30 or not test.any() or (not problem.regression and len(set(target[train])) < 2):
            continue
        model = model_type(**options).fit(frame[train], target[train])
        predicted[test] = model.predict(frame[test]) if problem.regression else model.predict_proba(frame[test])[:, 1]
    covered = np.isfinite(predicted)
    if covered.sum() < 30:
        return None
    full = model_type(**options).fit(frame, target)
    gains = full.booster_.feature_importance(importance_type='gain')
    total = float(gains.sum()) or 1.0
    top = [{'feature': str(name), 'gain_share': round(float(g)/total, 4)}
           for name, g in sorted(zip(frame.columns, gains), key=lambda pair: -pair[1])[:5] if g > 0]
    if problem.regression:
        residual = target[covered]
        baseline = float(np.sum((residual-residual.mean())**2)) or 1e-12
        r2 = 1-float(np.sum((residual-predicted[covered])**2))/baseline
        return {'measure': 'residual_r2_oof', 'value': round(r2, 4), 'explainable': r2 >= EXPLAINABLE_R2, 'top_features': top}
    from sklearn.metrics import roc_auc_score
    if len(set(target[covered])) < 2:
        return None
    auc = float(roc_auc_score(target[covered], predicted[covered]))
    return {'measure': 'error_auc_oof', 'value': round(auc, 4), 'explainable': auc >= EXPLAINABLE_AUC, 'top_features': top}


# --------------------------------------------------------------------------- report

def _examples(problem, data, meta, mask, conditions, explain):
    rows = np.flatnonzero(mask)
    worst = rows[np.argsort(-problem.loss[rows])[:EXAMPLES]]
    wanted = list(dict.fromkeys([c['column'] for c in conditions]+[t['feature'] for t in (explain or {}).get('top_features', [])]))
    result = []
    for row in worst:
        record = {}
        for column in wanted[:8]:
            source = meta if column in meta.columns else data if column in data.columns else None
            if source is not None:
                record[column] = _jsonable(source[column].iloc[row])
        record['truth'] = _jsonable(problem.y[row])
        record['prediction'] = _jsonable(problem.p[row] if problem.regression else problem.predicted[row])
        record['loss'] = _jsonable(round(float(problem.loss[row]), 6))
        result.append(record)
    return result


def _metric_text(value):
    return f'{value:.6g}'


def analyze(data, truth, prediction, fold, spec, *, meta_history=None, entity=None, seed=42, deadline=None):
    """Mine slices on fit-only OOF and test remedies. Never raises for analysis problems."""
    started = time.monotonic()
    try:
        problem = Problem(spec, truth, prediction)
    except (ValueError, KeyError, TypeError) as error:
        return {'version': VERSION, 'available': False, 'reason': 'loss_unavailable: '+str(error)[:200], 'slices': []}
    excluded = set(spec.get('excluded_features', []))|set(spec.get('target_columns') or [spec['target_column']])
    meta = actions.meta_frame(_base_output(problem) if problem.regression else problem.predicted, regression=problem.regression,
                              probabilities=problem.probabilities, history=meta_history)
    history_last = meta['__history_last'].to_numpy() if '__history_last' in meta else None
    data = data.reset_index(drop=True)
    try:
        chosen = mine(problem, data, meta, fold, excluded, seed=seed, deadline=deadline)
    except (ValueError, TypeError, MemoryError) as error:
        return {'version': VERSION, 'available': False, 'reason': 'mining_failed: '+str(error)[:200], 'slices': []}
    if history_last is not None and problem.regression:
        # Structural candidate: every row whose entity has earlier labelled periods.
        # Its excess may be zero while a history blend still helps everywhere.
        rule = [{'column': '__history_count', 'op': 'gt', 'value': 0.5}]
        mask = actions.rule_mask(actions.rule_frame(data, meta), rule)
        if mask.sum() >= MIN_LEAF_ROWS and not any((c['mask'] == mask).all() for c in chosen):
            total = float(problem.loss.sum()) or 1.0
            share = float(mask.mean())
            loss_share = float(problem.loss[mask].sum())/total
            folds = np.asarray(fold)
            lifts = [float(problem.loss[mask & (folds == f)].sum()/max(problem.loss[folds == f].sum(), 1e-12))
                     / max((mask & (folds == f)).sum()/(folds == f).sum(), 1e-12)
                     for f in np.unique(folds) if (mask & (folds == f)).any()]
            chosen.append({'node': -1, 'depth': 1, 'conditions': rule, 'mask': mask, 'rows': int(mask.sum()),
                           'row_share': share, 'loss_share': loss_share, 'excess': loss_share-share,
                           'lift': loss_share/max(share, 1e-12), 'fold_lifts': lifts, 'structural': True})
    base = float(problem.score())
    menu = (actions.REGRESSION_ACTIONS if problem.regression else
            actions.PROBABILITY_ACTIONS if problem.kind == 'probabilities' else actions.LABEL_ACTIONS)
    slices = []
    for index, candidate in enumerate(chosen, 1):
        if deadline is not None and time.monotonic() > deadline:
            break
        mask = candidate['mask']
        remedies = []
        for action in menu:
            if action == 'blend_history' and history_last is None:
                continue
            try:
                result = evaluate_remedy(problem, action, mask, fold, history_last)
            except (ValueError, TypeError, ZeroDivisionError):
                result = None
            if result:
                remedies.append(result)
        remedies.sort(key=lambda r: (-r['robust'], -r['oof_gain']))
        try:
            explain = explainability(problem, data, meta, mask, fold, excluded, seed=seed) if not (
                deadline is not None and time.monotonic() > deadline) else None
        except (ValueError, TypeError, MemoryError) as error:
            explain = {'measure': 'unavailable', 'reason': str(error)[:200], 'explainable': False, 'top_features': []}
        best = remedies[0] if remedies and remedies[0]['robust'] else None
        verdict = 'fixable' if best else 'explainable' if explain and explain.get('explainable') else 'noise'
        if candidate.get('structural') and not best:
            continue  # A structural population is reported only when a remedy works on it.
        uses_history = any(c['column'].startswith(actions.HISTORY_PREFIX) for c in candidate['conditions'])
        slice_id = 'H1' if candidate.get('structural') else f'S{index}'
        record = {'slice_id': slice_id, 'rule': actions.describe(candidate['conditions']),
                  'conditions': candidate['conditions'], 'entity': entity if (uses_history or any(
                      r['action'] == 'blend_history' for r in remedies)) else None,
                  'rows': candidate['rows'], 'row_share': round(candidate['row_share'], 6),
                  'loss_share': round(candidate['loss_share'], 6), 'excess': round(candidate['excess'], 6),
                  'lift': round(float(candidate['lift']), 4), 'fold_lifts': [round(float(v), 4) for v in candidate['fold_lifts']],
                  'consistent_folds': int(sum(bool(v > 1) for v in candidate['fold_lifts'])),
                  'verdict': verdict, 'remedies': [], 'explainability': explain,
                  'examples': _examples(problem, data, meta, mask, candidate['conditions'], explain)}
        if problem.regression:
            fy, fp = actions.forward(problem.y[mask], problem.scale), actions.forward(problem.p[mask], problem.scale)
            record['bias'] = round(float(np.mean(fp-fy)), 6)
            record['bias_scale'] = problem.scale
        else:
            pairs = pd.DataFrame({'truth': problem.y[mask], 'prediction': problem.predicted[mask]})
            confused = pairs[pairs.truth != pairs.prediction].value_counts().head(5)
            record['confusions'] = [{'truth': str(t), 'prediction': str(p), 'rows': int(c)} for (t, p), c in confused.items()]
        for remedy in remedies:
            if remedy['params'] is None:
                continue
            correction = {'slice_id': slice_id, 'conditions': deepcopy(candidate['conditions']), 'action': remedy['action'],
                          'params': remedy['params'], 'scale': problem.scale,
                          'entity': deepcopy(entity) if (uses_history or remedy['action'] == 'blend_history') else None}
            record['remedies'].append({'remedy_id': slice_id+'/'+remedy['action'], **{k: remedy[k] for k in (
                'action', 'oof_metric_before', 'oof_metric_after', 'oof_gain', 'relative_gain', 'folds_improved', 'folds', 'robust')},
                'correction': correction})
        record['journal'] = journal_line(record, spec)
        slices.append(record)
    def order(record):
        robust = [r for r in record['remedies'] if r['robust']]
        return ({'fixable': 0, 'explainable': 1, 'noise': 2}[record['verdict']],
                -(robust[0]['relative_gain'] if robust else record['excess']))
    slices.sort(key=order)
    return plain({'version': VERSION, 'available': True, 'metric_id': spec['metric']['metric_id'],
            'oof_metric': base, 'loss_measure': ('absolute' if problem.regression and problem.absolute else
                                                 'squared' if problem.regression else
                                                 'log_loss' if problem.kind == 'probabilities' else
                                                 'class_weighted_miss' if registry.class_balanced(spec) else 'miss'),
            'loss_scale': problem.scale, 'rows': len(problem.loss), 'history_meta': history_last is not None,
            'entity': entity, 'slices': slices, 'seconds': round(time.monotonic()-started, 3),
            'interpretation': ('Slices are rules over inputs and the base model/entity meta, ranked by loss share minus row share. '
                               'Remedies are frozen corrections cross-fitted over OOF folds; robust means held-out folds improved. '
                               'Selection-row effects are measured only after a remedy is executed as a candidate.')})


def plain(value):
    """JSON-safe copy: NumPy scalars become Python numbers, non-finite floats become null."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return _jsonable(value)


def journal_line(record, spec):
    metric = spec['metric']['metric_id']
    text = (f"{record['slice_id']} [{record['rule']}]: {100*record['row_share']:.1f}% rows -> "
            f"{100*record['loss_share']:.1f}% loss (x{record['lift']:.1f}, folds {record['consistent_folds']}/{len(record['fold_lifts'])})")
    robust = [r for r in record['remedies'] if r['robust']]
    if robust:
        best = robust[0]
        text += (f"; {best['action']} OOF {metric} {_metric_text(best['oof_metric_before'])}->{_metric_text(best['oof_metric_after'])}"
                 f" ({best['folds_improved']}/{best['folds']} folds)")
    explain = record.get('explainability') or {}
    if explain.get('measure') in {'residual_r2_oof', 'error_auc_oof'}:
        features = ', '.join(t['feature'] for t in explain.get('top_features', [])[:3])
        text += f"; {explain['measure']}={explain['value']}" + (f" via {features}" if features else '')
    return text+f" => {record['verdict']}"


def compact_card(record, source_artifact_id=None, limit_examples=4):
    """What the researcher reads: rule, numbers, verdict, executable remedy references. Idempotent."""
    if any('remedy_ref' in r for r in record.get('remedies', [])) or ('remedies' in record and 'examples' in record
                                                                    and 'conditions' not in record):
        card = deepcopy(record)
        card['examples'] = card.get('examples', [])[:limit_examples]
        return card
    card = {k: deepcopy(record[k]) for k in ('slice_id', 'rule', 'rows', 'row_share', 'loss_share', 'excess', 'lift',
                                            'consistent_folds', 'verdict', 'journal') if k in record}
    for key in ('bias', 'bias_scale', 'confusions'):
        if key in record:
            card[key] = deepcopy(record[key])
    card['remedies'] = [{'remedy_ref': (source_artifact_id+'#' if source_artifact_id else '')+r['remedy_id'],
                         **{k: r[k] for k in ('action', 'oof_metric_before', 'oof_metric_after', 'relative_gain',
                                              'folds_improved', 'folds', 'robust')},
                         'params': deepcopy(r['correction']['params'])} for r in record.get('remedies', [])[:3]]
    explain = record.get('explainability')
    if explain:
        card['explainability'] = {k: deepcopy(explain.get(k)) for k in ('measure', 'value', 'explainable', 'top_features')}
    card['examples'] = deepcopy(record.get('examples', [])[:limit_examples])
    return card


def membership_sha256(mask, index):
    return hashlib.sha256(json.dumps(sorted(map(str, np.asarray(index)[np.asarray(mask, dtype=bool)]))).encode()).hexdigest()


def resolve_remedy(reference, artifacts, spec, columns=None):
    """``<fit_diagnostics artifact>#<slice>/<action>`` -> validated correction and the base it was measured on.

    The correction is relabelled with the full reference, so every executed
    correction names its immutable source report.
    """
    from agent.errors import AgentError
    if not isinstance(reference, str) or '#' not in reference:
        raise AgentError('INVALID_SLICE_REMEDY', 'Use remedy_ref exactly as shown: <artifact_id>#<slice_id>/<action>')
    artifact_id, local = reference.split('#', 1)
    try:
        meta = artifacts.meta(artifact_id)
    except (KeyError, AgentError) as error:
        raise AgentError('INVALID_SLICE_REMEDY', 'Unknown diagnostic artifact in '+reference) from error
    if meta.get('kind') != 'fit_diagnostics':
        raise AgentError('INVALID_SLICE_REMEDY', 'A remedy must come from a fit diagnostics report')
    from agent.artifacts import file_sha256
    if file_sha256(artifacts.resolve(artifact_id)) != meta['sha256']:
        raise AgentError('MEMORY_SOURCE_CHANGED', 'Diagnostic report changed')
    report = artifacts.read_json(artifact_id)
    if report.get('task_spec_id') not in (None, spec.get('task_spec_id')):
        raise AgentError('INVALID_SLICE_REMEDY', 'The remedy belongs to another task contract')
    for record in (report.get('error_slices') or {}).get('slices', []):
        for remedy in record.get('remedies', []):
            if remedy['remedy_id'] == local:
                correction = {**deepcopy(remedy['correction']), 'slice_id': reference}
                try:
                    correction = actions.validate_corrections([correction], spec, columns)[0]
                except ValueError as error:
                    raise AgentError('INVALID_SLICE_REMEDY', str(error)) from error
                return {'correction': correction, 'robust': bool(remedy.get('robust')),
                        'base': {'experiment_id': report.get('experiment_id'), 'recipe_id': report.get('source_recipe_id'),
                                 'hyperparameters': deepcopy(report.get('source_hyperparameters')),
                                 'feature_factory': bool(report.get('source_feature_factory'))}}
    raise AgentError('INVALID_SLICE_REMEDY', 'No such remedy in '+artifact_id+': '+local)
