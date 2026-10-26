"""Fit-only error slices and executable corrections checked on a last inner fold.

Rules and correction parameters use discovery folds. The diagnostic worker also
excludes check labels from discovery model fits and entity discovery. A holdout
win is conditional evidence; unresolved means this lab did not establish a
remedy, not that the region is irreducible noise. Selection and final closed
confirmation independently judge the delivered candidate.
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
MIN_HOLDOUT_SLICE_ROWS = 20  # slice rows needed in the holdout fold for a verdict
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
            from agent.significance import row_losses
            self.scale = actions.correction_scale(spec)
            self.absolute = registry.regression_objective(spec) == registry.ABSOLUTE
            # The declared metric's own per-row loss (MAPE: |y-p|/|y|, RMSLE: log1p
            # difference squared); never a log of clipped negatives.
            loss = row_losses(spec, self.y, self.p)
            if loss is None:
                residual = self.y-self.p
                loss = np.abs(residual) if self.absolute else residual**2
            self.loss = np.asarray(loss, dtype=float)
            self._row_loss = row_losses
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

    def row_loss(self, truth, prediction):
        """Per-row declared-metric loss on original units (regression)."""
        loss = self._row_loss(self.spec, truth, prediction)
        if loss is None:
            residual = np.asarray(truth, dtype=float)-np.asarray(prediction, dtype=float)
            loss = np.abs(residual) if self.absolute else residual**2
        return np.asarray(loss, dtype=float)

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

def _fit_regression(problem, action, rows, history_last):
    """Frozen params for ``action`` estimated on ``rows`` (slice rows of the estimation folds)."""
    fy = actions.forward(problem.y[rows], problem.scale)
    fp = actions.forward(problem.p[rows], problem.scale)
    truth = problem.y[rows]
    def loss(values):
        # Candidate parameters are judged by the declared metric's own row loss.
        return float(np.mean(problem.row_loss(truth, actions.inverse(values, problem.scale))))
    if action == 'shift':
        residual = fy-fp
        value = float(np.median(residual) if problem.absolute else np.mean(residual))
        return {'value': value} if math.isfinite(value) and value != 0 and loss(fp+value) < loss(fp) else None
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


def split_roles(fold):
    """Discovery rows (mining and parameter fitting) and one untouched holdout inner fold.

    The holdout is the last fold: the latest period under a temporal protocol.
    Nothing measured on it influenced which slice was chosen or its parameters.
    """
    folds = np.asarray(fold)
    values = np.unique(folds)
    if len(values) < 2:
        return None, None
    check = folds == values.max()
    return np.flatnonzero(~check), np.flatnonzero(check)


def evaluate_remedy(problem, action, mask, discovery, check, history_last=None):
    """Parameters fitted on discovery slice rows; effect measured only on the holdout fold."""
    disc_rows = discovery[mask[discovery]]
    if problem.regression:
        params = _fit_regression(problem, action, disc_rows, history_last) if len(disc_rows) >= 10 else None
    elif action == 'remap':
        params = _fit_labels(problem, mask, discovery)
    else:
        params = _fit_probabilities(problem, mask, discovery)
    result = {'action': action, 'params': params, 'evaluation': 'held_out_inner_fold', 'folds': 1}
    if params is None:
        return {**result, 'robust': False, 'oof_metric_before': None, 'oof_metric_after': None, 'oof_gain': 0.0,
                'relative_gain': 0.0, 'folds_improved': 0, 'holdout_slice_rows': 0, 'discovery_gain': 0.0}
    def effect(rows):
        local = np.zeros(len(mask), dtype=bool)
        local[rows] = mask[rows]
        corrected = np.array(_base_output(problem), copy=True)
        corrected[local] = _apply(problem, action, params, local, history_last)[local]
        before, after = float(problem.score(rows)), float(problem.score(rows, corrected))
        return before, after, int(local.sum())
    try:
        before, after, inside = effect(check)
        discovery_gain = problem.gain(*effect(discovery)[:2])
    except ValueError:  # The declared metric can be undefined on a small holdout.
        return {**result, 'robust': False, 'oof_metric_before': None, 'oof_metric_after': None, 'oof_gain': 0.0,
                'relative_gain': 0.0, 'folds_improved': 0, 'holdout_slice_rows': 0, 'discovery_gain': 0.0}
    gain = problem.gain(before, after)
    relative = gain/max(abs(before), 1e-12)
    return {**result, 'oof_metric_before': before, 'oof_metric_after': after, 'oof_gain': float(gain),
            'relative_gain': float(relative), 'folds_improved': int(gain > 0), 'holdout_slice_rows': inside,
            'discovery_gain': float(discovery_gain),
            'robust': bool(inside >= MIN_HOLDOUT_SLICE_ROWS and gain > 0 and relative >= MIN_RELATIVE_GAIN)}


def explainability(problem, data, meta, mask, discovery, check, excluded, seed=42, cpu_threads=1):
    """Slice residual (regression) or error (classification) predicted on the holdout fold from discovery rows."""
    import lightgbm as lgb
    train_rows, test_rows = discovery[mask[discovery]], check[mask[check]]
    if len(train_rows) < 40 or len(test_rows) < MIN_HOLDOUT_SLICE_ROWS:
        return None
    rows = np.concatenate([train_rows, test_rows])
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
        model_type = lgb.LGBMClassifier
    train = np.arange(len(rows)) < len(train_rows)
    if not problem.regression and (len(set(target[train])) < 2 or len(set(target[~train])) < 2):
        return None
    options = dict(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=max(10, len(train_rows)//50),
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=cpu_threads, verbose=-1)
    model = model_type(**options).fit(frame[train], target[train])
    predicted = model.predict(frame[~train]) if problem.regression else model.predict_proba(frame[~train])[:, 1]
    gains = model.booster_.feature_importance(importance_type='gain')
    total = float(gains.sum()) or 1.0
    top = [{'feature': str(name), 'gain_share': round(float(g)/total, 4)}
           for name, g in sorted(zip(frame.columns, gains), key=lambda pair: -pair[1])[:5] if g > 0]
    if problem.regression:
        residual = target[~train]
        reference = target[train].mean()  # The holdout mean is not available to a deployed model.
        baseline = float(np.sum((residual-reference)**2)) or 1e-12
        r2 = 1-float(np.sum((residual-predicted)**2))/baseline
        return {'measure': 'residual_r2_holdout', 'value': round(r2, 4), 'explainable': r2 >= EXPLAINABLE_R2, 'top_features': top}
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(target[~train], predicted))
    return {'measure': 'error_auc_holdout', 'value': round(auc, 4), 'explainable': auc >= EXPLAINABLE_AUC, 'top_features': top}


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


def analyze(data, truth, prediction, fold, spec, *, meta_history=None, entity=None, seed=42, deadline=None, cpu_threads=1):
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
    folds = np.asarray(fold)
    discovery, check = split_roles(folds)
    holdout = discovery is not None
    if not holdout:
        discovery, check = np.arange(len(folds)), np.arange(0)
    # Slices are mined on discovery rows only, so the holdout fold judges a rule
    # it never helped choose; the full-row mask is recomputed from the rule text.
    full = actions.rule_frame(data, meta)
    try:
        from types import SimpleNamespace
        chosen = mine(SimpleNamespace(loss=problem.loss[discovery]), data.iloc[discovery].reset_index(drop=True),
                      meta.iloc[discovery].reset_index(drop=True), folds[discovery], excluded, seed=seed, deadline=deadline)
    except (ValueError, TypeError, MemoryError) as error:
        return {'version': VERSION, 'available': False, 'reason': 'mining_failed: '+str(error)[:200], 'slices': []}
    for candidate in chosen:
        candidate['mask'] = actions.rule_mask(full, candidate['conditions'])
    if history_last is not None and problem.regression:
        # Structural candidate: every row whose entity has earlier labelled periods.
        # Its excess may be zero while a history blend still helps everywhere.
        rule = [{'column': '__history_count', 'op': 'gt', 'value': 0.5}]
        mask = actions.rule_mask(full, rule)
        if mask[discovery].sum() >= MIN_LEAF_ROWS and not any((c['mask'] == mask).all() for c in chosen):
            loss, inside = problem.loss[discovery], mask[discovery]
            share, loss_share = float(inside.mean()), float(loss[inside].sum())/(float(loss.sum()) or 1.0)
            chosen.append({'node': -1, 'depth': 1, 'conditions': rule, 'mask': mask, 'rows': int(inside.sum()),
                           'row_share': share, 'loss_share': loss_share, 'excess': loss_share-share,
                           'lift': loss_share/max(share, 1e-12), 'fold_lifts': [], 'structural': True})
    base = float(problem.score())
    menu = (actions.REGRESSION_ACTIONS if problem.regression else
            actions.PROBABILITY_ACTIONS if problem.kind == 'probabilities' else actions.LABEL_ACTIONS)
    slices = []
    for index, candidate in enumerate(chosen, 1):
        mask = candidate['mask']
        holdout_rows = int(mask[check].sum())
        holdout_lift = None
        if holdout_rows and float(problem.loss[check].sum()) > 0:
            holdout_lift = (float(problem.loss[check][mask[check]].sum())/float(problem.loss[check].sum()))/(holdout_rows/len(check))
        timed_out = deadline is not None and time.monotonic() > deadline
        remedies, explain = [], None
        if holdout and holdout_rows >= MIN_HOLDOUT_SLICE_ROWS and not timed_out:
            for action in menu:
                if action == 'blend_history' and history_last is None:
                    continue
                try:
                    result = evaluate_remedy(problem, action, mask, discovery, check, history_last)
                except (ValueError, TypeError, ZeroDivisionError):
                    result = None
                if result:
                    remedies.append(result)
            remedies.sort(key=lambda r: (-r['robust'], -r['oof_gain']))
            try:
                explain = explainability(problem, data, meta, mask, discovery, check, excluded, seed=seed, cpu_threads=cpu_threads)
            except (ValueError, TypeError, MemoryError) as error:
                explain = {'measure': 'unavailable', 'reason': str(error)[:200], 'explainable': False, 'top_features': []}
        best = remedies[0] if remedies and remedies[0]['robust'] else None
        verdict = 'fixable' if best else 'explainable' if explain and explain.get('explainable') else 'unresolved'
        if candidate.get('structural') and not best:
            continue  # A structural population is reported only when a remedy works on it.
        # Unresolved is a statement about this lab, not about the data: say why.
        reason = (None if verdict != 'unresolved' else 'no_holdout_fold' if not holdout else 'analysis_deadline' if timed_out
                  else 'too_few_holdout_rows' if holdout_rows < MIN_HOLDOUT_SLICE_ROWS
                  else 'tested_corrections_and_residual_model_found_nothing')
        uses_history = any(c['column'].startswith(actions.HISTORY_PREFIX) for c in candidate['conditions'])
        slice_id = 'H1' if candidate.get('structural') else f'S{index}'
        record = {'slice_id': slice_id, 'rule': actions.describe(candidate['conditions']),
                  'conditions': candidate['conditions'], 'entity': entity if (uses_history or any(
                      r['action'] == 'blend_history' for r in remedies)) else None,
                  'rows': candidate['rows'], 'row_share': round(candidate['row_share'], 6),
                  'loss_share': round(candidate['loss_share'], 6), 'excess': round(candidate['excess'], 6),
                  'lift': round(float(candidate['lift']), 4), 'fold_lifts': [round(float(v), 4) for v in candidate['fold_lifts']],
                  'consistent_folds': int(sum(bool(v > 1) for v in candidate['fold_lifts'])),
                  'holdout_rows': holdout_rows, 'holdout_lift': round(holdout_lift, 4) if holdout_lift is not None else None,
                  'verdict': verdict, 'unresolved_reason': reason, 'remedies': [], 'explainability': explain,
                  'examples': _examples(problem, data, meta, mask, candidate['conditions'], explain)}
        if problem.regression:
            fy, fp = actions.forward(problem.y[mask], problem.scale), actions.forward(problem.p[mask], problem.scale)
            record['bias'] = round(float(np.mean(fp-fy)), 6) if mask.any() else None
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
                'action', 'oof_metric_before', 'oof_metric_after', 'oof_gain', 'relative_gain', 'folds_improved', 'folds',
                'robust', 'evaluation', 'holdout_slice_rows', 'discovery_gain')},
                'correction': correction})
        record['journal'] = journal_line(record, spec)
        slices.append(record)
    def order(record):
        robust = [r for r in record['remedies'] if r['robust']]
        return ({'fixable': 0, 'explainable': 1, 'unresolved': 2}[record['verdict']],
                -(robust[0]['relative_gain'] if robust else record['excess']))
    slices.sort(key=order)
    return plain({'version': VERSION, 'available': True, 'metric_id': spec['metric']['metric_id'],
            'oof_metric': base, 'loss_measure': ('declared_metric_row_loss' if problem.regression else
                                                 'log_loss' if problem.kind == 'probabilities' else
                                                 'class_weighted_miss' if registry.class_balanced(spec) else 'miss'),
            'loss_scale': problem.scale, 'rows': len(problem.loss), 'history_meta': history_last is not None,
            'discovery_rows': int(len(discovery)), 'holdout_rows': int(len(check)),
            'entity': entity, 'slices': slices, 'seconds': round(time.monotonic()-started, 3),
            'interpretation': ('Slices are mined on discovery inner folds and ranked by loss share minus row share there. '
                               'Remedy parameters are fitted on discovery rows; robust means the untouched last inner fold improved. '
                               'Unresolved means this lab found no correction or residual signal, not that the slice is unfixable. '
                               'Selection and closed confirmation rows still judge every executed candidate.')})


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
            f"{100*record['loss_share']:.1f}% loss (x{record['lift']:.1f}"
            + (f", holdout x{record['holdout_lift']:.1f}" if record.get('holdout_lift') is not None else '') + ")")
    robust = [r for r in record['remedies'] if r['robust']]
    if robust:
        best = robust[0]
        text += (f"; {best['action']} holdout {metric} {_metric_text(best['oof_metric_before'])}->{_metric_text(best['oof_metric_after'])}")
    explain = record.get('explainability') or {}
    if explain.get('measure') in {'residual_r2_holdout', 'error_auc_holdout'}:
        features = ', '.join(t['feature'] for t in explain.get('top_features', [])[:3])
        text += f"; {explain['measure']}={explain['value']}" + (f" via {features}" if features else '')
    if record.get('unresolved_reason'):
        text += f" ({record['unresolved_reason']})"
    return text+f" => {record['verdict']}"


def compact_card(record, source_artifact_id=None, limit_examples=4):
    """What the researcher reads: rule, numbers, verdict, executable remedy references. Idempotent."""
    if any('remedy_ref' in r for r in record.get('remedies', [])) or ('remedies' in record and 'examples' in record
                                                                    and 'conditions' not in record):
        card = deepcopy(record)
        card['examples'] = card.get('examples', [])[:limit_examples]
        return card
    card = {k: deepcopy(record[k]) for k in ('slice_id', 'rule', 'rows', 'row_share', 'loss_share', 'excess', 'lift',
                                            'consistent_folds', 'holdout_rows', 'holdout_lift', 'verdict',
                                            'unresolved_reason', 'journal') if k in record}
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
