"""Frozen per-slice prediction corrections, shared by the fit-only lab and executed pipelines.

A correction is a rule (a conjunction of conditions on raw inputs and on model or
entity-history meta columns), one action and frozen parameters. ``agent.error_slices``
estimates parameters on fit-only OOF rows; ``SliceCorrectionModel`` applies exactly
these functions to the wrapped estimator's predictions, so what the lab measured is
what the pipeline delivers. Rules see only inputs available at prediction time:
raw features, the base model's own prediction and strictly earlier target history
of the same entity. Truth never enters a rule.
"""
from copy import deepcopy
import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

PREDICTION = '__prediction'          # regression: base prediction (original units)
PREDICTED_CLASS = '__predicted_class'  # classification: base predicted label
CONFIDENCE = '__confidence'          # classification with probabilities: max class probability
HISTORY_PREFIX = '__history_'
HISTORY = ('__history_last', '__history_mean', '__history_count', '__history_gap', '__history_delta')
REGRESSION_ACTIONS = ('shift', 'shrink', 'blend_history')
LABEL_ACTIONS = ('remap',)
PROBABILITY_ACTIONS = ('recalibrate',)
NUMERIC_OPS = ('le', 'gt')
SET_OPS = ('in', 'not_in')
MAX_CORRECTIONS = 4
MAX_CONDITIONS = 4
MAX_SET_VALUES = 64
RATIO_BOUNDS = (0.2, 5.0)

CORRECTION_SCHEMA = {
    'type': 'array', 'maxItems': MAX_CORRECTIONS, 'items': {'type': 'object', 'properties': {
        'slice_id': {'type': 'string', 'minLength': 1, 'maxLength': 120},
        'conditions': {'type': 'array', 'minItems': 1, 'maxItems': MAX_CONDITIONS, 'items': {'type': 'object'}},
        'action': {'enum': [*REGRESSION_ACTIONS, *LABEL_ACTIONS, *PROBABILITY_ACTIONS]},
        'params': {'type': 'object'},
        'scale': {'enum': ['identity', 'log1p']},
        'entity': {'anyOf': [{'type': 'object'}, {'type': 'null'}]}},
        'required': ['slice_id', 'conditions', 'action', 'params', 'scale', 'entity'], 'additionalProperties': False}}


# --------------------------------------------------------------------------- rules

def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def condition_mask(frame, condition):
    column, op = condition['column'], condition['op']
    values = frame[column]
    if op in NUMERIC_OPS:
        number = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
        missing = number.isna().to_numpy()
        present = number.to_numpy(dtype=float)
        with np.errstate(invalid='ignore'):
            inside = present <= condition['value'] if op == 'le' else present > condition['value']
        return np.where(missing, bool(condition.get('missing', False)), inside)
    keys = values.astype('object').where(values.notna(), None).map(lambda v: None if v is None else str(v))
    members = set(condition['value'])
    inside = keys.map(lambda key: key in members).to_numpy(dtype=bool)
    return inside if op == 'in' else ~inside


def rule_mask(frame, conditions):
    mask = np.ones(len(frame), dtype=bool)
    for condition in conditions:
        mask &= condition_mask(frame, condition)
    return mask


def describe_condition(condition):
    column, op, value = condition['column'], condition['op'], condition['value']
    if op in NUMERIC_OPS:
        text = f'{column} {"<=" if op == "le" else ">"} {value:.6g}'
        return text+(' or missing' if condition.get('missing') else '')
    shown = ['missing' if v is None else str(v) for v in value[:6]]
    more = f' (+{len(value)-6})' if len(value) > 6 else ''
    return f'{column} {"in" if op == "in" else "not in"} {{{", ".join(shown)}{more}}}'


def describe(conditions):
    return ' AND '.join(describe_condition(c) for c in conditions)


def needs_history(corrections):
    for correction in corrections:
        if correction['action'] == 'blend_history':
            return True
        if any(c['column'].startswith(HISTORY_PREFIX) for c in correction['conditions']):
            return True
    return False


def meta_frame(base, *, regression, labels=None, probabilities=None, history=None):
    """Meta columns aligned with the predicted rows; ``base`` is the base model output."""
    meta = {}
    if regression:
        values = np.asarray(base, dtype=float)
        meta[PREDICTION] = values.reshape(len(values), -1)[:, 0]
    else:
        meta[PREDICTED_CLASS] = np.asarray(base).reshape(-1).astype(str)
        if probabilities is not None:
            meta[CONFIDENCE] = np.asarray(probabilities, dtype=float).max(axis=1)
    frame = pd.DataFrame(meta)
    if history is not None:
        for name in ('last', 'mean', 'count', 'gap'):
            frame[HISTORY_PREFIX+name] = np.asarray(history[HISTORY_PREFIX+name], dtype=float)
        if regression:
            frame['__history_delta'] = frame[PREDICTION]-frame['__history_last']
    return frame


def rule_frame(X, meta):
    """Raw inputs plus meta; meta names are reserved and never collide with inputs."""
    frame = X.reset_index(drop=True).copy()
    for column in meta:
        frame[column] = meta[column].to_numpy()
    return frame


# --------------------------------------------------------------------------- actions

def correction_scale(spec):
    """log1p only where the metric itself is defined on log1p of nonnegative values (RMSLE/MSLE)."""
    from agent.metrics import definition_of
    try:
        definition = definition_of(spec)
    except Exception:  # Unknown or classification metric: corrections act on original values.
        return 'identity'
    return 'log1p' if spec.get('task_kind') == 'regression' and definition.nonnegative_target else 'identity'


def forward(values, scale):
    values = np.asarray(values, dtype=float)
    return np.log1p(np.maximum(values, 0.0)) if scale == 'log1p' else values


def inverse(values, scale):
    values = np.asarray(values, dtype=float)
    return np.expm1(values) if scale == 'log1p' else values


def apply_regression(prediction, mask, action, params, scale, history_last=None):
    """Corrected copy of 1-D predictions; rows outside ``mask`` are unchanged."""
    result = np.asarray(prediction, dtype=float).copy()
    if not mask.any():
        return result
    current = forward(result[mask], scale)
    if action == 'shift':
        corrected = current+params['value']
        result[mask] = inverse(corrected, scale)
    elif action == 'shrink':
        w = params['weight']
        result[mask] = inverse((1-w)*current+w*params['center'], scale)
    elif action == 'blend_history':
        last = np.asarray(history_last, dtype=float)[mask]
        usable = np.isfinite(last)
        w = params['weight']
        blended = current.copy()
        blended[usable] = (1-w)*current[usable]+w*forward(last[usable], scale)
        result[mask] = inverse(blended, scale)
    else:
        raise ValueError('Unknown regression correction: '+str(action))
    return result


def apply_labels(predicted, mask, params):
    result = np.asarray(predicted).astype(object).copy()
    mapping = params['mapping']
    for index in np.flatnonzero(mask):
        result[index] = mapping.get(str(result[index]), result[index])
    return result.astype(str)


def apply_probabilities(probabilities, mask, params, labels):
    result = np.asarray(probabilities, dtype=float).copy()
    if not mask.any():
        return result
    ratios = np.asarray([params['ratios'].get(str(label), 1.0) for label in labels], dtype=float)
    scaled = result[mask]*ratios
    totals = scaled.sum(axis=1, keepdims=True)
    result[mask] = np.divide(scaled, totals, out=result[mask], where=totals > 0)
    return result


# --------------------------------------------------------------------------- validation

def validate_corrections(corrections, task_spec, columns=None):
    """Structural check shared by the controller and worker; returns a normalized copy."""
    if not isinstance(corrections, list) or not 1 <= len(corrections) <= MAX_CORRECTIONS:
        raise ValueError(f'slice_corrections must contain 1..{MAX_CORRECTIONS} corrections')
    regression = task_spec['task_kind'] == 'regression'
    kind = task_spec['metric']['prediction_kind']
    if regression and len(task_spec.get('target_columns') or [task_spec['target_column']]) != 1:
        raise ValueError('Slice corrections support one regression target')
    allowed_actions = REGRESSION_ACTIONS if regression else PROBABILITY_ACTIONS if kind == 'probabilities' else LABEL_ACTIONS
    meta = ({PREDICTION} if regression else {PREDICTED_CLASS} | ({CONFIDENCE} if kind == 'probabilities' else set()))
    labels = [str(label) for label in task_spec.get('allowed_labels') or []]
    expected_scale = correction_scale(task_spec)
    entity = None
    result = []
    for index, raw in enumerate(corrections):
        where = f'slice_corrections[{index}]'
        if not isinstance(raw, dict) or set(raw) != {'slice_id', 'conditions', 'action', 'params', 'scale', 'entity'}:
            raise ValueError(where+' needs exactly slice_id, conditions, action, params, scale, entity')
        correction = deepcopy(raw)
        if correction['action'] not in allowed_actions:
            raise ValueError(f'{where}.action must be one of {allowed_actions} for this task')
        if correction['scale'] != expected_scale:
            raise ValueError(f'{where}.scale must be {expected_scale} for this metric')
        uses_history = correction['action'] == 'blend_history' or any(
            isinstance(c, dict) and str(c.get('column', '')).startswith(HISTORY_PREFIX) for c in correction['conditions'])
        if uses_history:
            spec = correction['entity']
            if (not isinstance(spec, dict) or set(spec) != {'columns', 'period_column'} or not isinstance(spec['columns'], list)
                    or not 1 <= len(spec['columns']) <= 5 or not isinstance(spec['period_column'], str)):
                raise ValueError(where+'.entity must be {columns: [1..5 names], period_column} when history is used')
            if columns is not None and any(c not in columns for c in [*spec['columns'], spec['period_column']]):
                raise ValueError(where+'.entity names an absent column')
            if entity is not None and spec != entity:
                raise ValueError('All history corrections must share one entity definition')
            entity = spec
            meta_names = meta | set(HISTORY)
        else:
            if correction['entity'] is not None:
                raise ValueError(where+'.entity must be null when no history is used')
            meta_names = meta
        conditions = correction['conditions']
        if not isinstance(conditions, list) or not 1 <= len(conditions) <= MAX_CONDITIONS:
            raise ValueError(f'{where}.conditions must contain 1..{MAX_CONDITIONS} conditions')
        for position, condition in enumerate(conditions):
            here = f'{where}.conditions[{position}]'
            if not isinstance(condition, dict) or not {'column', 'op', 'value'} <= set(condition) or set(condition)-{'column', 'op', 'value', 'missing'}:
                raise ValueError(here+' needs column, op, value and optional missing')
            column, op, value = condition['column'], condition['op'], condition['value']
            if not isinstance(column, str) or (column.startswith('__') and column not in meta_names):
                raise ValueError(here+'.column is not an allowed meta column: '+str(column))
            if not column.startswith('__') and columns is not None and column not in columns:
                raise ValueError(here+'.column is absent: '+column)
            if op in NUMERIC_OPS:
                if not _finite_number(value) or not isinstance(condition.get('missing', False), bool):
                    raise ValueError(here+' numeric conditions need a finite value and boolean missing')
            elif op in SET_OPS:
                if ('missing' in condition or not isinstance(value, list) or not 1 <= len(value) <= MAX_SET_VALUES
                        or any(v is not None and not isinstance(v, str) for v in value)):
                    raise ValueError(here+' set conditions need 1..64 string values (null = missing)')
            else:
                raise ValueError(here+'.op must be le, gt, in or not_in')
        params = correction['params']
        action = correction['action']
        if action == 'shift':
            ok = set(params) == {'value'} and _finite_number(params['value'])
        elif action == 'shrink':
            ok = set(params) == {'weight', 'center'} and _finite_number(params['center']) and _finite_number(params['weight']) and 0 < params['weight'] <= 1
        elif action == 'blend_history':
            ok = set(params) == {'weight'} and _finite_number(params['weight']) and 0 < params['weight'] <= 1
        elif action == 'remap':
            mapping = params.get('mapping')
            ok = (set(params) == {'mapping'} and isinstance(mapping, dict) and mapping
                  and all(str(k) in labels and str(v) in labels and str(k) != str(v) for k, v in mapping.items()))
        else:
            ratios = params.get('ratios')
            ok = (set(params) == {'ratios'} and isinstance(ratios, dict) and ratios and all(
                str(k) in labels and _finite_number(v) and RATIO_BOUNDS[0] <= v <= RATIO_BOUNDS[1] for k, v in ratios.items()))
        if not ok:
            raise ValueError(f'{where}.params are invalid for {action}')
        result.append(correction)
    return result


# --------------------------------------------------------------------------- executable wrapper

class SliceCorrectionModel(BaseEstimator):
    """Wrap a fitted-in-place estimator with frozen slice corrections.

    History used by rules/actions comes from the rows this model is fitted on and
    only from strictly earlier periods of the same entity (EntityHistory).
    Each rule is evaluated on the base model's meta, never on corrected output.
    """
    def __init__(self, estimator=None, corrections=None, task_spec=None):
        self.estimator, self.corrections, self.task_spec = estimator, corrections, task_spec

    def fit(self, X, y):
        self.corrections_ = validate_corrections(self.corrections, self.task_spec, list(X.columns))
        self.estimator.fit(X, y)
        self.history_ = None
        if needs_history(self.corrections_):
            from agent.strong import EntityHistory
            entity = next(c['entity'] for c in self.corrections_ if c['entity'])
            self.history_ = EntityHistory(entity['columns'], entity['period_column'], prefix=HISTORY_PREFIX).fit(X, y)
        if self.task_spec['task_kind'] != 'regression':
            self.classes_ = np.asarray([str(c) for c in self.estimator.classes_])
        return self

    def _history(self, X):
        return self.history_.transform(X) if self.history_ is not None else None

    def _base_probabilities(self, X):
        raw = np.asarray(self.estimator.predict_proba(X), dtype=float)
        return raw

    def predict_proba(self, X):
        if self.task_spec['task_kind'] == 'regression':
            raise TypeError('Regression has no class probabilities')
        probabilities = self._base_probabilities(X)
        predicted = self.classes_[probabilities.argmax(axis=1)]
        frame = rule_frame(X, meta_frame(predicted, regression=False, probabilities=probabilities, history=self._history(X)))
        for correction in self.corrections_:
            if correction['action'] == 'recalibrate':
                probabilities = apply_probabilities(probabilities, rule_mask(frame, correction['conditions']),
                                                    correction['params'], list(self.classes_))
        return probabilities

    def predict(self, X):
        history = self._history(X)
        if self.task_spec['task_kind'] == 'regression':
            base = np.asarray(self.estimator.predict(X), dtype=float).reshape(-1)
            frame = rule_frame(X, meta_frame(base, regression=True, history=history))
            last = frame['__history_last'].to_numpy() if history is not None else None
            result = base
            for correction in self.corrections_:
                result = apply_regression(result, rule_mask(frame, correction['conditions']), correction['action'],
                                          correction['params'], correction['scale'], last)
            return result
        if any(c['action'] == 'recalibrate' for c in self.corrections_):
            return self.classes_[self.predict_proba(X).argmax(axis=1)]
        base = np.asarray(self.estimator.predict(X)).reshape(-1).astype(str)
        probabilities = self._base_probabilities(X) if callable(getattr(self.estimator, 'predict_proba', None)) else None
        frame = rule_frame(X, meta_frame(base, regression=False, probabilities=probabilities, history=history))
        result = base
        for correction in self.corrections_:
            result = apply_labels(result, rule_mask(frame, correction['conditions']), correction['params'])
        return result
