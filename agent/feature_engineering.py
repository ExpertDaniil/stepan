"""Owned, composable feature programs with inductive fit/transform semantics.

No labels, files, estimator search or mutable inference statistics are used here.
Arbitrary approved custom pipelines remain available alongside this small language.
"""
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from agent.feature_schema import GROUP_OPS, PROGRAM_VERSION, validate_program, operation_facts

def _keys(values):
    def key(value):
        if pd.isna(value):
            return 'M'
        if isinstance(value, (bool, np.bool_)):
            return 'B:'+str(bool(value))
        if isinstance(value, (int, float, np.integer, np.floating)):
            if not np.isfinite(value):
                return 'M'
            # CSV inference may yield int in one batch and float in another.
            return 'N:'+str(int(value)) if value == int(value) else 'N:'+repr(float(value))
        text = str(value)
        return 'S:'+str(len(text))+':'+text
    return values.map(key)


def _number(values):
    return pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)


class FeatureProgram(TransformerMixin, BaseEstimator):
    def __init__(self, program):
        self.program = program

    def fit(self, X, y=None):
        self.program_ = validate_program(self.program, X.columns)
        self.execution_spec_ = operation_facts(self.program_)
        self.input_columns_ = list(X.columns)
        self.statistics_ = {}
        self.report_ = []
        frame = X.copy()
        for step in self.program_:
            op, cols, name = step['op'], step['columns'], step['name']
            stats = {}
            if op == 'frequency':
                stats['counts'] = _keys(frame[cols[0]]).value_counts(normalize=True).to_dict()
            elif op in GROUP_OPS:
                group, values = _keys(frame[step['group_by']]), _number(frame[cols[0]])
                grouped = values.groupby(group, sort=False)
                if group.nunique() > 20000:
                    raise ValueError('Grouped feature exceeds 20000 fitted groups')
                if op == 'group_rank':
                    stats['sorted'] = {key: np.sort(part.dropna().to_numpy(dtype=float)) for key, part in grouped}
                else:
                    stats['values'] = (grouped.std() if op == 'group_std' else grouped.mean()).to_dict()
            self.statistics_[name] = stats
            frame[name] = self._apply(frame, step, stats)
            values = frame[name]
            self.report_.append({'name': name, 'op': op, 'inputs': cols,
                'program_id': self.execution_spec_['program_id'], 'verification': 'owned_operation_executed',
                'implementation': PROGRAM_VERSION, 'units': 'not_inferred',
                'group_by': step.get('group_by'), 'fit_rows': len(frame),
                'missing_fraction': float(values.isna().mean()), 'unique': int(values.nunique()),
                'statistics_scope': 'fit_rows_without_target'})
        return self

    def transform(self, X):
        if set(X.columns) != set(self.input_columns_):
            raise ValueError('Feature program input column set changed after fit')
        frame = X.loc[:, self.input_columns_].copy()
        for step in self.program_:
            frame[step['name']] = self._apply(frame, step, self.statistics_[step['name']])
        return frame

    @staticmethod
    def _apply(frame, step, stats):
        op, columns = step['op'], step['columns']
        raw = frame[columns[0]]
        if op == 'missing':
            return raw.isna().astype(float)
        if op == 'frequency':
            return _keys(raw).map(stats['counts']).fillna(0.0)
        if op == 'cross':
            result = _keys(raw)
            for column in columns[1:]:
                part = _keys(frame[column])
                result = result.str.len().astype(str)+':'+result+part.str.len().astype(str)+':'+part
            return result
        if op in {'month', 'dayofweek', 'hour'}:
            if pd.api.types.is_numeric_dtype(raw):
                raise ValueError('Numeric time needs explicit units in a custom pipeline; datetime operators expect strings/datetimes')
            dates = pd.to_datetime(raw, errors='coerce', utc=True)
            return getattr(dates.dt, op).astype(float)
        values = [_number(frame[c]) for c in columns]
        value = values[0]
        if op in GROUP_OPS:
            groups = _keys(frame[step['group_by']])
            if op == 'group_rank':
                result = pd.Series(np.nan, index=frame.index, dtype=float)
                for group, positions in groups.groupby(groups, sort=False).indices.items():
                    reference = stats['sorted'].get(group)
                    if reference is not None and len(reference):
                        sample = value.iloc[positions].to_numpy(dtype=float)
                        ranks = np.searchsorted(reference, sample, side='right')/len(reference)
                        ranks[~np.isfinite(sample)] = np.nan
                        result.iloc[positions] = ranks
                return result
            aggregate = groups.map(stats['values'])
            return value/aggregate.replace(0, np.nan) if op == 'relative_to_group' else aggregate
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            if op == 'log1p':
                result = np.log1p(value.where(value >= 0))
            elif op == 'signed_log1p':
                result = np.sign(value)*np.log1p(np.abs(value))
            elif op == 'square':
                result = value*value
            elif op == 'sum':
                result = pd.concat(values, axis=1).sum(axis=1, min_count=len(values))
            elif op == 'product':
                result = value.copy()
                for other in values[1:]:
                    result = result*other
            elif op == 'difference':
                result = value-values[1]
            elif op in {'cyclic_sin', 'cyclic_cos'}:
                phase = 2*np.pi*((value-step.get('origin', 0)) % step['period'])/step['period']
                result = np.sin(phase) if op == 'cyclic_sin' else np.cos(phase)
            else:
                result = value/values[1].replace(0, np.nan)
        return result.replace([np.inf, -np.inf], np.nan)


def make_feature_pipeline(task_spec, params):
    from agent.ml import build_pipeline
    from agent.predictions import targets
    options = deepcopy(params)
    base = options.pop('base_recipe_id')
    model = options.pop('model_parameters')
    program = validate_program(options.pop('feature_program'), excluded=[*targets(task_spec), *task_spec['excluded_features']])
    for key in ('_cpu_threads', '_seed'):
        if key in options:
            model[key] = options.pop(key)
    for key in list(options):
        if key.startswith('model__') and key[7:] and not key[7:].startswith('_'):
            model[key[7:]] = options.pop(key)
    if options or base.startswith(('python-', 'dummy-')):
        raise ValueError('Feature programs require one owned learned base recipe')
    return Pipeline([('features', FeatureProgram(program)), ('model', build_pipeline(base, task_spec, model))])
