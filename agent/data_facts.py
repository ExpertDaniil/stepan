"""Deterministic train/test facts that decide the deployment scenario before any model.

Only feature values of the unlabelled test table are used, never test targets.
Nothing here knows dataset or column names: every rule is a measured property.
"""
import numpy as np
import pandas as pd

PERIOD_MAX_UNIQUE_FRACTION = 0.2   # Period-like ordinal columns repeat heavily.
PERIOD_MIN_TRAIN_VALUES = 3        # A split needs at least one earlier fit period.
TEST_AFTER_TRAIN_SHARE = 0.99      # Almost the whole test lies beyond the train range.


def _ordered(values):
    """Numeric values, or ISO-like datetimes as integers; None when not orderable."""
    if pd.api.types.is_bool_dtype(values):
        return None
    if pd.api.types.is_numeric_dtype(values):
        result = pd.to_numeric(values, errors='coerce')
        return result.astype(float)
    sample = values.dropna().astype(str).head(200)
    if sample.empty or not sample.str.match(r'^\d{4}-\d{2}-\d{2}').all():
        return None
    parsed = pd.to_datetime(values, errors='coerce', utc=True)
    if parsed.notna().mean() < 0.99:
        return None
    return pd.Series(parsed.astype('int64'), index=values.index).where(parsed.notna()).astype(float)


def future_period_column(train, test, excluded=()):
    """Return the period column the test follows in time, with evidence, or None.

    A candidate must be present in both tables, complete in train, repeat heavily
    (a period rather than a row counter), have at least three training periods and
    place nearly all test rows strictly after the latest training period.
    """
    excluded = set(excluded)
    best = None
    for column in train.columns:
        if column in excluded or column not in test.columns:
            continue
        fit, future = _ordered(train[column]), _ordered(test[column])
        if fit is None or future is None or fit.isna().any() or future.notna().mean() < 0.99:
            continue
        periods = fit.nunique()
        if periods < PERIOD_MIN_TRAIN_VALUES or periods > PERIOD_MAX_UNIQUE_FRACTION*len(fit):
            continue
        share = float((future.dropna() > fit.max()).mean())
        if share < TEST_AFTER_TRAIN_SHARE:
            continue
        evidence = {'column': column, 'train_periods': int(periods), 'test_after_train_share': round(share, 6),
                    'train_max': float(fit.max()), 'test_min': float(future.min())}
        if best is None or (share, -periods) > (best['test_after_train_share'], -best['train_periods']):
            best = evidence
    return best


def duplicate_profile(train, test, features):
    """How often fit rows repeat exactly, versus how often test rows repeat fit rows."""
    if not features:
        return {'train_duplicate_share': 0.0, 'test_seen_in_train_share': 0.0}
    key = lambda frame: pd.util.hash_pandas_object(frame[features].astype(str), index=False)
    fit, future = key(train), key(test)
    return {'train_duplicate_share': round(float(fit.duplicated(keep=False).mean()), 6),
            'test_seen_in_train_share': round(float(future.isin(set(fit)).mean()), 6)}


def deployment_facts(train, test, target_columns, excluded=()):
    """Facts consumed by validation choice and research context. Never raises."""
    facts = {'errors': []}
    skip = set(target_columns) | set(excluded)
    try:
        facts['future_period'] = future_period_column(train, test, skip)
    except Exception as error:  # A fact is optional evidence, never a run failure.
        facts['future_period'] = None
        facts['errors'].append('future_period: '+type(error).__name__+': '+str(error)[:200])
    try:
        features = [c for c in train.columns if c not in skip and c in test.columns]
        facts['duplicates'] = duplicate_profile(train, test, features)
    except Exception as error:
        facts['duplicates'] = None
        facts['errors'].append('duplicates: '+type(error).__name__+': '+str(error)[:200])
    return facts


def scenario_proposal(facts, seed):
    """A temporal split is chosen from data alone; other scenarios stay with the interpreter."""
    period = (facts or {}).get('future_period')
    if not period:
        return None
    return {'strategy': 'temporal_holdout', 'seed': seed,
            'parameters': {'valid_fraction': .2, 'column': period['column']},
            'reason': 'Data fact: %.1f%% of test rows lie after the latest of %d training periods of %s.'
                      % (100*period['test_after_train_share'], period['train_periods'], period['column'])}
