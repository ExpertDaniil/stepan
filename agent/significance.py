"""Conditional paired bootstrap of protocol-defined groups, periods or rows.

This estimates sensitivity to the observed resampling units. It cannot measure
training-seed variance, unseen regimes, or correct adaptive multiple selection.
"""
import numpy as np

from agent import metrics as registry

RESAMPLES = 200
NONADDITIVE_RESAMPLES = 100
CONFIDENCE = 0.95            # Conditional bootstrap win fraction for support.
MIN_RELATIVE_GAIN = 0.001    # and at least 0.1% of the reference value


def row_losses(spec, truth, prediction):
    """Per-row losses whose weighted mean is a monotone transform of the metric, else None."""
    key = spec['metric']['metric_id']
    if spec['task_kind'] == 'regression':
        y = np.asarray(truth, dtype=float)
        p = np.asarray(prediction, dtype=float).reshape(y.shape)
        if y.ndim > 1:
            y, p = y.reshape(len(y), -1), p.reshape(len(p), -1)
        if key in {'rmse', 'mse', 'r2'}:
            loss = (y-p)**2
        elif key == 'mae':
            loss = np.abs(y-p)
        elif key in {'rmsle', 'msle'}:
            if (y < 0).any() or (p < 0).any():
                return None
            loss = (np.log1p(y)-np.log1p(p))**2
        elif key == 'mape':
            loss = np.abs(y-p)/np.maximum(np.abs(y), np.finfo(float).eps)
        elif key == 'smape':
            denominator = np.abs(y)+np.abs(p)
            loss = np.divide(2*np.abs(y-p), denominator, out=np.zeros_like(denominator), where=denominator > 0)
        else:
            return None
        return loss.mean(axis=1) if loss.ndim > 1 else loss
    labels = [str(label) for label in spec.get('allowed_labels') or []]
    y = np.asarray(truth).reshape(-1).astype(str)
    if spec['metric']['prediction_kind'] == 'probabilities':
        if key != 'log_loss':
            return None
        matrix = np.asarray(prediction, dtype=float)
        matrix = matrix/np.clip(matrix.sum(axis=1, keepdims=True), 1e-15, None)
        index = np.asarray([labels.index(v) for v in y])
        return -np.log(np.clip(matrix[np.arange(len(y)), index], 1e-15, 1))
    if key in {'accuracy', 'f1_micro', 'precision_micro', 'recall_micro'}:
        return (np.asarray(prediction).reshape(-1).astype(str) != y).astype(float)
    return None


def paired_bootstrap(spec, truth, reference, candidate, *, seed=0, resamples=None, clusters=None, unit='row'):
    """Paired resampling of whole declared units, conditional on the observed data.

    The win fraction is not a posterior probability of transfer and does not
    correct adaptive selection or training-seed variance. A single temporal
    period cannot estimate between-period uncertainty.
    """
    direction = 1.0 if spec['metric']['direction'] == 'maximize' else -1.0
    base = float(registry.score(spec, truth, reference))
    value = float(registry.score(spec, truth, candidate))
    gain = direction*(value-base)
    n = len(np.asarray(truth))
    if clusters is None:
        codes = np.arange(n)
    else:
        import pandas as pd
        if len(clusters) != n:
            raise ValueError('Bootstrap units must align with prediction rows')
        codes = pd.factorize(pd.Series(clusters).fillna('__missing_bootstrap_unit'), sort=False)[0]
    units = int(codes.max())+1 if n else 0
    relative = gain/max(abs(base), 1e-12)
    common = {'gain': gain, 'relative_gain': relative, 'reference_value': base, 'candidate_value': value,
              'resampling_unit': unit, 'independent_units': units,
              'interpretation': 'Conditional bootstrap win fraction; no correction for adaptive search or model seeds.'}
    if units < 2:
        return {**common, 'probability_better': None, 'bootstrap_win_fraction': None, 'interval': None,
                'significant': False, 'resamples': 0, 'method': 'unavailable',
                'unavailable': 'insufficient_independent_units'}
    rng = np.random.default_rng(seed)
    a_loss, b_loss = row_losses(spec, truth, reference), row_losses(spec, truth, candidate)
    if a_loss is not None and b_loss is not None:
        count = resamples or RESAMPLES
        # Resample units uniformly, retaining every row and its original weight
        # within a unit. Unequal unit sizes use the resampled row denominator.
        sizes = np.bincount(codes, minlength=units)
        sums_a = np.bincount(codes, weights=a_loss, minlength=units)
        sums_b = np.bincount(codes, weights=b_loss, minlength=units)
        # Keep memory proportional to a small batch of units, not all resamples.
        a_means, b_means = [], []
        for start in range(0, count, 16):
            weights = rng.multinomial(units, np.full(units, 1.0/units), size=min(16, count-start))
            denominator = weights @ sizes
            a_means.extend((weights @ sums_a)/denominator)
            b_means.extend((weights @ sums_b)/denominator)
        a_mean, b_mean = np.asarray(a_means), np.asarray(b_means)
        if spec['metric']['metric_id'] in {'rmse', 'rmsle'}:
            a_mean, b_mean = np.sqrt(a_mean), np.sqrt(b_mean)
        gains = a_mean-b_mean
        method = 'weighted_row_losses'
    else:
        count = resamples or NONADDITIVE_RESAMPLES
        gains = []
        truth_array, a, b = np.asarray(truth), np.asarray(reference), np.asarray(candidate)
        order = np.argsort(codes, kind='stable')
        members = np.split(order, np.cumsum(np.bincount(codes))[:-1]) if units != n else None
        for _ in range(count):
            drawn = rng.integers(0, units, units)
            rows = order[drawn] if members is None else np.concatenate([members[i] for i in drawn])
            try:
                value = direction*(float(registry.score(spec, truth_array[rows], b[rows]))
                                   - float(registry.score(spec, truth_array[rows], a[rows])))
                if np.isfinite(value):gains.append(value)
            except ValueError:
                continue
        gains = np.asarray(gains)
        method = 'recomputed_metric'
    if not len(gains):
        return {**common, 'probability_better': None, 'bootstrap_win_fraction': None, 'interval': None,
                'significant': False, 'method': method, 'resamples': 0, 'unavailable': 'undefined_resamples'}
    fraction = float(np.mean(gains > 0))
    return {**common, 'probability_better': fraction, 'bootstrap_win_fraction': fraction,
            'interval': [float(np.quantile(gains, .05)), float(np.quantile(gains, .95))],
            'significant': bool(gain > 0 and fraction >= CONFIDENCE and relative >= MIN_RELATIVE_GAIN),
            'interval_scale': ('per_row_loss' if method == 'weighted_row_losses' and spec['metric']['metric_id'] in
                               {'r2', 'accuracy', 'f1_micro', 'precision_micro', 'recall_micro'} else 'metric'),
            'method': method, 'resamples': int(len(gains)),
            'rule': f'bootstrap win fraction >= {CONFIDENCE} and relative gain >= {MIN_RELATIVE_GAIN}'}


def admits_update(verdict):
    """Publication/utility policy, distinct from a claim of statistical support.

    With one temporal block a material measured gain may advance the reference,
    explicitly unconfirmed. Otherwise retain the reference unless supported.
    """
    return bool(verdict.get('significant') or (
        verdict.get('unavailable') == 'insufficient_independent_units'
        and verdict.get('gain', 0) > 0 and verdict.get('relative_gain', 0) >= MIN_RELATIVE_GAIN))
