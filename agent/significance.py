"""Paired bootstrap over evaluation rows: is a candidate better than a reference beyond row noise?

Two fixed prediction vectors are compared on the same rows. Resampling rows
captures how much of an observed difference could come from which rows happened
to be in the selection set; consistent per-row gains are detected even when small,
parameter jitter that helps some rows and hurts others is not. Metrics that are
means of a per-row loss are resampled with weight vectors (vectorized); other
metrics are recomputed per resample through the registry.
"""
import numpy as np

from agent import metrics as registry

RESAMPLES = 200
NONADDITIVE_RESAMPLES = 100
CONFIDENCE = 0.95            # P(candidate better) needed to call an improvement real
MIN_RELATIVE_GAIN = 0.001    # and at least 0.1% of the reference value


def _row_losses(spec, truth, prediction):
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


def paired_bootstrap(spec, truth, reference, candidate, *, seed=0, resamples=None):
    """Bootstrap the declared metric's difference ``candidate`` vs ``reference`` on shared rows.

    Returns gain (positive = candidate better under the metric direction) on all
    rows, its 5%/95% bootstrap interval, P(gain > 0) and the decision ``significant``.
    """
    direction = 1.0 if spec['metric']['direction'] == 'maximize' else -1.0
    base = float(registry.score(spec, truth, reference))
    value = float(registry.score(spec, truth, candidate))
    gain = direction*(value-base)
    n = len(np.asarray(truth))
    rng = np.random.default_rng(seed)
    a_loss, b_loss = _row_losses(spec, truth, reference), _row_losses(spec, truth, candidate)
    if a_loss is not None and b_loss is not None:
        count = resamples or RESAMPLES
        weights = rng.multinomial(n, np.full(n, 1.0/n), size=count).astype(float)
        a_mean, b_mean = weights @ a_loss/n, weights @ b_loss/n
        if spec['metric']['metric_id'] in {'rmse', 'rmsle'}:
            a_mean, b_mean = np.sqrt(a_mean), np.sqrt(b_mean)
        # Every additive loss is minimized; r2 and accuracy rank exactly like their losses.
        gains = a_mean-b_mean
        method = 'weighted_row_losses'
    else:
        count = resamples or NONADDITIVE_RESAMPLES
        gains = np.empty(count)
        truth_array, a, b = np.asarray(truth), np.asarray(reference), np.asarray(candidate)
        valid = 0
        for index in range(count):
            rows = rng.integers(0, n, n)
            try:
                gains[valid] = direction*(float(registry.score(spec, truth_array[rows], b[rows]))
                                          - float(registry.score(spec, truth_array[rows], a[rows])))
                valid += 1
            except ValueError:
                continue  # A resample can lack a class an undefined metric needs.
        gains = gains[:valid]
        method = 'recomputed_metric'
    if not len(gains):
        return {'gain': gain, 'probability_better': None, 'interval': None, 'significant': False,
                'method': method, 'resamples': 0, 'reference_value': base, 'candidate_value': value}
    probability = float(np.mean(gains > 0))
    relative = gain/max(abs(base), 1e-12)
    return {'gain': gain, 'relative_gain': relative, 'probability_better': probability,
            'interval': [float(np.quantile(gains, .05)), float(np.quantile(gains, .95))],
            'significant': bool(gain > 0 and probability >= CONFIDENCE and relative >= MIN_RELATIVE_GAIN),
            'interval_scale': ('metric' if method == 'recomputed_metric' or spec['metric']['metric_id'] not in {'r2', 'accuracy'}
                               else 'per_row_loss'),
            'method': method, 'resamples': int(len(gains)), 'reference_value': base, 'candidate_value': value,
            'rule': f'P(better) >= {CONFIDENCE} and relative gain >= {MIN_RELATIVE_GAIN}'}

