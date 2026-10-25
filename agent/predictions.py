"""Explicit prediction contracts shared by execution and independent validation.

Supported adapters: supervised labels, class probabilities, numeric single/multiple
targets. Ranking and unsupervised protocols are not inferred from table shapes.
"""
from agent.errors import AgentError


def targets(spec):
    return list(spec.get('target_columns') or [spec['target_column']])


def read_training(path, spec):
    from tools.data import read_table
    dtype = {spec['target_column']: str} if spec['task_kind'] != 'regression' else None
    return read_table(path, dtype=dtype)


def prediction_kind(spec, phase='train'):
    if phase != 'train' and spec['output'].get('prediction_kind'):
        return spec['output']['prediction_kind']
    return spec['metric']['prediction_kind']


def validation_columns(spec):
    if spec['metric']['prediction_kind'] == 'probabilities':
        return ['probability_'+str(i) for i in range(len(spec['allowed_labels']))]
    count = len(targets(spec))
    return ['prediction'] if count == 1 else ['prediction_'+str(i) for i in range(count)]


def checked_matrix(values, rows, columns, kind, *, labels=(), full_probabilities=True):
    import numpy as np
    array = np.asarray(values)
    if array.ndim == 1 and columns == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape != (rows, columns):
        raise AgentError('INVALID_PREDICTIONS', 'Prediction dimensions do not match the declared row/output contract',
                         details={'actual_shape': list(array.shape), 'expected_shape': [rows, columns]})
    if kind == 'labels':
        if any(v is None for v in array.flat):
            raise AgentError('INVALID_PREDICTIONS', 'Missing class label')
        array = array.astype(str)
        if not set(array.flat) <= set(labels):
            raise AgentError('INVALID_PREDICTIONS', 'Unknown predicted class label')
        return array
    try:
        array = array.astype(float)
    except (ValueError, TypeError) as error:
        raise AgentError('INVALID_PREDICTIONS', 'Expected numeric predictions') from error
    if not np.isfinite(array).all():
        raise AgentError('INVALID_PREDICTIONS', 'Non-finite predictions')
    if kind == 'probabilities':
        if (array < 0).any() or (array > 1).any():
            raise AgentError('INVALID_PREDICTIONS', 'Probabilities must lie between zero and one')
        if full_probabilities and not np.allclose(array.sum(axis=1), 1, atol=1e-6, rtol=0):
            raise AgentError('INVALID_PREDICTIONS', 'Class probabilities must sum to one')
    return array


def predict(model, frame, spec, phase):
    import numpy as np
    kind = prediction_kind(spec, phase)
    if kind != 'probabilities':
        return checked_matrix(model.predict(frame), len(frame), len(targets(spec)), kind,
                              labels=spec['allowed_labels'])
    if not callable(getattr(model, 'predict_proba', None)):
        raise AgentError('UNSUPPORTED_PREDICTION_INTERFACE', 'This task requires predict_proba')
    classes = getattr(model, 'classes_', None)
    if classes is None:
        raise AgentError('PROBABILITY_CLASS_ORDER_REQUIRED', 'predict_proba must expose fitted classes_')
    classes = [str(c) for c in classes]
    labels = spec['allowed_labels']
    if len(set(classes)) != len(classes) or not set(classes) <= set(labels):
        raise AgentError('INVALID_PREDICTIONS', 'Fitted class order is ambiguous or contains unknown classes')
    raw = checked_matrix(model.predict_proba(frame), len(frame), len(classes), 'probabilities')
    aligned = np.zeros((len(frame), len(labels)), dtype=float)
    for index, label in enumerate(classes):
        aligned[:, labels.index(label)] = raw[:, index]
    if phase != 'train':
        output = spec['output']
        order = output.get('class_labels') or [output['positive_label']]
        aligned = aligned[:, [labels.index(label) for label in order]]
    return aligned
