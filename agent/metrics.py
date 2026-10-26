"""Single metric registry: scoring, direction, prediction kind and training proxy.

Every consumer (task resolution, the independent Evaluator, recipes, decision
tuning) reads metric semantics from here; nothing else hardcodes metric names.

Scorers receive truth and predictions on the ORIGINAL target scale and never
repair predictions: the scored values are exactly the submitted values. Output
domains (e.g. RMSLE requires nonnegative predictions) are enforced once, where
predictions are produced, by ``apply_output_domain``.

Resolution never aborts on an unknown metric name. A known alias resolves
exactly; an unknown name becomes an explicit, logged proxy of the same
prediction kind, and ``metric['declared']`` records what the task asked for.
Scores reported under a proxy are the proxy's scores, never presented as the
declared metric.
"""
from dataclasses import dataclass, field
import math
import re

from agent.errors import AgentError

REGRESSION = frozenset({'regression'})
BINARY = frozenset({'binary_classification'})
CLASSIFICATION = frozenset({'binary_classification', 'multiclass_classification'})

# Training proxies. ``objective`` is what learners optimize when the declared
# metric is not directly trainable; ``decision`` says whether class decisions
# may be tuned on inner data against the declared scorer; ``balanced`` asks for
# inverse-frequency class weights because rare classes weigh as much as common.
SQUARED, ABSOLUTE, LOG_SQUARED = 'squared', 'absolute', 'log_squared'


@dataclass(frozen=True)
class MetricDef:
    metric_id: str
    direction: str
    prediction_kind: str
    task_kinds: frozenset
    description: str
    parameters: frozenset = frozenset()
    objective: str | None = None          # regression proxy objective
    decision: bool = False                # labels: tune class decisions on inner data
    balanced: bool = False                # classification: inverse-frequency weights
    nonnegative_target: bool = False      # metric undefined for negative truth
    output_floor: float | None = None     # predictions are clipped to this lower bound
    aliases: tuple = field(default=())


def _defs():
    values, labels, probabilities = 'values', 'labels', 'probabilities'
    items = [
        MetricDef('rmse', 'minimize', values, REGRESSION, 'Root mean squared error.', objective=SQUARED,
                  aliases=('root_mean_squared_error', 'rms_error')),
        MetricDef('mse', 'minimize', values, REGRESSION, 'Mean squared error.', objective=SQUARED,
                  aliases=('mean_squared_error', 'l2')),
        MetricDef('mae', 'minimize', values, REGRESSION, 'Mean absolute error.', objective=ABSOLUTE,
                  aliases=('mean_absolute_error', 'l1')),
        MetricDef('medae', 'minimize', values, REGRESSION, 'Median absolute error.', objective=ABSOLUTE,
                  aliases=('median_absolute_error', 'mdae')),
        MetricDef('r2', 'maximize', values, REGRESSION, 'Coefficient of determination (uniform target average).',
                  objective=SQUARED, aliases=('r_squared', 'r2_score', 'coefficient_of_determination', 'r^2')),
        MetricDef('rmsle', 'minimize', values, REGRESSION, 'Root mean squared logarithmic error, log1p scale.',
                  objective=LOG_SQUARED, nonnegative_target=True, output_floor=0.0,
                  aliases=('root_mean_squared_log_error', 'root_mean_squared_logarithmic_error', 'rmsle_score')),
        MetricDef('msle', 'minimize', values, REGRESSION, 'Mean squared logarithmic error, log1p scale.',
                  objective=LOG_SQUARED, nonnegative_target=True, output_floor=0.0,
                  aliases=('mean_squared_log_error', 'mean_squared_logarithmic_error')),
        MetricDef('mape', 'minimize', values, REGRESSION,
                  'Mean absolute percentage error as a fraction (sklearn definition, eps-guarded zero truth).',
                  objective=LOG_SQUARED, aliases=('mean_absolute_percentage_error',)),
        MetricDef('smape', 'minimize', values, REGRESSION,
                  'Symmetric MAPE as a fraction: mean(2|y-p|/(|y|+|p|)), 0/0 counted as 0.',
                  objective=LOG_SQUARED, aliases=('symmetric_mean_absolute_percentage_error', 'symmetric_mape')),
        MetricDef('accuracy', 'maximize', labels, CLASSIFICATION, 'Share of exactly correct labels.',
                  aliases=('accuracy_score', 'acc')),
        MetricDef('balanced_accuracy', 'maximize', labels, CLASSIFICATION, 'Mean per-class recall.',
                  decision=True, balanced=True, aliases=('balanced_accuracy_score', 'bacc')),
        MetricDef('mcc', 'maximize', labels, CLASSIFICATION, 'Matthews correlation coefficient (multiclass generalization).',
                  decision=True, balanced=True, aliases=('matthews_corrcoef', 'matthews_correlation', 'phi_coefficient')),
        MetricDef('cohen_kappa', 'maximize', labels, CLASSIFICATION, "Unweighted Cohen's kappa.",
                  decision=True, balanced=True, aliases=('kappa', 'cohens_kappa', 'cohen_kappa_score')),
        MetricDef('qwk', 'maximize', labels, CLASSIFICATION,
                  "Quadratic weighted Cohen's kappa over parameters.label_order (ordinal classes).",
                  parameters=frozenset({'label_order'}), decision=True, balanced=True,
                  aliases=('quadratic_weighted_kappa', 'quadratic_kappa', 'weighted_kappa', 'cohen_kappa_quadratic')),
        MetricDef('log_loss', 'minimize', probabilities, CLASSIFICATION, 'Multiclass cross-entropy over allowed labels.',
                  aliases=('logloss', 'cross_entropy', 'multi_logloss', 'binary_logloss', 'logarithmic_loss', 'neg_log_loss')),
        MetricDef('roc_auc', 'maximize', probabilities, BINARY, 'Binary ROC AUC of the positive-class probability.',
                  parameters=frozenset({'positive_label'}), aliases=('auc', 'roc_auc_score', 'auroc', 'area_under_roc_curve')),
        MetricDef('roc_auc_ovr', 'maximize', probabilities, CLASSIFICATION,
                  'Macro one-vs-rest ROC AUC over classes present with both outcomes.',
                  aliases=('multiclass_roc_auc', 'roc_auc_ovr_macro', 'macro_roc_auc', 'ovr_auc', 'multiclass_auc')),
        MetricDef('pr_auc', 'maximize', probabilities, BINARY, 'Average precision of the positive-class probability.',
                  parameters=frozenset({'positive_label'}),
                  aliases=('average_precision', 'average_precision_score', 'auprc', 'auc_pr', 'precision_recall_auc', 'ap')),
    ]
    # F-family and precision/recall: binary (explicit positive) and averaged forms.
    names = {'f1': ('f1_score', 'f_score', 'f_measure'), 'precision': ('precision_score', 'ppv'),
             'recall': ('recall_score', 'sensitivity', 'tpr')}
    for base, extra in names.items():
        items.append(MetricDef(base, 'maximize', labels, BINARY, f'Binary {base} of parameters.positive_label.',
                               parameters=frozenset({'positive_label'}), decision=True, balanced=True,
                               aliases=extra+(base+'_binary', 'binary_'+base)))
        for average in ('macro', 'weighted', 'micro'):
            items.append(MetricDef(f'{base}_{average}', 'maximize', labels, CLASSIFICATION,
                                   f'{average.capitalize()}-averaged {base} over allowed labels.',
                                   decision=average != 'micro',
                                   # Weighted F1 kept its historical class weighting in the strong learner.
                                   balanced=average == 'macro' or (base, average) == ('f1', 'weighted'),
                                   aliases=(f'{average}_{base}', f'{base}_score_{average}')))
    return {item.metric_id: item for item in items}


METRICS = _defs()
# Monotone equivalents score exactly like their target metric for selection.
EQUIVALENT = {'gini': ('roc_auc', 'Gini = 2*AUC-1 is a monotone transform of ROC AUC; AUC is reported.'),
              'normalized_gini': ('roc_auc', 'Normalized Gini = 2*AUC-1 is a monotone transform of ROC AUC; AUC is reported.'),
              'neg_mean_squared_error': ('mse', 'Negated MSE ranks candidates exactly like MSE.'),
              'neg_root_mean_squared_error': ('rmse', 'Negated RMSE ranks candidates exactly like RMSE.'),
              'neg_mean_absolute_error': ('mae', 'Negated MAE ranks candidates exactly like MAE.'),
              'error_rate': ('accuracy', 'Error rate = 1-accuracy; accuracy is reported.'),
              'zero_one_loss': ('accuracy', 'Zero-one loss = 1-accuracy; accuracy is reported.')}
PROXY = {'values': 'rmse', 'labels': 'f1_macro', 'probabilities': 'log_loss'}


def normalize_name(name):
    text = str(name or '').strip().lower().replace('^', '').replace('²', '2')
    text = re.sub(r'[^a-z0-9]+', '_', text).strip('_')
    return text


def _alias_table():
    table = {}
    for metric in METRICS.values():
        for name in (metric.metric_id, *metric.aliases):
            key = normalize_name(name)
            if key in table and table[key] != metric.metric_id:
                raise RuntimeError('Ambiguous metric alias: '+key)
            table[key] = metric.metric_id
    return table


ALIASES = _alias_table()


def get(metric_id):
    try:
        return METRICS[metric_id]
    except KeyError:
        raise AgentError('UNSUPPORTED_METRIC', str(metric_id)) from None


def supported_summary():
    """Compact catalogue for prompts and diagnostics."""
    return [{'metric_id': m.metric_id, 'direction': m.direction, 'prediction_kind': m.prediction_kind,
             'task_kinds': sorted(m.task_kinds), 'parameters': sorted(m.parameters)} for m in METRICS.values()]


# --------------------------------------------------------------------------- resolution

def _numeric_order(labels):
    try:
        values = [float(label) for label in labels]
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(v) for v in values):
        return None
    return [label for _, label in sorted(zip(values, labels))]


def _lookup(name, notes):
    """(metric_id, status) for an implemented name, alias or monotone equivalent; (None, 'proxy') otherwise."""
    key = normalize_name(name)
    if key in ALIASES:
        resolved = ALIASES[key]
        return resolved, 'exact' if resolved == name else 'alias'
    if key in EQUIVALENT:
        resolved, note = EQUIVALENT[key]
        notes.append(note)
        return resolved, 'equivalent'
    return None, 'proxy'


POSITIVE_WORDS = ('1', 'true', 'yes', 'y', 'positive', 'pos', 'default', 'fraud', 'churn', 'attack', 'malicious', 'spam')


def default_positive_label(labels, truth=None):
    """Deterministic positive class for a binary metric: conventional names, then the minority."""
    labels = [str(label) for label in labels]
    if len(labels) != 2:
        raise AgentError('TASK_SPEC_CONFLICT', 'A positive label needs exactly two classes')
    lowered = {label.strip().lower(): label for label in labels}
    for word in POSITIVE_WORDS:
        if word in lowered and len(lowered) == 2:
            other = next(key for key in lowered if key != word)
            if other not in POSITIVE_WORDS:
                return lowered[word], f'conventional positive name {lowered[word]!r}'
    if truth is not None:
        import pandas as pd
        counts = pd.Series(truth).astype(str).value_counts()
        counts = counts.reindex(labels, fill_value=0)
        if counts.iloc[0] != counts.iloc[1]:
            label = str(counts.idxmin())
            return label, f'minority class {label!r} (no conventional positive name)'
    return sorted(labels)[-1], f'last sorted label {sorted(labels)[-1]!r} (balanced classes, no conventional name)'


def resolve(draft_metric, task_kind, labels, truth=None):
    """Return a complete, validated MetricSpec; never raises for an unknown name.

    Exact names and aliases resolve exactly. Task-kind adaptations (binary/multiclass
    forms of the same quantity) and proxies are explicit in ``declared``.
    Raises only for internally inconsistent inputs (bad task kind, invalid labels).
    """
    if task_kind not in REGRESSION | CLASSIFICATION:
        raise AgentError('UNSUPPORTED_TASK_KIND', str(task_kind))
    metric = dict(draft_metric or {})
    supplied = dict(metric.get('parameters') or {})
    declared = dict(metric.get('declared') or {})
    notes = [str(note) for note in declared.get('notes') or [] if str(note).strip()]
    chosen = str(metric.get('metric_id') or '')
    # The task's own metric name wins over an interpreter's proxy whenever we
    # implement it; otherwise the interpreter's choice is kept as a declared proxy.
    original = str(declared.get('metric_id') or '') or chosen
    requested = original
    resolved, status = _lookup(original, notes)
    if resolved is None and chosen and chosen != original:
        resolved, _ = _lookup(chosen, [])
        if resolved is not None:
            status = 'proxy'
            notes.append(f'No implementation of {original!r}; the interpreter chose the proxy {resolved}. Scores are proxy scores.')
    multiclass = task_kind == 'multiclass_classification'
    if resolved is not None and task_kind not in get(resolved).task_kinds:
        # Same quantity, other class cardinality: adapt explicitly instead of failing.
        adapted = {('roc_auc', True): 'roc_auc_ovr', ('f1', True): 'f1_macro', ('precision', True): 'precision_macro',
                   ('recall', True): 'recall_macro', ('pr_auc', True): None}.get((resolved, multiclass))
        if adapted:
            notes.append(f'{resolved} is binary-only; the multiclass task uses {adapted}.')
            resolved, status = adapted, 'adapted'
        else:
            notes.append(f'{resolved} does not apply to {task_kind}.')
            resolved = None
    if resolved is None:
        kind = metric.get('prediction_kind')
        if task_kind == 'regression':
            kind = 'values'
        elif kind not in {'labels', 'probabilities'}:
            kind = 'labels'
        resolved, status = PROXY[kind], 'proxy'
        notes.append(f'No implementation of {requested or "an unnamed metric"!r}; optimizing and reporting '
                     f'the proxy {resolved} with the same prediction kind. Scores are proxy scores.')
    definition = get(resolved)
    if task_kind == 'regression' and definition.nonnegative_target and truth is not None:
        import numpy as np
        values = np.asarray(truth, dtype=float)
        if (values < 0).any():
            notes.append(f'{resolved} is undefined for negative targets present in training; using rmse.')
            definition, status = get('rmse'), 'proxy'
    parameters = {}
    for name in definition.parameters:
        if name == 'positive_label':
            positive = supplied.get('positive_label')
            if positive is None or str(positive) not in labels:
                positive, why = default_positive_label(labels, truth)
                notes.append('positive_label defaulted to '+why+'.')
            parameters['positive_label'] = str(positive)
        elif name == 'label_order':
            order = supplied.get('label_order')
            if not order or sorted(map(str, order)) != sorted(labels) or len(set(map(str, order))) != len(labels):
                order = _numeric_order(labels)
                if order is None:
                    order = sorted(labels)
                    notes.append('label_order defaulted to lexicographic order; the task gave no numeric ordering.')
                else:
                    notes.append('label_order defaulted to numeric order of the class names.')
            parameters['label_order'] = [str(label) for label in order]
    if definition.metric_id.partition('_')[0] in {'f1', 'precision', 'recall'}:
        # Explicit vocabulary: classes absent from a validation subset still count.
        parameters['labels'] = list(labels)
        parameters['zero_division'] = 0
    ignored = sorted(set(supplied)-set(parameters)-{'labels', 'zero_division'})
    if ignored:
        notes.append('Ignored metric parameters without an implementation: '+', '.join(ignored)+'.')
    result = {'metric_id': definition.metric_id, 'direction': definition.direction,
              'prediction_kind': definition.prediction_kind, 'parameters': parameters}
    if status != 'proxy':
        # A proxy legitimately differs in direction/kind; for implemented metrics the
        # registry is authoritative and a disagreeing draft is corrected, not fatal.
        for field_name in ('direction', 'prediction_kind'):
            given = metric.get(field_name)
            if given not in (None, getattr(definition, field_name)):
                notes.append(f'Declared {field_name} {given!r} replaced by the registry value {getattr(definition, field_name)!r}.')
    if status != 'exact' or notes:
        result['declared'] = {'metric_id': requested or None, 'status': status, 'notes': notes}
    return result


# --------------------------------------------------------------------------- scoring

def _labels(spec):
    return [str(label) for label in spec.get('allowed_labels') or []]


def _regression(metric_id, y, p):
    import numpy as np
    from sklearn import metrics as m
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if p.shape != y.shape:
        p = p.reshape(y.shape)
    if metric_id == 'rmse':
        return math.sqrt(m.mean_squared_error(y, p))
    if metric_id == 'mse':
        return float(m.mean_squared_error(y, p))
    if metric_id == 'mae':
        return float(m.mean_absolute_error(y, p))
    if metric_id == 'medae':
        return float(m.median_absolute_error(y, p))
    if metric_id == 'r2':
        return float(m.r2_score(y, p))
    if metric_id in {'rmsle', 'msle'}:
        if (y < 0).any() or (p < 0).any():
            raise ValueError(f'{metric_id} requires nonnegative truth and predictions')
        value = float(m.mean_squared_log_error(y, p))
        return math.sqrt(value) if metric_id == 'rmsle' else value
    if metric_id == 'mape':
        return float(m.mean_absolute_percentage_error(y, p))
    if metric_id == 'smape':
        denominator = np.abs(y)+np.abs(p)
        ratio = np.divide(2*np.abs(y-p), denominator, out=np.zeros_like(denominator), where=denominator > 0)
        return float(np.mean(ratio))
    raise AgentError('UNSUPPORTED_METRIC', metric_id)


def label_score(metric, labels, y, p):
    """Declared label metric for 1-D string arrays; also used by decision tuning."""
    import numpy as np
    from sklearn import metrics as m
    key = metric['metric_id']
    parameters = metric.get('parameters') or {}
    y, p = np.asarray(y).astype(str).reshape(-1), np.asarray(p).astype(str).reshape(-1)
    if key == 'accuracy':
        return float(m.accuracy_score(y, p))
    if key == 'balanced_accuracy':
        # sklearn warns and ignores predicted-only classes; recall per present class is the definition.
        present = [label for label in labels if (y == label).any()]
        if not present:
            raise ValueError('balanced_accuracy needs at least one present class')
        return float(np.mean([np.mean(p[y == label] == label) for label in present]))
    if key == 'mcc':
        return float(m.matthews_corrcoef(y, p))
    if key == 'cohen_kappa':
        return float(m.cohen_kappa_score(y, p, labels=labels))
    if key == 'qwk':
        return float(m.cohen_kappa_score(y, p, labels=list(parameters['label_order']), weights='quadratic'))
    base, _, average = key.partition('_')
    function = {'f1': m.f1_score, 'precision': m.precision_score, 'recall': m.recall_score}.get(base)
    if function is None:
        raise AgentError('UNSUPPORTED_METRIC', key)
    if not average:
        return float(function(y, p, pos_label=parameters['positive_label'], average='binary', labels=labels, zero_division=0))
    return float(function(y, p, average=average, labels=labels, zero_division=0))


def _probabilities(metric, labels, y, prediction):
    import numpy as np
    from sklearn import metrics as m
    key = metric['metric_id']
    y = np.asarray(y).astype(str).reshape(-1)
    prediction = np.asarray(prediction, dtype=float)
    if key == 'log_loss':
        return float(m.log_loss(y, prediction, labels=labels))
    if key in {'roc_auc', 'pr_auc'}:
        positive = metric['parameters']['positive_label']
        score = prediction[:, labels.index(positive)]
        truth = y == positive
        return float(m.roc_auc_score(truth, score) if key == 'roc_auc' else m.average_precision_score(truth, score))
    if key == 'roc_auc_ovr':
        values = [m.roc_auc_score(y == label, prediction[:, index]) for index, label in enumerate(labels)
                  if 0 < (y == label).sum() < len(y)]
        if not values:
            raise ValueError('roc_auc_ovr needs a class with both outcomes present')
        return float(np.mean(values))
    raise AgentError('UNSUPPORTED_METRIC', key)


def score(spec, truth, prediction):
    """Declared (or explicitly proxied) metric of a spec on original-scale predictions."""
    import numpy as np
    metric = spec['metric']
    definition = get(metric['metric_id'])
    if spec['task_kind'] == 'regression':
        return _regression(definition.metric_id, truth, prediction)
    labels = _labels(spec)
    if definition.prediction_kind == 'probabilities':
        return _probabilities(metric, labels, truth, prediction)
    return label_score(metric, labels, truth, np.asarray(prediction).reshape(-1))


def better(direction, candidate, reference):
    return candidate < reference if direction == 'minimize' else candidate > reference


# --------------------------------------------------------------------------- learner proxies

def definition_of(spec):
    return get(spec['metric']['metric_id'])


def regression_objective(spec):
    """squared / absolute / log_squared for learners; squared for unknown or classification."""
    try:
        definition = definition_of(spec)
    except (AgentError, KeyError, TypeError):
        return SQUARED
    return definition.objective or SQUARED


def default_target_transform(spec):
    """Train-only target transform implied by the metric when a recipe does not set one."""
    if spec.get('task_kind') != 'regression':
        return 'identity'
    return 'shifted_log1p' if regression_objective(spec) == LOG_SQUARED else 'identity'


def class_balanced(spec):
    try:
        return definition_of(spec).balanced
    except (AgentError, KeyError, TypeError):
        return False


def tunes_decisions(spec):
    try:
        definition = definition_of(spec)
    except (AgentError, KeyError, TypeError):
        return False
    return definition.prediction_kind == 'labels' and definition.decision


def apply_output_domain(spec, values):
    """Clip numeric predictions into the metric's domain; identity for other kinds."""
    import numpy as np
    if spec.get('task_kind') != 'regression':
        return values
    try:
        floor = definition_of(spec).output_floor
    except (AgentError, KeyError, TypeError):
        return values
    if floor is None:
        return values
    return np.maximum(np.asarray(values, dtype=float), floor)
