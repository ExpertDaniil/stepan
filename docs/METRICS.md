# Metrics: registry, proxies, no constant fallbacks

`agent/metrics.py` is the only place that knows metric semantics: name, direction,
prediction kind, scorer, training proxy and output domain. Task resolution, the
Evaluator, the strong learner and recipe construction read it; nothing else
hardcodes metric names.

## Implemented

| Kind | metric_id |
|---|---|
| regression, values | `rmse`, `mse`, `mae`, `medae`, `r2`, `rmsle`, `msle`, `mape`, `smape` |
| classification, labels | `accuracy`, `balanced_accuracy`, `mcc`, `cohen_kappa`, `qwk`, `f1_macro`/`_weighted`/`_micro`, `precision_*`, `recall_*`; binary `f1`, `precision`, `recall` |
| classification, probabilities | `log_loss`; binary `roc_auc`, `pr_auc`; multiclass `roc_auc_ovr` |

Every scorer is checked against an independent sklearn/numpy reference
(`tests/test_metrics.py`). Names are normalized and matched against aliases
(`AUC`, `Matthews_CorrCoef`, `quadratic weighted kappa`, `R²`, ...).

## Resolution never aborts on a metric name

`metrics.resolve(draft_metric, task_kind, labels, truth)` returns a complete
`MetricSpec`. Whenever the result differs from the literal declaration,
`metric.declared = {metric_id, status, notes}` records why, and
`task.metric_resolved` is logged (WARNING for proxies):

- `alias` – another spelling of an implemented metric;
- `equivalent` – a monotone transform scored by its implemented twin (`gini` → `roc_auc`);
- `adapted` – binary name on a multiclass task (`roc_auc` → `roc_auc_ovr`, `f1` → `f1_macro`);
- `proxy` – no implementation; the same prediction kind is optimized with
  `rmse` / `f1_macro` / `log_loss`. Scores are then proxy scores and are never
  reported as the declared metric.

Registry direction and prediction kind are authoritative for implemented
metrics; a disagreeing draft is corrected and noted instead of failing.
Binary positive labels default to conventional names (`1`, `true`, `yes`,
`attack`, ...) and otherwise the minority class; `qwk` orders numeric class
names numerically. Log metrics with negative training targets fall back to `rmse`.

The interpreter prompt asks for the closest listed metric plus `declared` when
the task's metric is missing. If it still answers `unsupported` (or the provider
fails without a recoverable draft), `task_interpreter.supervised_reduction`
rebuilds a supervised draft from the files when exactly one train-only column is
requested by the sample submission. A constant emergency output scores zero, so
it remains only for tables without one supervised target.

## Learner proxies

| Metric family | Learners do |
|---|---|
| squared (`rmse`, `mse`, `r2`) | squared loss |
| absolute (`mae`, `medae`) | L1 objective for LightGBM/HGB/CatBoost/strong |
| log (`rmsle`, `msle`, `mape`, `smape`) | train on `log1p` targets (`ShiftedLogTargetModel`) unless a recipe sets `target_transform` |
| class-balanced labels (`f1_macro`, `balanced_accuracy`, `mcc`, kappas, binary F-family) | inverse-frequency class weights in the strong learner |
| decision labels (all label metrics except `accuracy`, `*_micro`) | strong learner tunes per-class offsets for the declared scorer on one inner half and keeps them only if the other half improves |

`apply_output_domain` clips regression outputs where a metric requires it
(RMSLE/MSLE: nonnegative) at prediction time, identically for validation and
submission, so the scored values are the delivered values.

An omitted output format follows the sample, not the metric: an AUC task whose
sample holds class names still submits labels; a single binary column of
probabilities is bound to the metric's positive label.
