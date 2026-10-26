# Error slices: where loss concentrates, what fixes it, what Qwen does about it

## Why the old loop gave Qwen nothing to work with

`error_profile` cut the fit-only OOF errors by one column at a time (missing /
quartile / category value) and ranked groups by total error mass. The biggest
groups always won: "State_6: 74% of SSE" with 84% of the rows, "energy_star
missing: 77% of SSE" with 77% of the rows. Neither is a hard region. The
researcher chased them with target encodings a GBDT already learns, the only
executable lever was a whole-model refit, and a +-0.7 RMSE seed variance hid
any slice-sized effect. In the reviewed run all post-23-minute "gains" were the
same four missing flags with jittered parameters.

## What replaces it

`agent/error_slices.py`, run inside `FitDiagnostics.complete` on the leader's
fit-only OOF predictions (no selection/confirmation label is read):

1. **Loss per row** in the declared metric's own terms (`agent.metrics`):
   squared error (log1p scale for RMSLE-type metrics), absolute error for
   MAE-type, class-weighted misses for balanced label metrics, log loss for
   probabilities.
2. **Mining.** A depth-3 regression tree over raw inputs *and meta*:
   `__prediction` / `__predicted_class` / `__confidence` (the base model's own
   output) and, under a temporal protocol, `__history_last/mean/count/gap/delta`
   (strictly earlier target history of the entity that `agent.strong` discovers
   on fit rows). Every node becomes a candidate rule; candidates are ranked by
   **excess** = loss share - row share (lift >= 1.2, >= 2 pp excess), nested or
   overlapping candidates are dropped. A structural slice "entity has history"
   is also tested, because a history blend can help everywhere with zero excess.
   Slice membership is always recomputed from the translated rule exactly as a
   deployed pipeline evaluates it, never from the tree's own routing.
3. **Fixability lab**, cross-fitted over the OOF folds (parameters estimated on
   the other folds, applied to the held-out fold, scored with the declared
   metric on all OOF rows):
   - regression: `shift` (mean/median residual), `shrink` toward the slice mean,
     `blend_history` with the entity's last value;
   - labels: `remap` predicted class a -> b where it raises the declared metric;
   - probabilities: `recalibrate` class ratios.
   A remedy is **robust** when the overall OOF metric improves by >= 0.1% and
   at least 2/3 of held-out folds improve. Final parameters are re-estimated on
   all OOF slice rows and frozen.
4. **Explainability**: a small cross-fitted LightGBM predicts the slice residual
   (regression, OOF R^2) or the error indicator (classification, OOF AUC), with
   top features by gain.
5. **Verdict**: `fixable` (robust remedy) > `explainable` (R^2 >= 0.05 / AUC >=
   0.65) > `noise`.

## What Qwen sees and does

`error_research` turns slices into problems (priority: verdict, then remedy
gain, then excess) and gives each researcher one card: rule, row/loss share,
lift, fold consistency, bias or confusions, verdict, up to three remedies with
`remedy_ref` (`<diagnostic artifact>#<slice>/<action>`), top explanatory
features and a few worst examples. The question says what to do:

- fixable -> `{"action":"features","slice_remedies":["<remedy_ref>"]}`. The
  controller binds the remedy to the exact version it was measured on (recipe,
  parameters, feature program and earlier corrections come from the immutable
  report), appends any proposed `feature_program`, and trains
  `SliceCorrectionModel` around it. A frozen correction on a different base
  would be meaningless (observed: shift calibrated on one linear fit worsened
  another from 1.19 to 4.11).
- explainable -> design features from the named inputs.
- noise -> spend research elsewhere.

`compact_report` shows slice cards instead of single-column groups; the
selection-level single-column groups are no longer copied into graph context.

## Execution

`agent/slice_actions.py` holds the rule language, the actions and
`SliceCorrectionModel`; the lab and the worker call the same functions, so
measured equals delivered. Rules use only prediction-time inputs. History used
by a deployed rule/action comes from the rows the model is fitted on, strictly
earlier periods only. `validate_corrections` checks action/metric/scale
compatibility, column existence, meta names and parameter domains in both the
controller and the worker.

## Measurement and journal

Every research candidate is still scored by the Evaluator on selection rows and
paired with its research base inside/outside the slice (`rule_membership`
recomputes the rule on selection rows using the base's own predictions and
fit-partition history). Closed confirmation still decides the finalists.

`research.journal` events give one readable line per mined slice and per
evaluated research candidate (background Optuna trials are not journalled):

```
S1 [x_numeric > 2.88789 or missing]: 6.9% rows -> 83.7% loss (x12.0, folds 3/3); shift OOF rmse 1.94937->1.83055 (2/3 folds) => fixable
researcher-1: remedies art-4b2a...#S1/shift:shift on S1 [...]: slice rmse 5.2802->3.73695, overall 1.18628->0.944384 => overall_improved
```

## Limits

The lab is evidence on fit-only OOF of one leader; it never replaces the
Evaluator or the closed confirmation (on a tiny synthetic task the confirmation
rows rejected a selection-improving remedy, as designed). Thresholds (excess,
lift, robustness, explainability) are deliberately conservative defaults, not
tuned on the public tasks. Live Qwen behaviour on the cards has not been run in
this environment (no provider credentials).
