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

`agent/error_slices.py` runs inside the leased OOF worker on fit-only predictions
(no selection/confirmation label is read). `FitDiagnostics.complete` only verifies
and publishes its artifact. The worker's CPU/RSS/timeout and finalization cancellation
cover both OOF fits and the residual-model lab:

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
3. **Holdout discipline.** The last inner OOF fold (the latest period under a
   temporal protocol) is held out. Slices are mined and remedy parameters are
   fitted on the other (discovery) folds only; the holdout fold, which played
   no part in choosing the rule or its parameters, decides robustness. Frozen
   parameters are the discovery-fitted ones that were checked.
   Every discovery base fit also excludes this last fold, including under random,
   stratified and group protocols. History key discovery and its target table use
   only the union of those fit rows. Existing OOF fit count is preserved; random
   discovery models consequently train on fewer rows. If too few folds remain,
   the diagnostic reports unavailable rather than manufacturing independence.
4. **Fixability lab**:
   - regression: `shift`, `shrink` toward the slice mean, `blend_history` with the
     entity's last value, all chosen by the declared metric's own row loss
     (MAPE uses |y-p|/|y|; log1p scale only for RMSLE/MSLE);
   - labels: `remap` predicted class a -> b where it raises the declared metric;
   - probabilities: `recalibrate` class ratios.
   A remedy is **robust** when the holdout fold improves by >= 0.1% of its
   metric with >= 20 slice rows there.
5. **Explainability**: a small LightGBM fitted on discovery slice rows predicts
   the holdout slice residual (R^2 against the discovery mean) or error (AUC).
6. **Verdict**: `fixable` > `explainable` > `unresolved` with a reason
   (`no_holdout_fold`, `too_few_holdout_rows`, `analysis_deadline`,
   `tested_corrections_and_residual_model_found_nothing`). Unresolved is a
   statement about this lab, not proof of noise; researchers are told so.

Remaining optimism: multiple proposed remedies are judged on the same inner
holdout, it is one fold of a sample, and the base model was selected adaptively.
Refit changes the base's training population. Selection scoring and closed
confirmation remain the guards; diagnostic support is not a transfer guarantee.

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
- unresolved -> an open question with its reason; not noise.

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
