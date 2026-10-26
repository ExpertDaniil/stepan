# Noise-aware acceptance

In the reviewed regression run every result after minute 23 was the same
representation with jittered LightGBM parameters (36.79 -> 36.49 RMSE). The
team's own fixed-parameter ablation shows about 0.7 RMSE between seeds, and
about 300 candidates were ranked on one selection set: the argmin was largely
selection noise, yet each new argmin triggered a refit and was reported to the
researchers as progress.

`agent/significance.py` compares two fixed prediction vectors on the same
selection rows with a paired bootstrap of the declared metric:

- metrics that are means of a per-row loss (RMSE/MSE/MAE/R2, RMSLE/MSLE,
  MAPE/SMAPE, accuracy, log loss) resample weight vectors (200 resamples,
  vectorized); others (F1 family, kappas, MCC, AUC, ...) are recomputed per
  resample (100);
- an improvement is real when P(candidate better) >= 0.95 and the relative
  gain is at least 0.1% of the reference value.

A consistent small gain (every row a little closer) passes; parameter jitter
that helps some rows and hurts others does not.

Where it acts:

1. **Publication during search.** `Orchestrator.noise_gate` keeps the
   published champion while a newer raw-best is not significantly better
   (`submission.noise_gate` logs each decision once per pair). This saves the
   refit and keeps a measured output instead of an indistinguishable one. It
   never acts during finalization: closed confirmation still chooses among the
   finalists on its own rows. `NOISE_AWARE_PUBLICATION=0` disables it.
2. **Research verdicts.** Paired comparisons of a research candidate with its
   base report `significance`; status `overall_improved` now requires it,
   otherwise `within_noise`. Researchers see `paired_verdict` in
   `candidate_results`, and the journal line shows `P(better)`.

`Store.best()` stays the raw argmin: graph export and research references
depend on that invariant. Seed variance of the learners themselves is not
re-measured here (that would cost extra fits); the strong learner already
averages three seeds, and averaging the final top configurations (a separate
change) reduces it further.
