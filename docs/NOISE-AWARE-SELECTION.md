# Conditional support and stable research references

`agent/significance.py` compares saved prediction vectors on selection rows.
RMSE/MSE/MAE/R2, RMSLE/MSLE, MAPE/SMAPE, accuracy and log loss use aggregated
row losses; other metrics are recomputed on each resample. Aggregation is batched
to keep memory bounded. No confirmation targets are read.

The evaluator chooses units from the frozen validation protocol:

- group holdout: resample entire declared groups;
- temporal holdout: resample entire observed validation periods;
- duplicate-group random holdout: resample observable duplicate feature groups;
- otherwise: IID rows, an explicit assumption rather than proof of independence.

A conditional bootstrap win fraction >= 0.95 and relative gain >= 0.1% gives
`significant=true` for compatibility with the previous internal wire shape.
`bootstrap_win_fraction` is its descriptive name; `probability_better` is retained
as an alias. This is not a posterior probability of transfer, a correction for
adaptive multiple search, or an estimate of training-seed variance. Dependencies
not represented by the protocol remain unknown.

With one observed period (or one group), between-unit uncertainty is unavailable.
The result has `significant=false`, a null win fraction and an explicit reason.
A material positive selection gain (the same 0.1% policy floor) may still advance
publication and the utility reference; it is reported as `unconfirmed_gain`, not
as statistical support. Thus a short history does not freeze useful measured
progress or manufacture thousands of independent periods from their rows.

## Where it acts

- Search publication retains the last verified CSV until the candidate passes
  this policy and refit/output validation. Final confirmation is independent of
  this publication heuristic and runs only after research closes.
- Research paired verdicts use the evaluator's same cached protocol-aware
  comparison. Non-supported positive gains are `within_noise` or
  `unconfirmed_gain`, with unit/count/unavailability details.
- Each branch's utility reference advances under the same policy. Advice,
  feedback, fresh context snapshots and `read_solution("best")` use that reference.
  Raw results and explicit experiment IDs remain readable; `Store.best()` keeps
  its exact raw ranking for evaluation, export and finalization.
- Background trial changes alone do not wake a waiting researcher. A supported
  utility change, candidate outcome, diagnostic or controller feedback can;
  timed recovery and explicit continuation retain their existing behavior.

Tests include four dependent groups repeated hundreds of times: IID row
resampling promotes a gain that group resampling leaves unsupported. Duplicating
those rows again does not increase the group win fraction. Tests also cover a
single temporal period, class metrics, stable aliases, branch isolation and
research recovery without parameter-trial churn.
