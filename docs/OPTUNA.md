# Optuna research segments

Adaptive `action=tune` now uses the organizers' pinned Optuna 4.2.1. The dispatcher calls
`Study.ask(fixed_distributions=...)`, submits the returned complete configuration through the
existing Runner, and calls `Study.tell` only with a committed valid Evaluator observation.
No model trains inside Optuna and no LLM call is needed per trial. There is no pruning in this
first integration. CPU, RAM, process cancellation, admission forecasts and final CSV ownership
remain unchanged. No runtime package installation is attempted.

For local development install the updated requirements.runtime.txt before starting the run.
The official requirements.txt/requirements.organizers.txt are unchanged. If Optuna/dependencies
cannot import, the tuning request receives OPTUNA_UNAVAILABLE feedback while the last verified
CSV remains available. Availability in context means discoverability, not a successful import.

## Research flow

The prompt separates research from mechanical parameter search: inspect data/validation,
establish affordable baselines, explore promising feature/preprocessing/model combinations,
and request bounded tuning when its expected value is higher than another concrete experiment.
A short initial calibration search can avoid eliminating a family because of poor defaults.
After tuning, the researcher receives the actual configurations, gains, failures and costs and
can return to data, change family, build an ensemble or authorize another segment. There are
no mandatory phase order, direction quotas, one-change rule or claim that a ceiling is known.

## Contract

Use `tuning.search_space` with 1..12 named domains:

```json
{
  "search_space": {
    "max_iter": {"type": "int", "low": 500, "high": 1500, "step": 100},
    "learning_rate": {"type": "float", "low": 0.02, "high": 0.1, "log": true}
  },
  "max_trials": 12,
  "budget_seconds": 300
}
```

Values illustrate syntax, not a universal allocation. int/float require ordered finite bounds.
Optional positive step must divide the interval; bounds are never silently rounded down. Log
domains require positive low; float log has no step, int log allows step=1. Categorical domains
use `choices` of distinct finite JSON scalars, including null/booleans where the recipe allows.
Bounds/choices must fit owned domains and explicit branch restrictions. All searched parameters
must have an initial value in the normalized anchor. Unsearched parameters stay fixed.

Legacy `parameters=[names]` is accepted for owned recipes and derives their full declared
domains with linear numeric sampling. If both fields are supplied, names must agree. Intended
ranges written only in prose have no execution effect; the current prompt requests typed space.
The existing 2..16 proposal and 1..3600 second capacities remain interface bounds, not recommended
allocations. A segment counts newly asked proposals, including cache/pending/admission rejections;
verified warm-start history does not consume those slots. An evaluated/claimed finite space ends
without endlessly asking for duplicate points. Cached exact known points reuse measurements;
Optuna does not promise every sampled point is new. Failed/admission-rejected points have no
fabricated metric and are recorded as FAIL, not PRUNED.

The anchor is a reference and is not implicitly trained outside the requested space. Owned
recipes keep their owned factory. Custom tuning uses explicit source, or reuses the current
measured custom base_code with code=null and its existing feature manifest. Custom source is
checked under the same generated-code policy; it never imports/runs Optuna. Custom domains
are explicit; no parameter semantics are inferred from arbitrary source. Source and fixed
parameters remain constant within a study; parameterized feature choices may be searched
explicitly and are still declared-only for arbitrary Python.

## Scope, execution and memory

Studies are keyed by run, graph branch, task/data/validation/metric, source SHA, seed, fidelity,
model family, fixed parameters and the normalized search space. Owned feature manifests are
derived from candidate parameters; custom declared manifests are part of scope. Identical
compatible authorizations reuse a study within the same run, including across peers on one
branch. Changing source, a fixed preprocessing choice, validation or domain creates another
study. Only visible lineage/explicit parent observations are imported. There is no cross-task
reuse or automatic resume of an old process.

TPESampler uses the configured seed and Optuna's ordinary startup behavior (10 startup trials
in the pinned version). Small budgets may therefore mostly sample startup points; superior
quality to random search or the previous tuner is not claimed. Seeded sampling does not imply
an identical parallel search trajectory when result arrival order differs. Saved configurations,
source, seeds, metrics and trial history support inspection and pipeline reconstruction. Search
replay across process restarts is not implemented.

Each proposed trial is bound to the actual Store experiment after G04/C12 reservation. Its
effective fingerprint is checked before an observation is accepted. Different studies may
propose the same point; Store/Graph exact deduplication still prevents duplicate fits. Pending
claims within one study prevent one peer from releasing another peer's claim. Every admission,
training failure, invalid prediction or cancellation remains visible without an invented score.
An optimizer bookkeeping error disables further asks and does not invalidate a committed ML
evaluation. The previous cheap owned-catalogue emergency fallback after API unavailability and
legacy catalog/code search remain available; they are not reported as Optuna authorizations.

`optuna/series-*-REV.json` snapshots record source_plan_id, study/series IDs, domains, trial to
plan/experiment/work-order links, outcomes, elapsed wall time, counts, before/after best and
stopping reason. They also retain the study's trial parameters/states/history. At the next
context construction, research_memory attaches series to their original decision, resolves
metrics through visible committed Store evaluations, and includes those experiment IDs in the
question's tested_experiment_ids. Ancestor snapshots after a fork event cut are excluded.
Exhausting a search segment never closes an idea or proves a model/feature direction exhausted.

Deadlines apply to queued/executing work: per-fit timeout is capped by the segment/phase time
remaining when a trial is proposed, and ordinary admission/lease checks run again before fit.
The existing process watchdog and global reserve remain the final controls. Forecast errors
and dispatcher/cleanup overhead mean this is not a promise of exact wall-time allocation.

## Verification status

Added regression cases cover typed joint ranges, immutable anchors, compatible history,
study reuse, custom-source preservation/reuse, pending claims, Evaluator-to-Optuna-to-memory results,
optimizer bookkeeping failure, expired segments, invalid domains and matching wire schemas. Existing adaptive count semantics
were updated for newly asked proposals. Tests, training, live API calls and package builds were
not run here at the user's explicit request. No quality/completion-rate gain is asserted.

API references for the pinned version: [Study ask/tell and add_trial](https://optuna.readthedocs.io/en/v4.2.1/reference/generated/optuna.study.Study.html)
and [TPESampler defaults](https://optuna.readthedocs.io/en/v4.2.1/reference/samplers/generated/optuna.samplers.TPESampler.html).
