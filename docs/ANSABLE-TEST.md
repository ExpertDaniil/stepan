# ansable_test: architecture specialists and fit-only ensembles

Latest runtime corrections and exact verification limits:
[ANSABLE-RUNTIME-FIXES.md](ANSABLE-RUNTIME-FIXES.md).

Experimental branch from the published `de7a28c3f6dc2082369a803df41a8e114160beeb`.
The original development checkout and its uncommitted work were left intact.
The user requested that further tests and ML runs stop; the final changes below
have not been exercised end to end. No live Qwen quality comparison or public
benchmark improvement is claimed. No platform submission was made.

## Run

Use the organizers' environment, or install the pinned requirements.runtime.txt
for development. The experimental subset now includes LightGBM 4.6.0,
PyTorch 2.5.1 and cloudpickle 3.1.1, all from the organizers' approved list.
CPU PyTorch wheels may be installed during environment preparation; the agent
never installs packages or downloads weights during a task.

```bash
git fetch origin
git switch ansable_test
git pull --ff-only origin ansable_test

ENSEMBLE_RESEARCH_ENABLED=1 GRAPH_PEERS=5 RESEARCH_MODE=adaptive BACKGROUND_TUNING_ENABLED=1 python solution.py "data_for_participants/Регрессия/task.md" --env-file .env
ENSEMBLE_RESEARCH_ENABLED=1 GRAPH_PEERS=5 RESEARCH_MODE=adaptive BACKGROUND_TUNING_ENABLED=1 python solution.py "data_for_participants/Многоклассовая классификация/task.md" --env-file .env
```

API configuration remains OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL from
the existing explicit configuration. No secrets are committed. The supplied
Qwen is used; no alternate provider is selected. Older `.env` values such as
RESEARCH_MODE=catalog or BACKGROUND_TUNING_ENABLED=0 must be changed for this mode.
The public settings are also documented in recipes/ansable.env. In the flat
platform ZIP, recipes/platform-env.json enables this experiment by default.
Use ENSEMBLE_RESEARCH_ENABLED=0 for the ordinary source runtime mode.

Build when ready:

```bash
python scripts/package_solution.py --output dist/ansable_test.zip
```

The existing packager owns the allowlisted flat ZIP and preserves official
requirements.txt byte for byte. This checkpoint does not include a verified ZIP.

## Five researchers, shared execution capacity

| Role | Owned estimator | Data research |
|---|---|---|
| native_categories | CatBoost | Meaning of categories, missingness, plausible row links |
| numeric_boosting | LightGBM | Scales, relations, aggregates and safe supervised representations |
| randomized_trees | ExtraTrees | Alternative partitions and complementary failure modes |
| structured_response | Linear model | Normalization, ratios, smooth basis functions and systematic bias |
| neural_representation | Small Transformer / MLP | Learned feature interactions and embeddings |

Five graph peers are independent research contexts. MAX_WORKERS still limits
training processes, normally two; Budget leases account for all CPU/RAM/analysis.
There are no five independent full-resource training jobs. Missing optional
packages produce an explicitly labelled fallback role. PyTorch networks are
trained from scratch. FT attention acts between a row's features and does not
establish object identity or recover missing history.

Researchers submit short questions, owned feature programs, or make_features
source returning a fit/transform estimator. The executor appends the assigned
predictor. Explicit tune actions and differing catalog model parameters are
rejected with OPTIMIZER_OWNED. Joint feature changes remain allowed. There are no
mandatory investigation/test_plan questionnaires and no lifetime direction cap.

## Optuna clusters

Each architecture cluster contains version-owned studies. A representation's
source/program stays fixed while code searches typed model axes. Class weighting
is available to the automatic classification study. Versions retain their own
measurements, costs and optimizer state; a stronger new initial score reduces
older versions' priority instead of deleting their studies. Temporary resource
failure/timeouts can be retried, while deterministic initial failure waits for
changed source/input. In specialist mode a newly queued representation does not
automatically kill a running trial. Next actions are admitted against fit, full
refit, inference/output and finalization reserves.

Priority combines measured quality, observed return, uncertainty, cluster
coverage, age and predicted cost. This is a transparent heuristic, not a proven
optimal policy. It has not been compared with the old scheduler at equal budget.

## Sixth component: numerical ensemble coordinator

The coordinator uses existing dispatcher/Runner/Evaluator/publication paths; it
does not require a sixth Qwen conversation. It selects measured architecture
representatives, describes aligned validation residuals/disagreement and proposes
an optional ensemble when the whole fit/refit cost fits Budget. Committee size
may fall below five when resources do not support all members. Multi-target
ensemble proposals are currently skipped; ordinary adapters remain available.

Every ensemble worker relearns the COMPLETE representation/model pipeline in
each inner fold. Its weights are nonnegative and sum to one. A small internal
meta-check can reject blending in favor of a single member. Temporal folds use
only earlier periods; missing first-period OOF predictions remain missing.
Group scenarios keep inner groups apart. The independent outer Evaluator scores
the resulting fitted ensemble; a fit on selection labels never manufactures an
authoritative ensemble score. Confirmation labels remain closed until finalization.

For Macro F1 labels, class-decision adjustments can be selected inside fit OOF
and accepted on the internal meta-check. They are saved with the model and only
affect predict; predict_proba is not declared calibrated. Rare unsupported
classes retain their original adjustment. This is still an adaptive internal
comparison, not statistical proof of generalization.

The best verified CSV stays published until a complete replacement passes the
ordinary validator. Confirmation ranking is separated from release affordability;
previous published artifacts can be reused only when data/task/code/runtime and
output hashes still agree. A costly ensemble is not automatically the winner.

## Semantic memory and context

Reuse the existing task model, graph lineage, immutable decisions, semantic
revisions and diagnostics. Meaning, units, availability and proposed row links
keep their source and uncertainty. Model prose never becomes verified fact.
Optional semantic updates are reference checked in executable-pipeline mode too.
Measurements retain task/data/protocol/code identities and costs.

Each researcher receives its architecture, best measured owned representation,
task/protocol, semantic summary, error recommendation, active/queued studies,
ensemble complementarity and remaining shared budget. Fresh feedback is delivered
while other work completes. Large representation source is stored exactly in the
frozen context and available through read_memory section=specialist_reference or
read_solution; the executor does not reconstruct it from a truncated summary.
Other scoped sections include pipeline_search, ensemble, semantic_model, decisions,
analyses and transcripts. Existing byte accounting and paging remain authoritative.
This preserves retrievable source history, not a promise that all history fits
simultaneously in Qwen's active context or that any compression is semantically perfect.

## Pro findings used here

- No hardcoded public columns, entity keys, target threshold or two-regime gate.
- Native HGB/LightGBM/CatBoost recipes remain distinct from legacy ordinal HGB.
- Target statistics are optional, fit-local, inner cross-fitted and strictly
  as-of for temporal inference; their training encoding differs from inference.
  Unknown groups fall back to an available prior, classification order is fixed,
  composite keys are typed/length-prefixed, and memory/cardinality are bounded.
- No global precomputed OOF feature matrix is reused across an enclosing fit.
- A failed combined experiment does not refute every component or an entire family.
- The renewable diagnostic queue and source-version checks are reused.
- Diverse architectures are a hypothesis about complementary errors, not proof.

## Current verification boundary

Runtime traceback follow-up: background submissions now bind C12 parent_ids
through the existing bind_trusted helper to the exact execution work order,
matching typed lineage. Composite recipes explicitly authorize visible, measured
representation donors on the same task/protocol as idea_donor edges; graph v1's
unsupported fitted-model ensemble compiler remains disabled. Budget trimming also
trims the recorded member IDs. This fix was reviewed from source without running
tests or ML jobs, as requested by the user.

Some component checks were run before the user's stop request, including small
CPU neural fits, OOF ensemble serialization and specialist context construction.
They do not cover the final snapshot. An early classification smoke exposed a
missing deepcopy import; it was corrected without rerunning. A coordinator read
of a nonexistent job-event phase field and a cold representation duration estimate
were also corrected. No further tests or ML runs were started after the request.

Before relying on results, the user should run both tasks and inspect
ensemble.candidate_queued, pipeline.execution_started, architecture_clusters,
research.action_proposed, validation.finalists_confirmed and submission.published.
Check that all expected roles are available, feature work actually occurs,
ensemble candidates are evaluated, jobs stop by deadline, and final CSV format
is valid. Paid Qwen policy quality, equal-budget ablations, fresh public scores,
unknown-task transfer and the full three-hour workload remain unmeasured.
