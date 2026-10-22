# Adaptive research v0.6

Полное русское руководство по действующей реализации: [PROJECT_GUIDE.md](PROJECT_GUIDE.md).
Ревизия границ и устранённые причины: [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md).

Дополнение после ревизии: инструменты предоставляются явно через `ResearchTools`; Gateway не владеет данными
и не выбирает вычислительную конфигурацию. Интерпретатор до закрепления задачи/проверки может читать страницы
полного сохранённого профиля через `inspect_schema`. Исследователь видит основание и параметры замороженной
проверки, размеры частей и хеш, но не индексы и ответы holdout. Стоимость ответа вычисляет Budget по плану,
подготовленному диспетчером; интерфейс модели и прогноз времени разделены.

## Problem and scope

The ab9dbc6c run spent 46 of 63 completed evaluations on program-generated local variants.
Its 16 research contexts lost types, missingness and the feature dictionary under the byte cap.
An early target-transform gain became a hard inherited requirement. These are restrictions
of the action interface and scheduler, not evidence that longer LLM replies would fix search.

This change adds a complete path from additional data evidence to a custom fitted pipeline,
while retaining trusted split/metric/publication ownership. It does not claim a benchmark gain.

## Responsibility and decision flow

1. Bootstrap interprets the current task, selects and freezes a validation protocol, and publishes
   a verified Dummy CSV. Catalogue warmups remain affordable initial probes.
2. C09/C10 obtains a model proposal for the next research action. The same bounded dialogue can
   call `analyze_data` before deciding; there is no new coordinator model or parallel agent.
3. `ResearchAction-v1` explicitly selects `catalog`, `pipeline`, `tune` or `stop`. The dispatcher
   authorizes the action, checks resource admission and commits the ordinary ExperimentPlan.
4. Runner owns fit/predict; Evaluator calculates authoritative metrics on the frozen split.
   Results, feature descriptions, hypotheses and evidence IDs enter bounded search memory.
5. Publisher refits the exact source/parameters on all training rows, validates the output and
   atomically replaces the last verified CSV. Generated source is pinned for reconstruction.

The LLM selects research content; Python owns execution permissions, budgets, comparability,
deduplication and publication. A rationale is not proof that a research direction is useful.

## Protected context and on-demand analysis

`profile_summary.data.core_profile` packs column name, dtype and train/test missingness together.
The core, task contract and measured base remain mandatory when verbose profile fields and
redundant history are removed. Schemas and tool replies also count against the API request cap;
the Gateway can compact duplicate profile/history without removing the core or base source.
An oversized mandatory core fails explicitly instead of silently losing all data information.
The compact core includes at most 128 columns and reports omissions. Larger schemas are
discoverable using `kind=schema`, `offset`, `limit` and `next_offset`; knowing omitted column
names in advance is unnecessary.

`analyze_data` accepts at most 12 selected feature columns and these query kinds:

| Kind | Information |
|---|---|
| schema | Complete paginated column names, train/test presence and roles; no values |
| summary | Missingness, cardinality, numerical quantiles/zero counts, frequent categories |
| association | Spearman correlations of selected numerical fit columns |
| group | Fit-only target summaries by a selected categorical column or numerical bins |
| compare | Fit versus unlabelled test feature summaries |
| dictionary | Available feature metadata for selected columns |
| report | Retrieve a saved analysis by report_id without recomputation |

The service is constructed on the dispatcher and reads selected columns in 4,096-row chunks.
It drops holdout rows, optionally filters by equality on an input feature, counts the complete matching
population and then keeps at most 20,000 uniformly sampled rows with a fixed seed. Rare groups therefore
remain queryable before sampling. Query execution does not touch SQLite. Holdout labels and raw labelled
rows are never returned. Target summaries require `include_target=true`; feature comparisons reject that flag.
The service checks phase/global time between chunks and uses a 120-second read allowance; one chunk read is
cooperative rather than preempted. It reports population/sample sizes and saves the bounded full report.
API replies have their own 6 KiB detail budget and state omissions. These are resource safeguards,
not statistically justified sampling rules or search allocations.

## Custom pipelines

`action=pipeline` requires explicit source, `recipe_id=python-pipeline-v1`, arbitrary finite
JSON parameters without reserved underscore names, and declared feature steps. Catalog actions
cannot supply source. Code is syntax/policy checked; it must export `make_pipeline(task_spec,
params)` returning fit/predict. Learned preprocessing belongs inside the returned estimator.
The plan can change features, preprocessing, target treatment, model and parameters jointly.
The authoritative score supports that complete solution; it does not isolate component effects.

The custom manifest records source, effective parameters and declared feature operations as
`declared_only`. It never certifies feature capabilities from parameter names. Deliberately
created controlled branches can still reject unverifiable custom code under hard constraints.

Generated pipelines run in the existing watched worker with sanitized environment, thread
limits and the same timeout/RSS controls. A worker IO audit guard denies raw artifact/dataset
opens, network, subprocesses and file mutations during construction/fit/predict. Installed
code remains importable. Joblib uses threads inside these workers. The guard is disabled only
for trusted output writing/snapshot code. AST checks and the Python audit guard are defense
in depth, **not an OS sandbox or a proof of arbitrary Python semantics**. Estimators requiring
extra files/processes may fail; the verified incumbent is preserved. No packages are installed
and no external/pretrained models are fetched during a task.

## Explicit tuning and empirical decisions

`action=tune` authorizes named axes of one owned catalogue recipe, a total trial count and a
wall-time allowance including the initial fit. The dispatcher may shorten the allowance to
protect the current phase and finalization reserve. Continuations rebase on measured feedback,
respect the selected axes and stop when count/time is consumed. Each fit keeps its watchdog.
Time is checked between admissions, so one already running fit can outlast the series allowance;
the fit and global job deadlines still apply. Limits of 16 trials and 3,600 seconds are bounded
tool capacities, not recommended allocations or promises of quality.

In adaptive mode an ordinary catalogue/custom proposal creates no implicit tuning series.
`RESEARCH_SERIES_SIZE` controls legacy series only. After LLM unavailability/quota exhaustion,
owned catalogue fallback remains possible and is logged; it never runs generated code silently.
A voluntary adaptive `stop` is respected. Two peers may propose different actions and see
compact measured results from their shared visible lineage.

Successful empirical transformations do not automatically trigger G14 component transfer or
an accept/reject fork in adaptive mode. A solution remains available in history without making
its transformations mandatory for descendants. G06/G02 remain available for deliberately
requested controlled component studies. Explicit restrictions retain their existing enforcement.
Final refinement restores the measured best but does not force empirical feature settings to
remain fixed in adaptive mode. Legacy catalog/code modes retain their previous branch policy.

An explicit finite tuning allowance prevents a single authorization from running indefinitely.
It does **not** guarantee semantic novelty or prevent the planner from repeatedly requesting
similar work. No direction quotas, bandit scores or RL policy are introduced without measurement.

## Compatibility and validation

`RESEARCH_MODE=adaptive` is the CLI/config default and is shown in `.env.example`. Existing
`.env` files explicitly setting `catalog` must be changed to enable it. `catalog` and `code`
remain compatibility/comparison paths. C00-C24 and ExperimentPlan wire schemas remain unchanged;
GResearchContext allows ResearchAction-v1 and two internal schemas document its producer/tools.
The flat package includes the new modules and readable prompt. Reconstruction keeps the custom
source and enables the same worker IO guard for refit.

Focused synthetic checks cover protected missingness, fit-only analysis, unlabelled test
comparison, tool-result-to-pipeline dispatch, generated-code graph admission, explicit tuning
limits, optional controlled branching, blocked raw data IO, classification and regression
training/publication, and byte-identical predictions after pinned-runtime reconstruction.
The release check also runs the existing state-machine/process/packaging tests.

Infrastructure tests establish executable behavior and regression coverage. Live Qwen behavior,
the organizers' exact pinned environment and public/private benchmark quality require separate
measurement. Compare catalogue and adaptive policies under equal time/API budgets, use a
control split not exposed to the planner, and measure final quality, completion rate, repeats,
action distribution and wall time. More feature operations alone are not success.
