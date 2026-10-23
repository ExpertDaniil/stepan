# Feature representation research

The previous run demonstrated only one generated feature pipeline among 34 evaluated
experiments. Availability of custom Python did not ensure representation exploration.
This update adds executable mechanisms, not a promise of a lower RMSE.

`features` is an adaptive ResearchAction. The LLM selects named, composable operations,
an owned base recipe and its parameters. The controller validates inputs against actual
feature columns and compiles a frozen `python-pipeline-v1` plan. Original inputs remain;
operations can consume earlier generated columns. Multiple changes are permitted.
Custom Python `pipeline` and explicit custom tuning remain available.

Feature-focused assignments rotate between existing peers/rounds. They admit `features`,
a changed declared custom input-feature pipeline, or `stop`; model-only tuning does not
consume that assignment. This guarantees attention, not useful features. A stop is respected.
`FEATURE_RESEARCH_ENABLED=0` disables rotation and controller seeds for comparison.

The dispatcher can run up to eight generic seed programs per visible branch while a different
peer reasons. They use fit-only missingness, numeric spread and categorical cardinality;
they do not encode public-task column names. Seeds are explicit controller-authorized
hypotheses, not LLM decisions, and are logged as such. All use ordinary graph work orders,
leases, frozen split, output checks and finalization. There is no third peer or SQL writer.
`FEATURE_SEED_BUDGET_FRACTION=.15` caps observed seed wall time relative to the task budget;
the currently admitted seed can overshoot the soft share by one fit. Its hard lease remains
bounded by `FEATURE_SEED_TIMEOUT_SECONDS=120` and the existing finalization reserve.
Once the small seed bank is exhausted, LLM research continues. No automatic Optuna series
is invented while waiting. A seed's measured source and parameters appear in feature_memory.

The owned transformer includes arithmetic, missing indicators, safe logs, categorical
crosses/frequencies, grouped mean/std/rank/relative value, and datetime components.
Frequencies and grouped reference distributions fit without targets on fit rows only.
Validation/test batches never recompute these statistics. Unknown groups return missing;
unseen frequency is zero; unsafe divisions/nonfinite values become missing. No clipping
or removal of rows is implicit. Output names cannot overwrite original/excluded columns.

`feature_memory` is built by the dispatcher from visible, trusted evaluations. It includes
diverse program identities, complete parameters, scores and immutable source references,
with explicit omission counts. A custom source's feature manifest is a declaration;
it is not independent evidence that every declared operation ran or caused improvement.

References informing the design (implementations are project-owned):
- https://github.com/nikhilsab/LLMFE : executable program evolution and experience buffers.
- https://github.com/IIIS-Li-Group/OpenFE : grouped/categorical operators and candidate selection.
- https://github.com/noahho/CAAFE : context-aware proposals and measured feedback.
- https://arxiv.org/abs/2602.17641 : instrumented feature evaluation; reported average gains are modest.

Verification in this session is static only, per the continuing user restriction. Synthetic
checks in `tests/test_feature_programs.py` are written, not executed. No agent, training,
paid API request, dependency installation or test suite has been run.
