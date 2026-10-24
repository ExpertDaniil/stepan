# Open investigation, compact context and platform verification

This update starts from remote `07f22e286cb495e9435c4083ecebbc586baa897e`, integrates
team commits `b84db1f` and `1c8b3b4`, and preserves
its runtime, fixed validation, independent evaluator, branch visibility and final
CSV publication. Runtime dependencies and official requirements are unchanged.

## Research interface

`RESEARCH_PROTOCOL=compact` is the default. `research-v2` uses one investigation
record. Legacy `research_notes/test_plan` artifacts remain readable and executable.
`RESEARCH_PROTOCOL=legacy` restores the older prompt and scientific-v1 record
requirements for comparison. `DATA_RESEARCH_MIN_SECONDS=0` is the default: there is
no compulsory 30-minute feature phase. Explicit legacy windows remain supported.

ResearchAction accepts `research` without an estimator or fit. `next_step=continue`
saves a checkpoint and requests a fresh scoped dialogue. `wait` and `finish` close
the work order without a fit. The scheduler wakes on new measured feedback or a
new distinct analysis operation. A checkpoint alone does not advance the information
generation; repeated empty continuations defer the peer. These are scheduling
heuristics, not a claim that two analyses exhaust a scientific question.

The gateway normalizes mechanical defaults and runs the Researcher provenance,
feature dependency, parameter, code and semantic checks inside its existing bounded
repair dialogue. The dispatcher still rechecks admission against current resources
and live activity. Training/analysis metrics never come from model-authored prose.

## Python analysis

`PYTHON_ANALYSIS_ENABLED=1` exposes `run_analysis` alongside `analyze_data`.
The model supplies `question`, `columns` and code exporting:

```python
def analyze(X_fit, y_fit, X_test, context):
    return {"missing": X_fit.isna().sum()}
```

`include_target=true` supplies only fit labels; otherwise y_fit is None.
`compare_test=true` supplies unlabelled test columns; otherwise X_test is None.
The deterministic fit sample is at most 20,000 rows / 64 MiB. Population counts
and sampling status accompany every report. This sampled tool does not establish
full-population invariants; use the exact streamed `analyze_data kind=check` when
appropriate. Select at most 40 non-target, non-excluded columns.

One separate process runs at most 60 seconds and one CPU under the shared 1 GiB
analysis slot. It has a clean environment without API credentials. AST/import and
Python audit IO guards block declared filesystem/network/subprocess operations;
these are defence in depth, **not a hostile-code OS sandbox**. Peak-RSS watchdog,
process-group cleanup and bounded stdout/results limit ordinary faulty generated
code. No runtime installation is performed. Reports, failed execution tracebacks
and exact source are retained. Identical computations are cached across dialogues;
changed explanatory wording alone does not create a new computation.

## Working context and memory

Full state is frozen in scoped artifacts before display compaction. Exact base
code and base parameters can move to `read_memory` sections `base_code` and
`base_result`; `full_context` preserves the original snapshot. Only the controller
hydrates those fields for validation/parameter inheritance. Detailed reports,
recipes, old facts and feedback can similarly move behind explicit references.
Duplicate current/base parameters are projected once. The task contract, restrictions, source identities and input-column map are retained.
No provider input limit is raised. Oversized irreducible task contracts still fail
explicitly with `context.rejected` byte diagnostics; arbitrary source documents
cannot be guaranteed to fit a finite context window.

Actual tool-execution time is budgeted separately from LLM waiting. The whole
dialogue and global publication reserve remain bounded. Checkpoints preserve the
question and evidence when a fresh dialogue is needed. If a tool transcript still
exceeds the wire budget after compaction, a controller-authored checkpoint retains
completed reports and requests a fresh dialogue; it is never labelled a model conclusion. Typed semantic updates are
validated before dependent feature interpretation against a provisional model;
original sources are immutable and model revisions remain unverified.

Every fit decision records parent experiment IDs, code SHA-256, dataset/protocol
identity and fingerprint. Analysis decisions retain source/report IDs without an
invented model score. Existing task/branch scope and conflict handling remain active.
Offline experience retrieval now incorporates the latest research question;
project method cards already include implementation examples and caveats.

## Runtime corrections

Cold warmup measures bounded ExtraTrees/HGB pilots before expensive CatBoost;
research can still choose larger configurations. No dataset-specific names are used.
All worker jobs use joblib threading under their existing CPU limits, preventing
unleased nested loky process pools. Final preparation/publication idempotency keys
include the current champion version, allowing confirmation to return to an earlier
candidate. Unlimited output budgets support extended deliberation. Reaching a
publication/phase deadline is distinguished from an independent provider failure.
The last verified CSV remains available across later failed trials.

## Verification entry points

```bash
python scripts/check.py --jobs 3
python scripts/demo_data_research.py --directory /tmp/research-demo --kind both --scientific
python scripts/package_solution.py --output /tmp/solution.zip
python scripts/verify_submission.py --zip /tmp/solution.zip --directory /tmp/zip-replay
python scripts/compare_research.py --task /absolute/task.md --output-dir /tmp/comparison --seconds 600 --modes ABCD --env-file /absolute/.env
```

`verify_submission.py` copies only root Python files and exact requirements, then
runs both synthetic task kinds through real Python EDA, a no-fit checkpoint, a
follow-up analysis, an owned feature fit and checked publication. Only LLM replies
are scripted. No repository PYTHONPATH is available to the child.

`compare_research.py` freezes one ZIP before all four conditions, preserves a fixed
seed/task/time budget and records per-run protocol events and evaluator reports.
A=legacy/standard tools, B=compact/standard tools, C=legacy/Python, D=compact/Python.
Repeated runs rotate order. A short single repeat is execution evidence, not a
statistical quality claim. Actual results and limitations belong in
`validation-open-investigation.json`; no target leaderboard score is promised.
