# Executable background-search mode

For a solution, use `read_solution({"experiment_id":"best"})` or an exact experiment ID.
Default view is pipeline (code + effective parameters + features + metric). Other
views are code, result, features and diagnostics. Follow next_read for a long output;
it pins the actual experiment ID even if the leader changes. Do not invent section
paths or use a feedback artifact as a source-code reference. `read_memory` is for
task definitions and non-solution reports; section=detail_index lists valid sections.
When submitting pipeline code, declare DEFAULT_PARAMS and optional SEARCH_SPACE;
omit tuning, investigation, test_plan and research_notes in this mode. Later legacy
tuning/scientific examples do not add requirements to executable mode.


# Tools v1

Runtime generates a complete tool catalogue, including the current JSON input schemas,
under `runs/<run_id>/task/TOOLS.md` from `contracts/contracts.schema.json`.

| Tool | Contract | Responsibility |
|---|---|---|
| read_task | C01 | UTF-8 Markdown and unambiguous task-local file resolution |
| profile_data | C03 | Schema and aggregate missing-value report; no raw table in LLM messages |
| write_code | C11 | Syntax/policy check and immutable code artifact |

The LLM research gateway exposes write_code through native tool calls. Reading and profiling
are performed by the bootstrap before research; their results enter the context.
Runner starts a fresh process for every training/refit job. File artifacts persist, Python variables do not.
Only Publisher writes the final submission path. Code checks are guardrails, not an OS sandbox.

## Adaptive tools and detailed contracts

Adaptive exposes analyze_data, read_memory and optional search_knowledge; write_code
is a legacy code-mode tool. Bootstrap reading/profiling is controller-owned.
read_memory takes {reference_id,section,offset=0,limit_bytes=4000}; follow next_offset.
Its reference is authorized by the issued context or delivered branch updates, not
an arbitrary path or artifact ID. It can read full source/diagnostics only through
explicit scoped sections. A linked ID alone is not permission to read its contents.

analyze_data always takes kind, columns, group_by and include_target. Examples use
synthetic placeholders; substitute actual names:

```json
{"kind":"summary","columns":["delta"],"group_by":null,"include_target":false,"derived":[{"op":"difference","name":"delta","columns":["counter_u","counter_v"]}],"filters":{"mode":"all","clauses":[{"column":"delta","op":"lt","value":0},{"column":"zone","op":"eq","value":"rare"}]}}
```

```json
{"kind":"group","columns":["zone","mode","quantity"],"group_by":["zone","mode"],"include_target":true}
```

Filters allow all/any of at most eight clauses: eq, ne, lt, lte, gt, gte, in,
is_missing, not_missing; comparisons accept value or other_column. Legacy
filter={column,value} is conjunctive with filters. No target/target-derived inputs.
Derived expressions allow at most eight stateless missing/log1p/signed_log1p/square/
sum/difference/product/ratio/cross operations, in dependency order. They reuse the
owned operators without fit, arbitrary Python, files or eval. Division by zero and
nonfinite numeric outputs become missing. Learned group statistics are experiments,
not analysis expressions. Filtering occurs before a deterministic <=20k-row sample;
reports give full matching counts and sampled group support. Analysis is serialized,
cooperatively time-limited and admitted as 1 CPU/1 GiB in the shared budget. A bounded
sample exceeding 64 MiB is rejected. CSV reads and native operations are cooperative,
not separately preempted OS jobs. Large/high-cardinality output is bounded and omissions
are explicit. Category coverage compares bounded samples, not entire-population truth.

errors requires experiment_id, explicit input/derived columns and include_target=false;
group_by=null. For a composite error group use a derived cross column. Evaluator uses
trusted saved predictions and selection labels only; no researcher score/labels,
confirmation labels or raw target rows are returned. Groups with support <5 are
suppressed. RMSE reports SSE, group SSE/support and fraction of full selection SSE;
other metrics keep their correct score, with class precision/recall/F1 where applicable.
Different feature groups overlap. A filtered/sampled diagnostic never estimates the
unseen population contribution. Undefined one-class ROC AUC is null.

Schema pages: kind=schema, columns=[], group_by=null, include_target=false,
offset=0,limit=12. Dictionary reads select actual names. Full saved reports can be
paged with kind=report,report_id,columns=[],group_by=null,include_target=false,
offset=0,limit_bytes=4000; follow next_offset (UTF-8 bytes). Frozen earlier analyses
are also in read_memory's analyses/index sections. No cache read expands branch scope.

Tuning syntax:

```json
{"search_space":{"depth":{"type":"int","low":2,"high":8,"step":2},"rate":{"type":"float","low":0.02,"high":0.1,"log":true},"style":{"type":"categorical","choices":["a","b"]}},"max_trials":6,"budget_seconds":180}
```

Names/numbers illustrate syntax, not universal allocations. Use actual allowed model
parameters. int/float require low/high; step must exactly divide the range. log needs
positive bounds; float log cannot use step; int log permits step=1. Categorical choices
may contain JSON null/booleans where supported. Custom tuning requires typed space.
Optional legacy parameters=[names] uses entire catalogue domains; if both are given,
names must agree. max_trials counts new proposals including cache hits/rejections,
not warm-start trials. Admission may shorten wall time. No pruning is currently enabled.
No API calls, package installation or fitting occur in these read tools.

# Semantic planning and exact logical checks

Computing analyze_data calls in semantic-planning-1 contexts include investigation
before observing the result. It contains question, basis_ids, columns,
interpretation, mechanism, assumptions, expected_observation,
alternative_explanation and decision_rule. Metadata/report reads are exempt.
Example: after a sourced subset_count relation [minor,major], a check query uses
kind="check", columns=["minor","major"], group_by=null, include_target=false,
checks=[{name:"nesting_violation",violation:{column:"minor",op:"gt",other_column:"major"}}].
Attach the actual relation ID and explicit same-window assumptions in investigation.
This streams ALL fit rows: violation_rows/evaluable_rows/missing_rows are exact for
the declared predicate. No validation/confirmation labels or raw target inputs.
Filters apply before counting. Derived expressions are stateless and named.
cyclic_sin/cyclic_cos take period>0 and optional origin; the period uses input units.
read_memory additionally exposes semantic_model, quantities/index and relations/index.
Source quotations are checked for occurrence; semantic interpretations remain unverified.
# Context continuation

Completed tool turns may be replaced by a research_transcript observation summary.
Its archive reference supports read_memory section=transcript_index, then the
listed tool/CALL_ID/result or tool/CALL_ID/arguments sections. Exact observations
and code are retained; previews are not new verified conclusions. Reuse successful
checks instead of rerunning them to rebuild the transcript. A fresh dialogue can
read transcripts_index in its frozen details, then the listed transcript sections.

## Error research and source drafts (executable mode)

`run_analysis` optionally accepts `experiment_id="best"` or an exact visible ID.
It aligns fit-only input/targets with that version's inner OOF predictions/errors
and fold IDs in `context['oof']`. `OOF_PENDING` never authorizes selection labels.
Large matrices stay in Python; return bounded aggregates. Diagnostics are sampled
when declared, overlap across feature groups, and do not establish attainable gain.

`edit_solution` accepts exactly one source: `experiment_id`, `draft_id` or `code`.
Optional edits are exact `old`/`new` replacements; each old fragment must match once.
It returns a scoped `draft_id`, including invalid source retained for repair.
A final `{ "action":"pipeline", "code_ref":"returned-draft-id" }` uses that source
without resending it. `drafts_index` / `draft/ID` keep visible drafts in later
bounded dialogues. IDs and assignment/provenance are managed by the system.
