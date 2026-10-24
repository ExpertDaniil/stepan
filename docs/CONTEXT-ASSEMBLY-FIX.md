# Shared C08/C10 context packing

The user supplied a traceback from `run-20261024T102152Z-9caacf1c`:
`refill -> G03 -> C08 -> ContextBuilder.build_graph -> CONTEXT_TOO_LARGE`.
The earlier `59c633c` patch changed Gateway packing, but left the independent
C08 shrinker unchanged. Thus it did not cover this earlier failure boundary.

The excerpt shows one researcher deferred while the other is still awaiting
an LLM response. It does not establish how the complete run ended, which source
commit ran, or the byte contribution of each context section.

## Confirmed code defects

- C08 and C10 used different shrinkers. C08 could reject a request before the
  newer Gateway packer ever saw it.
- Analysis history copied up to six complete queries. Each accepted query can
  approach 8 KiB and includes investigation prose. The old C08 shrinker did not
  project these queries. This is a concrete unbounded-by-total-bytes contributor;
  its exact contribution in this run needs the complete run artifacts.
- C08 also retained large duplicate protected fact summaries and parameters.
  Protecting an experiment ID need not duplicate its entire plan when the exact
  base and frozen experiment documents remain available.
- The scheduler called `repeated` twice for a single context failure. Thus
  `repetitions=2` was not evidence of two observed attempts.

## Change

`agent.context_budget.compact_research_request` now owns adaptive packing for
both ContextBuilder entry paths and Gateway. The Gateway method delegates to it.
The effective hard limit and the assembly/finalization reserves remain unchanged.

The shared packer projects investigation prose in recent analysis queries only
after full records have been frozen. Artifact IDs, query operations/inputs,
filters and computed check observations remain visible. Exact questions are
reachable through `detail_index` / `analyses/index`. Feedback excerpts and large
duplicate historical plan fields similarly link to scoped source records.
Full task text, semantic task, task contract, restrictions and exact base
code/parameters remain protected. New omission markers live inside the extensible
profile; the closed GContextUser contract remains unchanged.

Both assembly paths use the same source of packing policy as later tool/final
requests. The non-adaptive compatibility shrinkers are preserved. If mandatory
content still cannot fit, ContextBuilder saves a redacted rejected context and
logs `context.rejected` with actual total, limit, top-level/profile field byte
sizes and its artifact ID. The error remains explicit rather than silently
truncating required code or task instructions.

`record_stall` counts one actual outcome once. Infrastructure/context errors,
phase closure and explicit researcher stops use a separate `deferred` flag.
Their log reports the actual count and `heuristic=false`. Two genuinely repeated
actions retain the existing heuristic. A new measurement generation releases
the deferral; another peer's pending dialogue is not cancelled by it. An early
refill context failure also sets the run's degraded flag.

## Validation

Seven additional synthetic regression methods were written, including C08
classification/regression reissue with six long analysis queries, preservation
of frozen sources and the exact measured base, rejected-context diagnostics,
honest repetition counts, generation changes and an actual rolling-refill
failure while the neighbor finishes. Fixtures inject scores; they do not claim
to measure ML quality.

The original user ban on tests, training, full/smoke agent runs, installations
and live LLM calls remains active. Consequently these regressions were **not
executed**. Static AST/schema/resource and whitespace checks passed. The exact
new live-run behavior and ML quality remain unverified.

After lifting the test restriction:

```bash
python3 -m unittest tests.test_context_pressure tests.test_semantic_research tests.test_platform_research -v
python3 scripts/check.py
```

For an already running process the imported code remains the old implementation;
the change is intended for the next launch. Do not treat a source update as proof
that the current process recovered.

## Follow-up verification: GResearchContext, 24 October 2026

At `db75175`, the runtime schema already includes ResearchAction-v1 and the four
metadata fields graph_request_max_bytes, detail_artifact_id, detail_scope and
semantic_contract. Both classification and regression G03 contexts pass this
schema. A dedicated regression also checks that unknown metadata still fails;
the validator remains strict. These are synthetic state-machine checks, not
replays of the user's two unprovided task runs.

Executing the previously unrun regressions exposed a separate packing failure:
a batch of four analysis replies and one knowledge reply remained at 99,192
bytes after packing, above 98,304. Every earlier reply had already been visited
with a 2,048-byte allowance before the later knowledge example was projected.
The packer never revisited those earlier replies when the final batch still
exceeded the complete-body limit.

Packing now performs at most two bounded passes, with 2,048 then 512 byte minimum
display allowances. The second pass runs only while the request is still too
large. Existing service artifacts, paired tool-call IDs, scope, support counts,
exact task/base content and complete observations remain protected; retrievable
group details have explicit omission markers. Irreducible content still produces
an explicit context error instead of raising the configured limit.

Current verification results: [validation-gresearchcontext.json](validation-gresearchcontext.json).
The flat ZIP is checked to contain the same current schema in stepan_resources.py;
an old ZIP or a mixture of versions can retain the earlier schema error.
