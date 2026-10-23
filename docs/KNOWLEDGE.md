# Offline research knowledge

Adaptive research automatically loads recipes/knowledge and seeds each context with one
method and one relevant upstream excerpt. Each peer can independently call search_knowledge
within the existing tool dialogue. It returns content immediately, not a list of URLs.
There is no extra coordinator, embedding model, network fetch or new package dependency.
The existing last decision turn remains tool-free; knowledge does not increase the API quota.

## Contents and provenance

The pinned AutoMind input has 450 competitions and 3307 normalized texts. The initial selection
includes 31 regression competitions, 11 multiclass competitions and one binary competition
retained for generic encoding methods: 43 competitions and 154 source texts in total.
Selection is an explicit reviewed task/modality scope, not blind acceptance of the original tags.
selection.json records a reason for every included/excluded competition. Exclusion from this
initial selection does not establish that a source is useless. Structural/time/group cases are
marked separately. Survey/storytelling, multilabel and unrelated modalities are not treated as
ordinary multiclass tabular tasks.

There are 17 project-authored method cards and four complete project-authored make_pipeline
examples: missingness flags, frequency features, selected log1p features and constant-column
removal. Examples keep learned state inside fit and forward the owned runtime parameters to
build_pipeline. They are reference strings, never automatically executed by the knowledge service.
They have not been locally executed and do not establish a score gain.

Original source text, source hash, competition description/URL, solution title, task scope and
metric are retained. A competition URL is not represented as the URL of a particular discussion.
Links-only stubs and code-like prose have explicit statuses; neither is a runnable notebook.
Related_source_ids are reading suggestions, not proof that the owned card is an exact extraction
or that its individual effect was established in those publications.

The linked Mercedes 11th-place notebook was inspected at pinned tree
30d9a6e95977b68d23971aba2919fc2a82259161 (notebook blob
32c50e6f517842165c8057d4d6923007b43d9a19). The constant-feature example is an independent
adaptation. Its original target-dependent grouping, test-distribution filtering, local helper
imports, Python-2 syntax, file IO and 100-fit bagging are not copied into the example. The
original notebook remains linked rather than being advertised as Python-3.12-ready code.

manifest.json pins input and generated-resource hashes. UPSTREAM_LICENSE.txt preserves the
repository notice; it does not establish separate rights to all third-party write-ups.

## Retrieval and use

search_knowledge accepts query and optional tags, ids, offset and max_results. A query searches
the current task's method/write-up pool; tags are soft relevance hints rather than an AND filter.
Reading known IDs returns complete method/examples or a paged original write-up. Unknown IDs
return explicit tool errors; unknown tags are reported and do not block textual matches.
match_info explains matched/unmatched tags and query terms. Each source page states offsets and omissions. A complete
example is omitted as a whole rather than cut into invalid Python. Response size safeguards
do not reinstate client input-context limits.

Ranking uses a deterministic lexical rarity score, a modest method-card preference and at most
one post per competition in a search result. These are initial retrieval heuristics, not a learned
allocation policy. ID reads can retrieve another post of the same competition. Context seeds
use feature types/missingness plus the current deferred/recent research question, omit
target/excluded columns from the profile query, and do not read frames or labels.
External data, leaderboard manipulation and instructions in sources never override runtime rules.

Owned preprocessing policy is shared between construction and context: numerical median
imputation/scaling and the categorical missing sentinel use agent/preprocessing.py. CatBoost's
numeric NaN/native string categories differ explicitly from HGB's median/ordinal pipeline.
Generated pipeline semantics remain source-defined. The service repeats current-base facts next
to retrieved experience and highlights metric/native-NaN/structure mismatches. This reduces
ambiguity; it does not guarantee that free-form model prose never contradicts those facts.
Package discoverability is reported separately from successful native imports.

knowledge_ids is an optional ResearchAction field (default empty). C09 accepts citations only
to items returned in its context/dialogue or previously read items preserved in visible research
memory. Memory carries provenance, not a replacement for the source content. Those IDs and knowledge retrieval artifacts
are not accepted as current-task measurements in evidence_ids. ExperimentPlan and fingerprints
retain execution semantics; reference choices do not change a plan's identity by themselves.
research/intent-plan-*.json saves citations and retrieval artifact IDs.

The immutable corpus/index is shared by peers; responses/cached results are copied. Artifacts
are written through Artifacts, never SQLite from research threads. Knowledge failure is logged
and the ordinary search can proceed without the optional tool. No source is downloaded, no
implementation is installed, and no experience is learned across task runs at runtime.

## Distribution and preparation

The six readable knowledge resources are included in the flat platform resource module and
the pinned owned-runtime snapshot. No extraction script, original archive or trained embedding
weights are needed on the platform. Final reconstruction retains the same owned code/resources.

Regenerate offline with the original supplied archive:

    python scripts/prepare_knowledge.py /path/to/automind-kaggle-knowledge.zip

The record allowlist is tied to the exact input SHA256. Generation does not call APIs or train
models. The script retains full selected write-ups and records all exclusions.

## Verification status

Final implementation, example pipelines, live API integration and platform package were not
run or tested locally at the user's explicit request. tests/test_knowledge.py adds regressions
for task filtering, real source/example delivery, paging, cache isolation and corpus integrity.
Existing gateway, graph, reconstruction and packaging checks remain applicable. No quality
gain or fully verified release is claimed.
