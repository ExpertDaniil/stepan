# Tool protocol and research memory

This change addresses the observed repeated tool batches, rejected single fenced actions,
empty multi-tag retrieval, and loss of unresolved ideas. It does not establish an ML score gain.

## Execution and history

`agent/llm/protocol.py` strictly parses arguments and canonicalizes each returned tool batch
by function name and parsed JSON arguments. Argument key order and provider call IDs do not
create a new request. Every retained call gets one fresh ID and exactly one tool reply.
Raw provider output remains in its response snapshot; normalized history is what the next
request actually sends. Content accompanying a structured tool batch is not replayed as an
additional unvalidated decision.

Each complete unique call is checked independently. A complete invalid sibling receives an
error result; it does not discard valid siblings. Unknown tools or malformed call envelopes
reject the batch before execution. Equivalent queries expressed with different argument values
are not semantically deduplicated. A batch of genuinely distinct valid calls remains allowed.

Within one completion operation, identical successes and errors are cached. Later repetitions
receive a compact reference to the first result and keep the assistant/tool protocol balanced.
An all-cached round ends that operation's tool phase and requests a final action without tools.
This is a no-progress rule, not a quota on models/features/research directions. Corrected queries
remain possible after an initial error. No extra LLM critic/coordinator is introduced.

A complete schema error does not request more output tokens merely because usage reached the
cap. An actually incomplete single tool argument or final JSON can still increase the output
limit within existing budgets. Repeated incomplete arguments do not increase it. A complete,
schema-valid final action is accepted even if the provider reports `finish_reason=length`.
Missing usage is not estimated; existing budget reservations/accounting remain unchanged.

Final output first passes the existing strict parser. Recovery permits one explicit JSON fence
surrounded by prose, followed by the full expected schema check. Multiple fences, JSON-looking
content outside the fence, duplicate keys, nonfinite values and XML tool commands are rejected.
Tool arguments do not use prose recovery. There is no substring search through nested malformed
objects and no execution of textual tool commands.

## Research decisions

`ResearchAction.research_notes` records the question, mechanism, choice over an affordable
alternative, interpretation of prior measurements and concrete deferred alternatives. It is
optional for compatibility with existing actions; the current prompt requests it. Missing notes
are logged and retain the original hypothesis with an explicit fallback label. Unexpressed ideas
cannot be recovered. Limits on note lengths/eight alternatives bound one record, not the search
budget or the number of accumulated ideas.

`research/decision-*.json` is an immutable native artifact, including for a `stop` decision.
Research threads write through Artifacts and never update SQLite. On the dispatcher,
`agent/research_memory.py` projects visible decision artifacts and trusted Store evaluations into
the next context. It does not add another state machine, DB table or LLM call.

Graph visibility requires a G04 event on the current lineage at its ancestor event cut. Sibling
and later-parent decisions do not enter a restored pre-idea context. Explicit donor measurements
are allowed without importing unrelated donor decisions. Graph-disabled mode projects this
run's graph-disabled decisions. Artifacts written for proposals not admitted through G04 remain
on disk but are not graph observations. Stop artifacts have no fabricated configuration/score.

Measurement association uses the existing fingerprint core: data, task, validation, recipe,
model parameters, feature steps, seed, fidelity and code SHA. Only committed valid evaluations
with no violations produce an authoritative primary metric. A cached exact configuration can
supply its earlier observation. A local tuning continuation remains in measured_search; it is
not falsely attributed to an idea through approximate parameter/code matching.

Idea IDs come from normalized question/mechanism text or a visible existing ID. First-seen
question/mechanism identities persist; references and deferred-test notes accumulate. This is
exact identity preservation, not semantic novelty detection. Every idea remains unresolved in
the controller. The model can prioritize/defer it; free-form claims never close it or install a
search ban. Open does not mean all ideas deserve a trial. Individual component effects are not
inferred from a joint pipeline outcome. The correctness of an interpretation is not guaranteed.

## Knowledge

Tags now contribute query terms rather than requiring an intersection of all tags. Task-kind
eligibility stays strict. Results explain matched/unmatched tags and query terms; unknown tags
are reported without blocking useful text matches. Relevance remains a lexical heuristic.
Initial retrieval also uses the latest available deferred question, otherwise the most recent
decision question, alongside the fixed profile. That is a retrieval hint, not evidence of merit.
Previously read knowledge IDs may be carried in memory; their full content is not implied to be
present, so implementation-dependent reuse still calls for reading the source.

## Verification limits

Regression tests were added for batch deduplication, valid siblings with a bad tail, repeat-only
rounds, real truncation, conservative JSON recovery, soft tags, exact configuration observations,
deferred-idea retention, stop decisions, reference provenance and graph/legacy visibility.
Tests, training, live LLM calls and submission packaging were not run here at the user's request.
The next real run must establish integration behavior and ML quality; neither is claimed here.
