# Semantic tool reads

The historical dialogue repeatedly read method-missing-flags while changing query/tags.
The old cache compared full argument JSON, though ID reads bypass ranking.

One shared normalizer now defines execution and transcript-cache identity. ID reads
canonicalize ignored ranking words, defaults and effective result count. Different IDs
and offsets remain different operations. Query searches normalize case/whitespace and
default fields. Analysis column sets are canonical; report reads fill omitted defaults
but incompatible new-analysis fields remain errors. Unknown fields are not discarded.

Every successful analysis result is also indexed under its report-read operation. Reading
that just-returned report reuses the earlier reply. An all-reused tool turn closes the
tool phase with final/repair capacity retained. Provider tool IDs are still normalized
and every retained tool call has exactly one reply. llm.tool_result_reused logs provenance.

query is now optional for KnowledgeQuery: {ids:[...]} is an intentional read, not a
schema failure. Empty searches still return an explicit error. Unknown analysis report
IDs return the actual available IDs; no report content or results are invented.

No cross-dialogue ban is imposed: a new dialogue may need source content again. Knowledge
service artifact caching uses normalized operations plus corpus version/current pipeline
facts. Synthetic transcript/read tests are written, not executed in this session.
