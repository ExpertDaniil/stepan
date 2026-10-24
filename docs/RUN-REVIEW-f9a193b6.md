# Classification run: failed research dialogues

Evidence: user-supplied `run-20261024T091305Z-f9a193b6.zip`.
All 89 files of its owned runtime match the source/resources at remote
`05411d939f681f643833fbddc461f8a0e707a89b` (the generated requirements lock
matches the official requirements). This is a local live-provider run, not
evidence establishing the cause of the earlier platform score of zero.

## Observed outcome

The run ended `degraded`, `research_unavailable`, with a valid ExtraTrees output.
Five experiments were evaluated: one Dummy and four controller warmups. No
LLM-proposed pipeline, feature program or tuning trial was admitted or trained.

| Configuration | Selection Macro F1 |
|---|---:|
| Dummy | 0.048415 |
| CatBoost | 0.548497 |
| HistGradientBoosting | 0.623623 |
| ExtraTrees | 0.623573 |
| Linear | 0.430135 |

The final one-time confirmation compared HistGradientBoosting (0.660254) with
ExtraTrees (0.663934), selecting the latter. Selection and confirmation values
are from different partitions; their difference is not a ranking bug or a
platform/private-task quality estimate.

## Confirmed failure chain

1. Initial researcher JSON bodies already occupied 88,814 and 89,850 bytes of
   the 98,304-byte graph limit, including output schema, tools and live feedback.
2. The first peer received four tool errors, then corrected the requests and
   computed four fit-only reports. After this second batch the body occupied
   128,352 bytes. The old compactor reduced it only to 114,799 bytes and raised
   `CONTEXT_TOO_LARGE` before another provider request.
3. The second peer requested a class summary and knowledge search. Its body
   occupied 101,855 bytes. The old compactor left 98,437 bytes, just 133 above
   the hard limit, and raised the same error.
4. The scheduler immediately counts `CONTEXT_TOO_LARGE` twice for each peer,
   deferring it until new measured feedback. There were no active experiments
   to produce such feedback. It finalized with 9,438 wall seconds remaining
   and 994,933 output tokens still available. This was not a provider quota
   or global time exhaustion.

The compactor handled only older `report` replies, skipped the last tool reply,
and did not project `search_knowledge` or `read_memory` replies. The complete
request lacked a sufficient tool-transcript packing path.

An additional producer/consumer mismatch caused two initial analysis errors:
fit-diagnostic artifact IDs were displayed in `scientific_protocol`, and accepted
by the plan validator, but not accepted by the analysis investigation validator.
Two other analysis calls incorrectly supplied `limit` for `group`; the model
corrected those after the returned error. Bootstrap needed three calls: truncated
output, an invalid source quotation, then a valid draft. Neither issue directly
caused termination in this run.

## Changes

- Aim to leave 16 KiB of headroom on the first tool turn, within the existing
  hard request limit. Later turns use the full limit to preserve new evidence.
- Remove redundant whitespace in nested JSON and project retrievable catalogue,
  family and knowledge details before discarding observations. Complete catalogue
  and family records are stored in the existing frozen detail snapshot.
- Compact all supported retrievable tool replies, including the latest one,
  without removing assistant/tool-call pairs. Keep scopes, supports, whole
  observations, source IDs and explicit omitted counts. Original service artifacts
  and tool cache values stay unchanged. Python examples are omitted whole, with
  instructions to retrieve their complete source. Shortened memory pages keep
  correct UTF-8 offsets and do not claim a complete record read.
- Preserve task contract/text, base code/result, semantic task and input names.
  A truly irreducible request still fails the existing hard byte check.
- Accept issued fit-diagnostic IDs consistently in analysis investigations and
  include newly delivered diagnostic IDs in feedback evidence. Arbitrary or
  out-of-scope artifact reads remain prohibited.
- Clarify in the tool description that group queries do not accept pagination.

## Verification limits

Nine synthetic regressions were added, covering full request pressure, a parallel
tool batch, the latest knowledge reply, scope/support preservation, immutable
source content, UTF-8 pagination, irreducible contracts and diagnostic citations.
They were **not run**. No training, agent/smoke run, dependency installation or
live LLM call was performed: the original user prohibition remains active.

Static AST/schema/resource checks pass: 132 Python files, 69 runtime modules,
19 allowlisted resources, 170 resolved schema references, 86 existing contract
examples and byte-identical official requirements. This does not establish that
the repaired live dialogue completes or improves ML quality.

After explicitly lifting the test prohibition, run:

```bash
python3 -m unittest tests.test_context_pressure tests.test_semantic_research tests.test_scientific_research -v
python3 scripts/check.py
```

A subsequent authorized live classification run must show `decision_received=true`,
admitted LLM experiments and independently evaluated scores. Do not spend a platform
submission on this change solely because static or mocked checks pass.
