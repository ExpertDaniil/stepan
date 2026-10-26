# Diagnostic audit rows and initial request pressure

The live regression run exposed 38 initial packing failures: no research request
reached the provider after task interpretation. The new diagnostic isolation
audit included `discovery_fit_row_ids` and `check_row_ids`. `compact_report`
copied both arrays into scientific context and per-turn feedback. Existing
compaction could not remove these repeated arrays, despite their not being
scientific observations. The controller then saved its own failure prose as the
next research question, producing repeated retries and growing continuation text.

The display projection now replaces these arrays with exact row counts. The
complete report, its hash, exact isolation audit, slice rules, measurements and
executable remedy references remain in the original artifacts. The same scoped
reader still exposes the full report. No byte/token limit or validation boundary
was relaxed, and model fitting/publication behavior is unchanged.

An initial packing failure now propagates to dispatcher recovery rather than
creating a model-like continuation. That failed input waits for a change in the
measured research generation; its own packing-error feedback cannot wake it.
Other peers and background training continue. A later overflow after tool work
still archives that work and returns a waiting checkpoint. Controller checkpoints
retain the scientific question without appending recursive failure instructions.

`tests/test_live_context_regression.py` exercises both task kinds through the
real ContextBuilder, Gateway, C09/C10 contracts and Researcher validator. Two
diagnostic copies and two mailbox copies each contain 10,000 audit IDs. The
full update exceeds 220,000 bytes, its projection is below 7,000 bytes, the
first complete request fits below 180,224 bytes, and a feature proposal reaches
the transport in one attempt. Full audit rows remain in scoped details. Further
tests cover irreducible initial input, peer independence, evidence-based recovery,
idempotent projection and 20 repeated post-tool checkpoints without question growth.

Provider replies in these checks are scripted. This verifies request delivery
and executable proposal acceptance, not live Qwen scientific quality or LB gain.
