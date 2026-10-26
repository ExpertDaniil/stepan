# Isolated interface repair in last_test

Base: verified source tree of remote `92661a6`, without modifying slice-research.

## Scope and boundaries

The optional engineer is the same configured provider/model in a separate, short
context. It replaces one existing final-action repair attempt, not every action
and not another independent research agent. A successful action adds no API call.
It currently repairs final executable ResearchAction interfaces only. Analysis
code/tools, arbitrary Python debugging, remedy management and semantic experiment
deduplication retain their existing behavior. This is deliberately narrow.

An immutable packet contains the complete original action, the failure, the
output contract, exact allowed field edits and the issued base experiment ID.
The body is never truncated. Packets above 32 KiB use ordinary repair instead.
No task tables, diagnostic rows, confirmation labels or full conversation are
added. No tools or fresh moving-best mailbox are offered to the engineer.

Permitted edits: fill absent/empty service fields with their documented defaults;
for a feature action with an absent/placeholder/custom recipe name, recover the
owned recipe from the issued base snapshot. Other recipe changes are forbidden.
Parameters, feature program, source/code_ref, remedies, action, evidence and all
other supplied fields must remain exactly unchanged. Comparison preserves JSON
value types and array order. Nonempty code or tuning cannot be silently removed.
Missing data, remedy caps, tuning-domain errors and ambiguous bases remain
researcher decisions. A repaired candidate still passes the same schema,
normalization, scoped draft resolution and Researcher validator as normal input.
No arbitrary code equivalence or general scientific intent certification is claimed.

## Budget, concurrency and fallback

The existing attempt counter, dialogue call/token reservations, provider semaphore,
cancellation and global/phase finalization deadlines are reused. There is no
recursive Gateway call, new thread, extra provider slot or unaccounted request.
At most one engineer turn per research dialogue is scheduled. Its request timeout
is at most 90 seconds by default and 35% of the remaining dialogue window, leaving
time for ordinary fallback when capacity remains. Tools remain unavailable.
The original output token allowance is retained; no tiny reasoning cap is imposed.

Refusal, invalid output, protected edits, unavailable provider or too-large packet
leave the old repair path available within the existing budget. If no capacity or
time remains, existing checkpoint/finalization behavior applies. This is graceful
fallback, not a guarantee of a further funded API call. Account exhaustion and
cancellation are not hidden. The verified champion/submission flow is unchanged.

## Configuration

`LLM_REPAIR_ENGINEER_ENABLED=0` restores ordinary research repair.
`Config.from_env` enables the feature for the platform profile and disables it
for development unless overridden; direct Config fixtures default to disabled.
`LLM_REPAIR_ENGINEER_TIMEOUT_SECONDS` defaults to 90 (valid range 1..300).
The flat submission uses platform defaults, with normal environment precedence.

## Initial verification

Nine contract/Gateway tests passed, including real Researcher acceptance of a
repaired feature action in exactly two total reserved API attempts, explicit
refusal, forbidden parameter mutation, transport failure fallback, disabled mode,
and no extra call for successful actions. The existing 38 gateway, recovery and
dialogue tests passed. Replies were scripted; no live provider behavior or public
metric gain has been established. Final release checks are recorded separately.
