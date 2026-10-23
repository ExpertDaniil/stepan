# Research latency controls

Observed in b99d0f8f: three no-job gaps totalled 708.663 seconds, about 47% of the
saved 1503.626-second fragment. There were 34 research API calls for five decisions.
Output was not truncated. Reasoning-token share does not prove reasoning quality.

Adaptive defaults now allow 90 seconds for starting further tool requests and
360 seconds for the entire dialogue, also bounded by operation/search/final reserves.
An in-flight tool request may finish past the soft tool boundary; the next request is
final-only. Final/repair call and token reservations are retained. Output size and
provider reasoning protocol are unchanged. Independent read tools may be batched in
one provider response; the client still executes them deterministically.

The small feature seed bank keeps a free peer doing measured representation work while
another waits. Exhausted banks/authorizations do not imply endless artificial fits.
Synchronous provider cancellation still depends on the SDK honoring its timeout;
the controller rejects late replies but cannot hard-kill a hung Python API thread.

`research.dialogue_summary`, `llm.tool_phase_closed`, `features.seed_proposed` and
`scripts/analyze_run.py` report actual dialogue time and merged train/refit intervals.
No-job seconds are not CPU utilization. Unfinished archives are labelled incomplete.
The reduction of the historical 47% idle share is unverified until a new run.

Synthetic latency/occupancy regressions were written, not executed. Static audit only.
