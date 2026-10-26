# Regression run 7061bdc8: blocked Qwen candidates and moving feedback

Evidence: the user-supplied `events.jsonl` excerpt of local run
`run-20261026T002129Z-7061bdc8` (budget 10800 s, live Qwen), 9808 events from
00:21:29 to 00:51:54 UTC (about 30 minutes). Model request/response bodies
(`llm/*.request.json`) were not supplied, so tool arguments and model text are
known only through logged events. The owned runtime in the 1-hour platform ZIP
(`tools_runtime_guard.py`, `agent_strong.py`) is byte-identical to the branch
at 83c7d46. The platform leaderboard value 268.281 comes from a different
execution without logs; nothing below explains or attributes it.

## Outcome in the excerpt

The published champion improved from Dummy 57.49 through the strong warmup
38.57 to 35.765 (00:42:40), entirely from warmups and background Optuna.
Researchers finished six dialogues. One more dialogue timed out without a
decision. Four candidates reached execution and all four failed after
1.5-2.4 s with `Generated pipeline blocked audit event ctypes.dlopen`:
`exp-8e24fb2e0893`, `exp-91afcf8cbb82`, `exp-bbe0eff5d5db` and
`exp-fa06ee0a64d7`. Qwen therefore proposed executable experiments, but its
measurable contribution was zero because execution blocked every candidate.

## Cause 1 (confirmed and reproduced): guard preparation missed owned backends

`prepare_pipeline_imports` preloaded a backend only for the top-level
`base_recipe_id` prefixes catboost/linear/sklearn/lightgbm and had no `strong-`
entry. `StrongTabularModel` imports LightGBM lazily inside `fit`
(`agent/strong.py:219`). LightGBM's `libpath.py` calls `LoadLibrary`, which
raises the `ctypes.dlopen` audit event after the guard is sealed. Custom code
that builds owned recipes through `build_pipeline("strong-...")` had the same
gap. Reproduced on synthetic data at HEAD, for both a feature program over
strong and a strong+CatBoost blend.

Fix: one recipe-prefix → backend table, applied recursively to the plan
hyperparameters and to recipe-ID string literals in the generated source.
Owned imports of `agent.strong` also map to LightGBM. Arbitrary ctypes loads
remain blocked (`test_native_guard`).

## Cause 2: an environment refusal looked like a failed hypothesis

The worker reported the refusal as generic `WORKER_FAILED`, and memory marked
such ideas `failed`. In the excerpt, researcher-2's third dialogue spent its
turns reading `execution/*` and `experiment/*` of the failed candidates and
drafting repairs for a defect in our runtime.

Fix:
- The guard now raises `GuardViolation` and records the nearest non-library
  caller. The runner reports `GUARD_ENVIRONMENT_BLOCKED` when the caller is an
  owned wrapper, and `GUARD_POLICY_BLOCKED` when it is the generated code.
- `failure_scope()` classifies stored errors by code, not by prose.
- Feedback says the candidate was not measured, and memory marks such ideas
  `environment_blocked`.
- An explicit resubmission requeues the same experiment after the existing
  quarantine backoff. `Store.requeue` is ported from 1e5196f and also covers
  timed-out and cancelled attempts.
- Candidate-code refusals stay terminal.

## Hypothesis: background updates distract researchers

The excerpt partly confirms this hypothesis, partly refutes it, and refines it.

- **Refuted: accumulation.** `research_updates` is appended to each request
  copy only, never to the stored transcript, so every request carries exactly
  one, the latest.
- **Confirmed: the update changes on almost every turn and steers reads.** Each
  dialogue received 8-9 distinct revisions in 9 calls, because every Optuna
  trial changes the four-row `results` window. The best solution changed 7
  times in 20 minutes. The instruction in every update said
  `Use read_solution(experiment_id="best")`. Of the 35 `read_solution` calls,
  32 targeted the current or a former best, and 18 were `result` views. A
  `result` view returns only state and a metric, and both were already in the
  update. Researcher-1's first dialogue spent four of its eight tool turns on
  such reads (calls 5, 6, 7 and 13). Researcher-2 never read its own graph
  base and followed the moving best (`74ba` → `0e0b` → `d7ba`).
- **Termination.** Every model decision came at the forced final turn
  (attempt 9, `final_decision_requested`). No dialogue ended earlier on its
  own. Dialogue wall time was almost entirely provider latency: 271 of 280 s
  and 425 of 433 s were spent inside calls.
- **Not explained by tool errors.** There were no invalid tool arguments and no
  failed analysis events. One weaker inconvenience: the tool guide sits behind
  `read_memory`, and dialogues spent 9 reads on it.
- **Two different bases.** The graph head of researcher-2 stayed the CatBoost
  warmup (46.84), while its first proposal was "blend the leader with the
  incumbent CatBoost". The research base in the error-research assignment is
  frozen at dispatch but was not restated in updates. Without the request
  bodies, the base text the model actually saw remains unverified.

Causality is not proven by these counts. They show the turns went to
re-orientation on a moving incumbent rather than to analysis.

Fix (existing work order, context and feedback; no quotas):
- `frame_updates` adds a fixed `research_frame`: the dialogue question, the
  research base and its metric, plus the current best as `utility_reference`.
- A turn whose updates change nothing relevant re-delivers the previous view
  with `changed_since_previous_turn=false`. Relevant means the best,
  controller feedback, pending work, diagnostics or series. Background trials
  remain readable through `details`. A new best is listed in `changes` and
  can still redirect the research.
- `read_solution` arguments get their defaults. `best` resolves to the
  delivered incumbent, so equal reads reuse the earlier result and a new best
  is read afresh.
- The updates instruction no longer asks for a best read on every turn.
- Paired comparison reports `versus_current_leader` next to the existing
  contrast with the research base.

## Memory (not a defect)

`omitted_fields: ["bounded_research_memory"]` is `compact_memory(...,
max_bytes=8000)` replacing the full projection, and details stay readable
through `read_memory`. Retained decisions grew from 0 to 4 across the
excerpt, and researchers read `decisions`, `ideas` and `experiments`. No
second memory mechanism was added.

## Validation

The rolling blocks 1-3 → 4 and 1-4 → 5 scored every validation row, with
`confirmation_rows=0`. Hundreds of trials selected on the same rows make the
winner optimistic. The existing confirmation now withholds a seeded 25% of
every scored period through the shared `partition()`. Both periods keep
selection rows, and the saved fold predictions are reused, so there is no
extra fit. Research scores, diagnostics and paired comparisons read selection
rows only. The leaderboard difference is not attributed to this change.

## Needs the next live run

- Whether framed updates lower result/best re-reads and bring decisions before
  the forced final turn. Compare `research.feedback_delivered.changed_since_previous_turn`,
  `research.solution_read` counts and `final_decision_requested` attempts.
- The first measured Qwen candidates, judged by `research.paired_comparison`
  (status vs base, `improvement_vs_leader`).
- `validation.finalists_confirmed` on the withheld rows.
- Provider latency per turn, which bounds the whole dialogue.
