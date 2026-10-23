> Актуальный контракт и статус проверки: [MASTER-UPDATE.md](MASTER-UPDATE.md).
> Описания прежних defaults/отключённого byte cap ниже относятся к истории изменений.

# Research dialogue completion and training progress

The observed run spent its exploration API allowance on tool turns and malformed JSON,
including a dialogue whose final answer could not be requested. Its first completed Optuna
trial also waited for another peer's API dialogue before the next trial was scheduled.
These changes address those execution paths. They do not establish a benchmark gain.

## Budget ownership

Before a ResearchAction-v1 dialogue starts, Budget atomically protects capacity for its final
request and one repair when both fit. Capacity includes actual API-attempt allowance and the
maximum output tokens per future request. With only one funded slot, the dialogue goes straight
to a final decision without tools. With no funded final slot, it does not start a tool dialogue.
Unused capacity is released in Gateway.complete's finally block, including on provider errors.
Worst-case output reservations can reduce concurrent dialogue capacity near a token cap even
when typical replies are shorter. They reserve capacity, not money or reported usage.

An optional tool request must acquire separate unreserved capacity. If it cannot, Gateway
removes tools and uses the protected final slot. Other concurrent dialogues and unprotected
requests cannot consume that slot. Attempt count and global/phase token and API limits remain
unchanged; with the ordinary eight-attempt dialogue and two protected slots, at most six turns
can offer tools, followed by a final decision and an optional repair. Early valid answers return
immediately. Reserving capacity does not increment the actual API-attempt count.

The controller also uses observed response durations to decide when to stop offering tools.
This time estimate is a heuristic. Protected capacity guarantees admission against competing
requests under unchanged caps and valid provider usage accounting. It cannot guarantee a valid
model answer, provider availability, exact billing, or a response before the operation deadline.
Retries can consume the protected repair opportunity. Enlarging output after truncation must
fit the remaining token budget without borrowing another dialogue's reservation.

After non-truncated invalid adaptive JSON, the complete visible answer and parser/schema error are saved in
an immutable llm/rejected-*.json artifact. The answer is retained in the same dialogue for an
addressed correction; no new tool phase starts. Provider reasoning_content is never replayed.
Every corrected answer still passes strict JSON parsing and the entire output schema. There is
no automatic acceptance of guessed brackets, missing required fields or unsupported keys.

The 32-call default is still a hard cap on physical API attempts, not 32 research decisions.
The exploration/refinement allocation is unchanged. These changes prevent another dialogue
from stealing protected completion capacity; they do not create additional research budget or
unlock refinement's reserved calls early. A long run may still exhaust reasoning capacity and
use the existing owned catalogue fallback.

## Dispatcher progress

For adaptive graph batches with one training worker, waiting for API futures uses a short poll.
When no reply is ready, the dispatcher can continue an active explicitly authorized Optuna
series for a peer without a pending dialogue, on an active branch in the current batch.
It obtains a fresh work order, retains the plan/ticket/owner binding, applies ordinary resource
and time admission, trains through Runner, evaluates independently, and tells Optuna from Store.
There is no new LLM call, implicit tuning authorization, new branch choice or experiment quota.

SQLite writes stay on the dispatcher. The CPU slot is not oversubscribed. A just-completed API
reply may wait for a training/refit call already in progress; this is not a fully asynchronous
Runner rewrite. Catalogue/code and multi-training-worker scheduling retain their previous path.
Finalization reserves, last verified CSV protection and process cancellation remain unchanged.

## Evidence to check

Synthetic regressions cover competing dialogue reservations, actual concurrent final/repair
requests, one-slot admission, phase/output caps, full malformed JSON correction, reservation
cleanup after a provider error, and three authorized trials completing before a slow peer's
dialogue returns. The dispatcher fixture injects evaluations and never trains an estimator.
The existing tool regression now locates its report by role because a protected final turn can
append an instruction after the last tool reply.

Tests, training, live API calls and packaging were not run here at the user's request. For the
next run inspect dialogue_reserved/released, final_decision_requested, json_repair_prepared,
call_started/output_validated, and tuning_while_reasoning. Check that the model actually receives
completed study outcomes and makes another validated decision; a populated unused context alone
does not demonstrate that research feedback influenced behavior.
