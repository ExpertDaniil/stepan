# Repository guide

- Read README.md, docs/IMPLEMENTATION.md and docs/LOGGING.md before changing cross-module behavior.
- Python 3.12; runtime dependencies must remain within requirements.organizers.txt. Do not install packages during a task run.
- solution.py remains the platform entry point. scripts/package_solution.py owns the allowlisted platform ZIP.
- The platform ZIP must be flat: only root .py files and the exact official requirements.txt survive copying. Export every owned Python module as a separate readable, byte-identical root .py file. Keep text resources readable in stepan_resources.py. Do not encode source code or embed a runtime archive. Runtime package directories under /tmp are allowed. Never upload the source repository archive as a submission.
- requirements.txt must match requirements.organizers.txt exactly. requirements.runtime.txt is the approved local-development subset. Never install dependencies during a task run.
- One invocation processes one task.md. Prefer adjacent data; do not recursively borrow optional metadata from unrelated task directories. Task discovery must not read CSV contents or call APIs.
- The user requested direct commits to main for this project. Preserve remote changes and use normal fast-forward updates; never force-push.
- contracts/contracts.schema.json and its C00–C24 examples define the message boundaries. Changes must update producer, consumer and relevant fixtures together.
- Keep task names, target columns, metrics and paths dynamic. Use synthetic tasks for tests; do not copy real datasets, keys or run artifacts into source control.
- SQLite mutations belong to the dispatcher thread. Research threads write artifacts through Artifacts and return validated plans.
- Evaluation computes scores from predictions. Researchers do not supply authoritative metrics.
- Preserve the last verified CSV until a complete better candidate has passed output validation. Stop process groups before releasing resources.
- Log observable model outputs and short explicit hypotheses; never invent unavailable provider reasoning. Keep secrets redacted.
- Use focused tests for local changes and scripts/check.py for a release or behavior spanning modules. Record actual verification limits honestly.

- Default research is catalogue-only ResearchChoice-v1. Keep code mode explicit; never silently execute LLM code as a catalogue choice.
- Per-request and per-operation LLM timeouts are distinct. Treat truncated output separately, reserve concurrent output tokens, and never infer RUB or missing usage from guesses.
- Keep estimator parameter validation and train-only target transforms covered by meaningful synthetic checks. Do not promise a particular benchmark score from infrastructure tests.
