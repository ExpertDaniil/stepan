# Tools v1

Runtime generates a complete tool catalogue, including the current JSON input schemas,
under `runs/<run_id>/task/TOOLS.md` from `contracts/contracts.schema.json`.

| Tool | Contract | Responsibility |
|---|---|---|
| read_task | C01 | UTF-8 Markdown and unambiguous task-local file resolution |
| profile_data | C03 | Schema and aggregate missing-value report; no raw table in LLM messages |
| write_code | C11 | Syntax/policy check and immutable code artifact |

The LLM research gateway exposes write_code through native tool calls. Reading and profiling
are performed by the bootstrap before research; their results enter the context.
Runner starts a fresh process for every training/refit job. File artifacts persist, Python variables do not.
Only Publisher writes the final submission path. Code checks are guardrails, not an OS sandbox.
