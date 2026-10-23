# Feature research: implementation and verification

Started from the actual remote feat/open-research-loop head
`b416b5b7253ea0d98a006e753cfffec9b6fac319`, with a clean matching checkout.
The work is a normal commit series on that branch; main is outside the scope.

## Commit boundaries

1. Composable inductive feature programs, adaptive features action and measured representation archive.
2. Rotating feature-focused assignments and a bounded controller seed bank.
3. Tool-phase wall budget, final/repair capacity and idle/decision observability.
4. Active peer/search publication and admission-time overlap reconsideration.
5. Semantic read identity, knowledge schema defaults and analysis report reuse.
6. Controller-computed evidence scopes; unverified model prose never becomes a fact.
7. Integration checks: frozen feature dependencies, preloaded owned backend, stable numeric
   category keys, bounded memory, custom model tuning overrides and reproducible static audit.

Feature programs and custom Python may coordinate multiple changes; no compulsory ablation
sequence or new LLM critic/agent role was added. Controller feature seeds are explicitly
labelled hypotheses and share the existing Runner/Evaluator/resource/final-output path.

## Performed

- AST parsing of 113 Python files, including all 62 owned runtime modules.
- Local schema vocabulary/reference checks and data-only AST equality of ResearchAction,
  AnalysisQuery and KnowledgeQuery declarations with contracts/contracts.schema.json.
- Readability/existence of 19 allowlisted text resources, flat module name collision checks,
  package-input policy checks and hashes of the existing knowledge corpus.
- Exact requirements byte equality with requirements.organizers.txt and the supplied
  official participant ZIP. SHA256: 9173b0388532fbd2a709e4f4b3029e3b9ccab685fce43a85261ffa1ba95efbe6.
- Feature/runtime guard dependencies included in the immutable execution snapshot.
- git diff --check and final tracked-file manifest verification.

Machine-readable evidence: validation-feature-static.json. There are 26 newly written
regression methods (314 total methods in the repository); this is a source count,
not a test execution result.

## Not performed

The continuing user restriction forbids local model fitting, agent smoke/full runs,
test suites, paid/live provider calls and dependency installation. None were performed.
No submission ZIP was built or executed. Static package-input checks do not prove the
eventual ZIP/platform run. The new synthetic tests are written but unexecuted.

Lower idle time, successful backend/proxy integration and better ML scores remain
unverified. Pending activity is not a measurement; domain overlap is an explicit heuristic.
An arbitrary custom feature declaration is not semantic proof of its implementation.
Free-form model interpretations remain unverified; no general sentence-level truth
classifier is claimed. The SDK must honor API timeouts to bound a synchronous call.

## User-side checks

Run from the updated repository with its existing approved environment:

```bash
python3 -m unittest tests.test_feature_programs tests.test_feature_search tests.test_feature_admission tests.test_research_latency tests.test_research_coordination tests.test_tool_semantics tests.test_evidence_scopes -v
python3 scripts/check.py
python3 scripts/package_solution.py
```

Then start a fresh run; changing files does not upgrade an already running process:

```bash
RESEARCH_MODE=adaptive FEATURE_RESEARCH_ENABLED=1 FEATURE_SEED_BUDGET_FRACTION=0.15 FEATURE_SEED_TIMEOUT_SECONDS=120 LLM_RESEARCH_TOOL_SECONDS=90 LLM_RESEARCH_DIALOGUE_SECONDS=360 LLM_MAX_CALLS=0 LLM_MAX_TOTAL_OUTPUT_TOKENS=0 GRAPH_REFINEMENT_ENABLED=0 python3 solution.py "data_for_participants/Регрессия/task.md" --env-file .env --profile platform
```

Inspect its saved directory using `python3 scripts/analyze_run.py runs/run-... --json`.
Compare completed decisions, no-train/refit wall seconds, feature programs actually fitted,
overlap deferrals, semantic reuse and scores on the same protocol. A lower local selection
score is not an independent confirmation or a guaranteed private leaderboard gain.
