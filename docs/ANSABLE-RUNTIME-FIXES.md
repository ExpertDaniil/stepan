# Runtime follow-up: source-reviewed, execution pending

This change follows the supplied classification log `run-20261025T160339Z-bf07e833`.
It preserves `ansable_test`, the existing graph/store, Budget leases, scoped
memory, Runner, Evaluator and atomic publication. No tests, ML jobs, paid calls
or platform submissions were run: the user's explicit instruction remains in force.

## Evidence before these changes

| Control / experiment | Local selection Macro F1 | Observed time / count | Conclusion |
|---|---:|---|---|
| Startup ExtraTrees | 0.63340277 | Published 16:10:38 UTC | Strongest observed score in this log |
| Completed background trials | Best 0.62317685 | 23 completed trials | No improvement over the startup leader in this fragment |
| Startup CPU neural model | 0.46307984 | Approximately five minutes | A standalone result; marginal ensemble value remains unknown |
| Queued three-member ensemble | Unavailable | Failed after 5.7121 s | Worker IO guard rejected native access |
| Research | No evaluated representation proposal observed | 32 API call starts; one saved draft | Analysis occurred, but conversion to a measured feature experiment is unconfirmed |
| Whole fragment | No official score | 24 min 34 s | Ended with KeyboardInterrupt, not the three-hour deadline |
| Current fixes | Unmeasured | Source/syntax review only | No claim of ML improvement, working ensemble or deadline verification |

Counts refer to the supplied file's 3,634 parsed event records,
not an independently reconstructed run directory. These are internal selection
measurements, not confirmation, outer audit, leaderboard or official test scores.

## Changes and their limits

1. `tools/runtime_guard.py` prepares constructor imports recursively, resolves
   the installed LightGBM module's declared C symbols, and initializes sklearn's
   separate threadpool controller before sealing IO. ctypes buffer allocation
   and address lookup remain permitted; dlopen/dlsym, processes, network and
   artifact IO remain blocked. Rejections now identify the precise audit event.
   The generic old error alone does not establish the exact first rejected event
   in that historical worker; its full stderr was not supplied. The blanket ctypes
   prohibition is directly incompatible with the installed LightGBM implementation.

2. Specialists edit `make_features` source. `read_solution(view=representation)`
   returns the exact editable transform, while view=code continues to return the
   complete predictor factory. In specialist contexts, experiment_id=best selects
   the owned architecture. A native recipe has an explicit identity representation;
   an owned feature program is exposed through its actual FeatureProgram factory.

3. `edit_solution` accepts optional `submit=true`, a short hypothesis and optional
   evidence/semantic updates. Successful explicit submission passes normal source,
   action and proposal validation, then finishes the dialogue. Researcher builds
   the plan; the dispatcher alone submits it through G04/C12 and Runner. Saving
   a draft alone never starts training. Invalid source is retained for repair.
   Remaining operations in a batch after an accepted submission are not executed.
   Final JSON remains supported. No new experiment queue or researcher-side SQL.

4. Specialist checkpoint time derives from measured provider latency within the
   configured dialogue maximum. It is renewable, not a lifetime direction cap.
   Exact source, analyses, feature interface and configurations are frozen before
   projecting duplicate queries, numeric model domains and oversized fact details.
   Source IDs, epistemic labels, relevant semantic memory and contradictions retain
   their existing representation. This is retrievable detail, not a guarantee that
   the model has every fact in active context or will reason usefully.

5. The five startup learners enter VersionSearch instead of forming a synchronous
   barrier before every dialogue. Background workers and research share the same
   resource accounting. A missing measured specialist reference is explicitly
   labelled unmeasured and uses the small startup parameters. Startup completion
   precedes automatic tuning and any configured mandatory data window; feature
   proposals and research can proceed while startup fits run.

6. Ensemble review retains the strongest and cheapest currently measured version
   of each architecture and considers affordable distinct-architecture committees.
   It prefers the largest affordable committee, then mean standalone quality and
   estimated cost. This is an explicit heuristic, not evidence of complementarity.
   Deferred committees can be reviewed again after resources/costs change. Actual
   completed ensemble durations enter subsequent estimates; the cost reserve includes
   inner fits during full refit. Weights still use fit-only OOF full pipelines;
   selection residual reports do not manufacture a score. No confirmation labels
   enter research. The best ready standalone publication remains available.

7. Admission respects each job's actual timeout, not only the global experiment
   timeout. Cooling/unaffordable initials cannot block all ready representations.
   Transient stopped attempts reuse candidate identity and immutable plans with
   increasing attempt numbers and fresh lease keys. Old events remain fenced;
   deterministic invalid code is paused until changed input/source. No running
   process is reopened or its lease released by this retry path.

8. Ctrl-C/SIGTERM logs explicit interruption, keeps a hash-verified publication or
   emergency CSV, and exits 130. It does not label manual cancellation as a deadline
   event or report a new metric. A verified champion receives a run summary.

## Semantic-to-experiment path

A specialist reads a task/dictionary definition and a fit-only error report,
records a proposed rate/exposure relationship as a hypothesis with those sources,
and implements a ratio/missingness treatment in `make_features`. It explicitly
submits that draft. The executor supplies the assigned measured estimator, pins
code/task/protocol identities and sends the version through ordinary fit/evaluation.
The measured experiment is bound to the saved decision and appears in shared
feedback/semantic research memory. A negative split result does not turn the
hypothesis into a universal prohibition. This describes the implemented path;
no successful live run of that complete path is claimed by this patch.

## Local reproduction

With the organizer data extracted under the repository as in the supplied run:

```bash
git pull --ff-only origin ansable_test
ENSEMBLE_RESEARCH_ENABLED=1 GRAPH_PEERS=5 RESEARCH_MODE=adaptive BACKGROUND_TUNING_ENABLED=1 python solution.py "data_for_participants/Многоклассовая классификация/task.md" --env-file .env
ENSEMBLE_RESEARCH_ENABLED=1 GRAPH_PEERS=5 RESEARCH_MODE=adaptive BACKGROUND_TUNING_ENABLED=1 python solution.py "data_for_participants/Регрессия/task.md" --env-file .env
```

Run one task per invocation. No new environment switches or credentials are
required. Observe `search.catalog_warmup_queued`, `research.executable_submitted`,
`research.action_proposed`, `pipeline.execution_started`, `experiment.evaluated`,
`experiment.requeued`, `submission.published` and `run.interrupted` when relevant.
An executable_submitted event proves accepted source, not successful model fitting.

The packager already exports every owned Python module and the specialist prompt.
No ZIP was built or execution-verified for this checkpoint. Full fit/refit,
serialization, API failure recovery, ensemble output, both task runs, deadline
behavior, equal-budget policy comparisons and unknown-task transfer remain to be
measured. In particular, no quality regression/improvement claim is supported by
the static checks performed here.
