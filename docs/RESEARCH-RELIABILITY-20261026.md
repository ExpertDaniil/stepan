# Executable research and closed temporal confirmation

The platform entry point and dependency allowlist stay unchanged. The packaged
profile supplies 10800 seconds per invocation, with live Qwen and independent
background tuning. No public dataset/column/label rules are added to the agent.

## Analysis and decision repair

Python analyses use the same validated import/backend preparation as generated
pipelines. An owned lazy backend may load before the execution guard closes;
arbitrary IO and new native loaders remain blocked. The trusted RSS watchdog
stays active under the guard. Compact aggregates retain JSON-safe NumPy scalar
handling; pandas missing values now also use JSON null instead of text markers.

Failed code, startup, memory and timeout executions retain their source and
report, including a typed failure, short message, traceback and repair hint.
Cancellation still stops publication-time work. Terminal events include the
short failure; there is no need to infer a cause from success=false alone.
The tool contract documents the actual Series/DataFrame shapes, optional target
and test inputs, OOF fields and residual sign.

For a non-group feature operator, group_by=null is canonicalized to omission.
An actual unsupported group column is rejected with its operation index and
repair instructions. Target-statistic keys belong in columns; the controller
does not guess them. Failed final decisions are archived and their continuation
retains the validation error, so the next dialogue can repair the saved proposal.

## Stable research feedback

The question and research base remain fixed within a dialogue. Background trial
IDs/states cannot alone refresh the view. A new leader, submitted candidate
outcome, pending research candidate, diagnostic or authorized series change can.
Candidate outcomes have a separate bounded window and scoped solution records,
so an unsuccessful representation does not disappear behind faster Optuna trials.
The exact branch mailbox remains readable. No model family or count quota is
introduced, and a new leader still updates the utility reference.

## Temporal validation

Randomly withholding rows from each rolling validation period is insufficient:
held labels of an earlier period enter the fit for a later selection block.
The existing temporal partition instead keeps the latest whole periods closed.
An invariant rejects any intersection of confirmation rows with any rolling fit.
Every candidate still predicts the original full validation partition; adaptive
metrics, diagnostics and paired comparisons read selection rows. Finalists use
the saved closed predictions after research is stopped, with no extra fit.
Small/single-period validation may report confirmation unavailable explicitly.

For five training periods and the usual two rolling blocks this gives selection
1–3 -> 4 and confirmation 1–4 -> 5. These are data-dependent periods, not names or
values hardcoded into the implementation. Full deployment refit may use all
training targets; its fit diagnostics do not guide the closed comparison.

## Verification and limits

Regression checks perturb only closed labels and verify unchanged rolling fits
and predictions for both regression and classification. Tool checks exercise a
failed analysis and repair, NumPy/pandas missing aggregates, sealed owned backend
training, process refusal and timeout cleanup. Feedback tests include changing
background job IDs and a losing research candidate with an unchanged leader.

The user-supplied event excerpts do not contain the source/tracebacks of their
six failed analyses. This release makes those failures observable and fixes
reproduced execution/interface defects; it does not invent their
historical causes. Scripted LLM checks do not establish improved live reasoning
or a leaderboard gain. The leaderboard execution has no available logs here.

`scripts/check.py --jobs 3` passed all 636 tests. A fixed-parameter ablation on
public regression data (fit periods 1–3, score period 4 only, never score period 5)
compared the best supplied log's four missing indicators with the identical
strong learner without them. RMSE without/with flags was 32.8544/32.8936 for seed
42 and 33.5224/32.5871 for seed 1042. The temporal split stayed fixed while model
seeds varied. This is a development comparison, not independent confirmation:
it supports retaining the representation as a candidate, not claiming a
seed-independent gain or newly discovered information in the missing indicators.

The flat-ZIP verifier accepts multiple evaluated configurations of the submitted
representation and checks the compact scheduler's saved checkpoint event.
It also checks successful Python analyses, the named executed feature program
and the final CSV IDs, columns, finite regression values/class vocabulary.

The release ZIP passed that verifier for regression and classification: each
completed two successful analyses, saved one checkpoint, evaluated the submitted
representation and published a valid CSV. Replay decisions were scripted and
each invocation used a 90-second test override. The shipped readable profile
was checked separately: live Qwen, adaptive research, 10800 seconds per task.
All 82 packaged Python modules match their source bytes, all text resources
match, and requirements match the official participant archive byte for byte.
No live API request, three-hour live run or leaderboard submission was made.
