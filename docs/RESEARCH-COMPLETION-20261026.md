# Completing the slice research path

This release continues Claude's `1abe56a` WIP. The three task/metric fixes remain:
stated supervised task kind beats low-cardinality dtype guesses, numeric binary
AUC output defaults to probabilities, and MAPE diagnostics use original-scale
metric losses. No public dataset names or column rules are added to the runtime.

## Independence and execution cost

The last diagnostic fold is excluded from every discovery model's fit. Merely
splitting the finished OOF table was insufficient: random/group OOF discovery
models had learned from that fold's labels. Key discovery and entity history
now also use only the union of discovery fit rows. Temporal as-of behavior stays
strict. Tests perturb held labels for regression and classification under
random, grouped and temporal protocols, then fit real models and compare their
predictions. Discovery predictions remain unchanged.

The number of OOF base fits is unchanged. Discovery random/group models use
smaller training sets; small class/group support can make this diagnostic
unavailable. These are honest costs of isolation, not independent extra data.
The lab still selects among multiple remedies on one inner holdout; downstream
selection and closed confirmation remain necessary. Refit can change how well
a frozen correction matches its base.

OOF prediction, history discovery and residual-model analysis now share one
existing Runner process/lease, CPU allocation, RSS guard, hard timeout and
publication-time cancellation. The worker writes a retained `fit_oof_analysis`
artifact; the dispatcher verifies rows and publishes it without training another
model. Pending diagnostics survive temporarily insufficient budget and resource
availability without consuming attempts.

## Useful evidence without changing the search objective

See [NOISE-AWARE-SELECTION.md](NOISE-AWARE-SELECTION.md). Raw best ranking is
preserved. Branch-local utility references, direct solution reads and recovery
generation avoid treating every background parameter trial as a new research
direction. Submitted candidate results, including losses, remain visible.
Bootstrap uses whole declared groups/periods; a one-period comparison stays
explicitly unconfirmed and can still advance on a material measured gain.

Search keeps a refit reserve for raw-best even while the publication gate retains
the existing CSV. The same rounded lease estimate plus dispatcher handoff is used
by admission, finalist filtering and final refit; a fractional estimate must not
be accepted and then truncated into an insufficient integer timeout.

## Verification of usefulness

The previous failing mining test prohibited any `x > threshold` rule, even one
found by chance from discovery errors. It now checks the actual boundary:
changing only held targets/predictions cannot change mined rules.

The end-to-end fixture provides a reproducible rare discontinuity that a linear
reference misses. Scripted model messages read real delivered diagnostic cards;
they cannot supply scores or force a correction to pass. The test checks the
full path through the leased lab, referenced remedy, training, paired evaluation,
closed confirmation, refit and a valid output CSV. It independently scores that
CSV on synthetic test targets and requires improvement over the same linear
reference. An explainable region without a useful menu remedy is not forced to
produce one.

An additional fixed-policy experiment used three independent synthetic dataset
generations (2200 fit, 500 selection, 500 untouched test rows). The selected shift
was fitted on discovery only. No test target chose a rule or its parameters:

| Dataset seed | Baseline test RMSE | Corrected test RMSE |
|---|---:|---:|
| 11 | 3.10234 | 1.77719 |
| 29 | 3.15319 | 1.95083 |
| 47 | 3.14581 | 1.96250 |

These exercise one controlled mechanism with different generated data, not
training-seed replication, a general AutoML-policy benchmark or a leaderboard
gain. Live Qwen reasoning and a three-hour provider run were not measured here.

The full `scripts/check.py --jobs 3` run passed 709 tests before the final refit
reserve adjustment. After that adjustment, 29 focused finalization/budget/boundary
checks passed; six diagnostic-budget checks also passed. The new waiting-budget
and raw-finalist reserve cases are included in those focused runs. The final flat
ZIP completed regression and classification replays, each with two successful
Python analyses, a checkpoint, an evaluated feature program and a valid CSV.
Replay uses scripted LLM responses and a 90-second test override; the shipped
profile remains live/adaptive with 10800 seconds per task.
