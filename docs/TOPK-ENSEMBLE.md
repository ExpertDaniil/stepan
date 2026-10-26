# Top-k ensemble candidate

`agent/ensemble.py` + `VersionSearch.maybe_add_ensemble`. After `ENSEMBLE_START_FRACTION`
(default 0.5) of the budget, every 15 min (max 3 per run) the controller takes up to
`FINAL_ENSEMBLE_SIZE` (default 3) best measured owned pipelines (catalogue recipes or
owned feature-program pipelines, incl. slice corrections) within 3% of the best
selection score, drops near-duplicates (prediction correlation/agreement >= 0.999), and
averages their saved selection predictions (mean for values/probabilities, plurality
vote for labels, ties to the better member). Only if that average beats the best member
is an ensemble queued as an ordinary version (`ENSEMBLE_SOURCE`, members rebuilt by owned
constructors with the validation policy bound). It is then judged by the Evaluator, the
noise gate and closed confirmation like any candidate. In a mock run the trained ensemble
scored exactly its precheck (1.12129), confirming deterministic reconstruction.
`FINAL_ENSEMBLE_SIZE=0` disables it. Custom LLM code members are not combined.
