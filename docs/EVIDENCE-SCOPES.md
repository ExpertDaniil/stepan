# Evidence scopes and model interpretations

The measured +1.8756 RMSE regression of the joint missing-flags/log-area pipeline did
not identify either component's contribution. A completed tuning budget did not
establish a model-family ceiling. These are limitations of the evidence, not evidence
that those research directions can never work.

The dispatcher computes comparison_evidence: trusted scores/direction, compatible
data/task/protocol/seed/fidelity, changed parameters, source/manifest changes and frozen
runtime agreement. Same-source owned recipes with one changed parameter and matching
runtime snapshots can receive a single_parameter_contrast scope. This remains a local
configuration observation on an adaptively used split, not statistical significance
or general causal attribution. Arbitrary custom source changes keep whole-configuration
scope even when a manifest claims only one component changed.

ResearchAction optionally includes structured claims with scope, statement, evidence_ids
and component. They are assessed without an additional LLM call. Invisible/unmeasured
references, joint component attribution and family-exhaustion claims remain unverified.
Only controller-computed score differences/limited contrasts become authoritative;
the original model text always retains model_statement_verified=false. Unsupported
claims create feedback, while a valid executable experiment may still proceed.

This is not a semantic truth detector for every sentence of free-form prose. Hypotheses,
priority reasons and interpretations remain labelled model-authored. They cannot close
ideas, mutate metric validity or create family/component bans. No mandatory single-change
sequence or extra model critic has been introduced. Synthetic scope checks are written,
not run; measured benchmark benefit awaits a real run.
