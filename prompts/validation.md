Choose a validation protocol from the task description and technical schema. Return only JSON:
{"strategy":"...","parameters":{"valid_fraction":0.2},"seed":42,"reason":"brief evidence-based justification"}.

Supported strategies: random_holdout; stratified_holdout (classification only);
group_holdout (parameters.column is the group feature); temporal_holdout (parameters.column is an ordered time/period feature).
For temporal data, later complete periods form validation and earlier periods form training.
Do not select a column only because its name looks like time: use the task semantics.
Use only training data for split design. A validation fraction between 0.05 and 0.5 is required.
Use the same protocol for all candidates. Never tune the split to make a candidate's metric look better.
Use semantic_task and inference-time availability. Distinguish future observations
of familiar entities, unseen entities, and random observations. A date is not
automatically a forecast; repeats do not prove a persistent entity identifier.
When the task explicitly supports a scenario, add parameters.scenario with
deployment (future_observations/new_entities/random_observations/unknown), quote
(an exact task excerpt), and time_column or entity_columns when relevant.
The executor checks evidence and strategy consistency. Unknown scenarios stay
unknown; record the alternative check needed in reason. Confirmation remains
closed; too few temporal periods must remain explicitly unavailable.
