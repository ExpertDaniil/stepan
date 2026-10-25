"""Owner B: task semantics and validation hypothesis, through the shared gateway."""
import json
import uuid
import re
from pathlib import Path

from agent.config import ROOT
from agent.errors import AgentError


def recover_contract(task, profile, artifacts):
    """Conservative consensus, independent of the provider and exact metric.

    A single missing train column is insufficient on its own. Require an
    agreeing sample, an explicit task mention and dictionary roles when present.
    Unsupported probability mappings and conflicting sources remain unresolved.
    """
    import pandas as pd
    from tools.data import read_table
    from agent.task import METRICS
    full = artifacts.read_json(profile['profile_artifact_id'])
    train_cols, test_cols = full['train_columns'], full['test_columns']
    missing = set(train_cols)-set(test_cols)
    if len(missing) != 1 or not profile.get('sample_submission_artifact_id'):
        return None
    target = next(iter(missing))
    text = task['task_text']
    if not re.search(r'(?<!\w)'+re.escape(target)+r'(?!\w)', text):
        return None
    dictionary = full.get('data_dictionary', [])
    if dictionary:
        declared = [r.get('feature') for r in dictionary if str(r.get('role', '')).lower() == 'target']
        if declared != [target]:
            return None
    sample = pd.read_csv(artifacts.resolve(profile['sample_submission_artifact_id']), dtype=str, keep_default_na=False)
    ids = [c for c in sample if c in test_cols]
    predicted = [c for c in sample if c not in test_cols]
    # Multi-target/probability output needs an explicit adapter, never dtype guessing.
    if predicted != [target] or not ids or len(sample) != full['test_rows']:
        return None
    if dictionary and any(not any(r.get('feature') == c and str(r.get('role', '')).lower() in {'id', 'identifier'} for r in dictionary) for c in ids):
        return None
    test_ids = pd.read_csv(artifacts.resolve(profile['test_artifact_id']), usecols=ids, dtype=str, keep_default_na=False)[ids]
    if not sample[ids].equals(test_ids) or test_ids.duplicated().any() or test_ids.eq('').any().any():
        return None
    inputs = {Path(p).name for p in task['candidate_paths'].values()}
    paths = {p for p in re.findall(r'[\w./-]+\.csv', text) if Path(p).name not in inputs}
    if len(paths) != 1:
        return None
    output_path = next(iter(paths))
    root = Path(task['task_root']).resolve()
    if not (root/output_path).resolve().is_relative_to(root):
        return None
    from agent.task_rules import infer_metric
    metric_id = infer_metric(text)
    regression = bool(re.search(r'регресси|\bregression\b', text, re.I)) or metric_id in {'rmse', 'mae', 'r2'}
    classification = bool(re.search(r'классификац|\bclassification\b', text, re.I)) or metric_id in {'f1_macro', 'f1_weighted', 'accuracy'}
    if regression == classification or re.search(r'probabilit|вероятност', '\n'.join(line for line in text.splitlines() if target in line), re.I):
        return None
    y = read_table(artifacts.resolve(profile['train_artifact_id']), dtype={target: str} if classification else None)[target]
    if y.isna().any():
        return None
    labels = sorted(y.astype(str).unique().tolist()) if classification else []
    if classification and (len(labels) < 2 or not set(sample[target]) <= set(labels)):
        return None
    output = {'path': str((root/output_path).resolve()), 'columns': list(sample.columns), 'id_columns': ids,
              'prediction_columns': predicted, 'prediction_kind': 'values' if regression else 'labels',
              'row_order': 'test', 'encoding': 'utf-8', 'separator': ',', 'write_index': False}
    spec = {'task_spec_id': 'emergency-'+profile['dataset_id'], 'dataset_id': profile['dataset_id'],
            'task_kind': 'regression' if regression else 'binary_classification' if len(labels) == 2 else 'multiclass_classification',
            'target_column': target, 'target_columns': [target], 'allowed_labels': labels, 'output': output,
            'excluded_features': ids, **{k: profile[k] for k in ('train_artifact_id', 'test_artifact_id', 'dictionary_artifact_id', 'sample_submission_artifact_id')}}
    draft = None
    if metric_id and METRICS[metric_id][1] == output['prediction_kind']:
        direction, kind = METRICS[metric_id]
        draft = {'task_kind': spec['task_kind'], 'target_column': target,
                 'metric': {'metric_id': metric_id, 'direction': direction, 'prediction_kind': kind, 'parameters': {}}, 'output': output}
    return {'spec': spec, 'draft': draft, 'target': y,
            'provenance': {'task_artifact_id': task['task_text_artifact_id'], 'profile_artifact_id': profile['profile_artifact_id'],
                           'sample_artifact_id': profile['sample_submission_artifact_id'], 'rule': 'schema_sample_dictionary_task_consensus'}}


class TaskInterpreter:
    def __init__(self, config, artifacts, bus, gateway, task, profile, demo=None):
        self.config, self.artifacts, self.bus, self.gateway = config, artifacts, bus, gateway
        self.task, self.profile, self.demo = task, profile, demo

    def ask(self, schema_id, purpose, prompt_file, content):
        messages = [{"role": "system", "content": (ROOT / "prompts" / prompt_file).read_text(encoding="utf-8")},
                    {"role": "user", "content": json.dumps(content, ensure_ascii=False)}]
        context = self.artifacts.write_json("contexts/bootstrap-" + uuid.uuid4().hex[:10] + ".json",
                    self.gateway.log.redactor.clean({"messages": messages, "metadata": {
                        "schema_id": schema_id, "demo_spec": self.demo,
                        "bootstrap_profile_artifact_id": self.profile['profile_artifact_id']}}), "llm_context")
        return self.bus.call("C10", {"purpose": purpose, "context_artifact_id": context["artifact_id"],
                                     "response_schema_id": schema_id, "tool_schema_ids": [],
                                     "max_output_tokens": self.config.llm_bootstrap_output_tokens,
                                     "deadline_seconds": self.config.llm_operation_timeout_seconds, "attempt_limit": 3}, self.gateway.complete)["data"]["parsed_payload"]

    def interpret(self, payload):
        try:
            draft = self.ask("TaskDraft-v1", "task_interpretation", "task-interpreter.md",
                         {"task_markdown": self.task["task_text"], "candidate_paths": payload["candidate_paths"],
                          "profile": self.profile["summary"], "task_root": self.task["task_root"],
                          "semantic_source_rules": "task quotes reference this exact Markdown with record=null; dictionary quotes reference zero-based rows of profile.data_dictionary. Omitted metadata remains unknown."})
        except Exception as error:
            recovered = getattr(self, 'recovered', None)
            if not recovered or not recovered['draft']:
                raise
            self.gateway.log.event('task.interpretation_recovered', code=getattr(error, 'code', type(error).__name__), provenance=recovered['provenance'])
            draft = recovered['draft']
        return {"draft": draft, "evidence": ["Source task artifact: " + self.task["task_text_artifact_id"]], "unresolved_fields": []}

    def deployment_facts(self, spec):
        """Label-free train/test facts; failures leave the facts empty, never the run."""
        cached = getattr(self, '_deployment_facts', None)
        if cached is not None:
            return cached
        from agent.data_facts import deployment_facts
        from tools.data import read_table
        try:
            train = read_table(self.artifacts.resolve(self.profile['train_artifact_id']))
            test = read_table(self.artifacts.resolve(self.profile['test_artifact_id']))
            facts = deployment_facts(train, test, spec.get('target_columns') or [spec['target_column']],
                                     spec.get('excluded_features', []))
        except Exception as error:
            facts = {'errors': ['tables: '+type(error).__name__+': '+str(error)[:200]],
                     'future_period': None, 'duplicates': None}
        self._deployment_facts = facts
        self.artifacts.write_json('task/deployment-facts.json', facts, 'deployment_facts')
        self.gateway.log.event('validation.deployment_facts', **facts)
        return facts

    def choose_validation(self, spec):
        """Data facts first, then explicit task declarations, then the LLM; never fatal."""
        from agent.data_facts import apply_duplicate_groups
        facts = self.deployment_facts(spec)
        return apply_duplicate_groups(self._choose_validation(spec, facts), facts)

    def _choose_validation(self, spec, facts):
        from agent.data_facts import scenario_proposal
        from agent.task_rules import validation_requirement, bind_requirement
        observed = scenario_proposal(facts, self.config.seed)
        if observed:
            self.gateway.log.event('validation.data_scenario', proposal=observed)
            return observed
        full = self.artifacts.read_json(self.profile['profile_artifact_id'])
        try:
            requirement = validation_requirement(self.task['task_text'],
                [c for c in full['train_columns'] if c not in spec.get('target_columns', [spec['target_column']])],
                full.get('data_dictionary', []))
        except AgentError as error:
            # Conflicting prose is evidence to record, not a reason to submit a constant.
            self.gateway.log.event('validation.contract_conflict', code=error.code, message=str(error)[:300])
            requirement = None
        if requirement and (requirement['column'] or requirement['strategy'] in {'random_holdout', 'stratified_holdout'}):
            proposal = {'strategy': requirement['strategy'], 'seed': self.config.seed,
                'parameters': {'valid_fraction': .2, **({'column': requirement['column']} if requirement['column'] else {})},
                'reason': 'Explicit task deployment and column declarations; no feature-name inference.'}
            self.gateway.log.event('validation.contract_resolved', requirement=requirement)
            return bind_requirement(proposal, requirement)
        try:
            from agent.semantic_planning import load_task_model
            proposal = self.ask("ValidationProposal-v1", "result_analysis", "validation.md",
                            {"task_markdown": self.task["task_text"], "task_spec": spec, "profile": self.profile["summary"],
                             'semantic_task': load_task_model(spec, self.artifacts), 'data_facts': facts})
        except Exception as error:
            self.gateway.log.event('validation.bootstrap_unavailable', code=getattr(error, 'code', type(error).__name__))
            return self.default_validation(spec, 'Provider unavailable; no scenario inferred from dtype or names.')
        try:
            return bind_requirement(proposal, requirement)
        except AgentError as error:
            self.gateway.log.event('validation.contract_conflict', code=error.code, message=str(error)[:300],
                                   proposal=proposal, requirement=requirement)
            return proposal

    def default_validation(self, spec, reason):
        return {'strategy': 'random_holdout' if spec['task_kind'] == 'regression' else 'stratified_holdout',
                'seed': self.config.seed, 'parameters': {'valid_fraction': .2}, 'reason': reason}
