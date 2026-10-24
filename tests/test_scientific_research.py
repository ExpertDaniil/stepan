"""Scientific records, true fit-only OOF, episode budgets and process isolation."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from agent.budget import Budget
from agent.context import output_schemas
from agent.errors import AgentError
from agent.fit_diagnostics import inner_folds, error_profile, execute_oof
from agent.llm.gateway import Gateway
from agent.orchestrator import Orchestrator
from agent.research_actions import validate_test_plan, validate_research_notes
from agent.runner import Runner
from agent.scientific_research import episode_context, ScientificResearch, synthetic_notes
from tests.helpers import Harness


def decision(user=None, **changes):
    action = {'action': 'catalog', 'recipe_id': 'linear-classifier-v1', 'hyperparameters': {'C': 1.0},
              'code': None, 'tuning': None, 'feature_steps': [], 'evidence_ids': [],
              'hypothesis': 'Test the synthetic learned reference on a fixed protocol.',
              'expected_effect': 'Measure a complete configuration without a claimed gain.'}
    action.update(changes)
    action['research_notes'] = synthetic_notes(action, user or {})
    return action


class Memorizer(BaseEstimator):
    def fit(self, X, y):
        self.lookup_ = dict(zip(X.index, y))
        return self

    def predict(self, X):
        return np.asarray([self.lookup_.get(i, 'A') for i in X.index])


class ScientificResearchTests(unittest.TestCase):
    def test_inner_folds_respect_groups_periods_and_class_support(self):
        data = pd.DataFrame({'y': ['A', 'B']*60, 'g': np.repeat(range(12), 10),
                             'time': np.repeat(range(12), 10)})
        spec = {'task_kind': 'multiclass_classification', 'target_column': 'y'}
        for strategy, column in [('stratified_holdout', None), ('group_holdout', 'g'), ('temporal_holdout', 'time')]:
            folds = inner_folds(data, spec, {'strategy': strategy, 'parameters': {'column': column}}, 42)
            seen = set()
            for split in folds:
                a, b = split['fit'], split['predict']
                self.assertFalse(set(a) & set(b))
                self.assertFalse(seen & set(b)); seen.update(b)
                if column == 'g':
                    self.assertFalse(set(data.iloc[a].g) & set(data.iloc[b].g))
                if column == 'time':
                    self.assertLess(data.iloc[a].time.max(), data.iloc[b].time.min())
            self.assertEqual(len(seen), 120 if column != 'time' else 90)
        with self.assertRaises(ValueError):
            inner_folds(pd.DataFrame({'y': ['A']*59+['B']}), spec,
                        {'strategy': 'random_holdout', 'parameters': {}}, 42)

    def test_error_discovery_finds_late_feature_and_excludes_every_target(self):
        n = 600
        data = pd.DataFrame({f'early_{i}': [0]*n for i in range(6)})
        data['important_late'] = np.repeat([0, 1], n//2)
        data['target'] = np.arange(n)
        data['target_2'] = np.arange(n)*2
        spec = {'task_kind': 'regression', 'target_column': 'target', 'target_columns': ['target', 'target_2'], 'excluded_features': []}
        y = data[['target', 'target_2']].to_numpy()
        p = y+data.important_late.to_numpy()[:, None]*10
        report = error_profile(data, y, p, np.arange(n)%3, spec)
        self.assertEqual(report['scanned_columns'], 7)
        self.assertEqual(report['hard_groups'][0]['feature'], 'important_late')
        self.assertEqual(report['hard_groups'][0]['folds_with_higher_error'], 3)
        self.assertEqual(report['hard_groups'][0]['prediction_bias'], 10)
        self.assertTrue(all(r['feature'] not in {'target', 'target_2'} for r in report['hard_groups']))
        categorical = pd.DataFrame({'category': ['nan']*200+[None]*200+['A']*200})
        report = error_profile(categorical, np.zeros(n), np.repeat([10, 0, 1], 200),
                               np.arange(n)%3, spec)
        literal = next(r for r in report['hard_groups'] if r['group'] == 'nan')
        self.assertEqual(literal['rows'], 200)
        self.assertEqual(literal['mean_error'], 10)

    def test_oof_uses_fresh_models_and_never_predicts_its_training_rows(self):
        spec = {'task_kind': 'multiclass_classification', 'target_column': 'y',
                'metric': {'prediction_kind': 'labels'}, 'allowed_labels': ['A', 'B']}
        X = pd.DataFrame({'x': range(60)}, index=range(100, 160))
        y = pd.Series(['A', 'B']*30, index=X.index)
        splits = inner_folds(pd.DataFrame({'y': y}), spec, {'strategy': 'stratified_holdout', 'parameters': {}}, 42)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'oof.csv'
            result = execute_oof(SimpleNamespace(make_pipeline=lambda task, params: Memorizer()), spec, {}, X, y,
                {'cpu_threads': 1, 'diagnostic': {'folds': splits}, 'prediction_path': str(output)}, None, ExitStack())
            actual = pd.read_csv(output)
        self.assertEqual(result['rows'], 60)
        self.assertEqual(actual['__row_index'].tolist(), list(range(100, 160)))
        self.assertEqual(actual.prediction.tolist(), ['A']*60)

    def test_new_context_requires_falsifier_and_an_explicit_question_switch(self):
        protocol = {'version': 'scientific-v1'}
        action = decision()
        validate_test_plan(action, protocol)
        with self.assertRaises(AgentError):
            validate_test_plan({k: v for k, v in action.items() if k != 'research_notes'}, protocol)
        action['research_notes']['test_plan']['falsifier'] = action['research_notes']['test_plan']['prediction']
        with self.assertRaises(AgentError):
            validate_test_plan(action, protocol)
        action = decision()
        with self.assertRaises(AgentError):
            validate_test_plan(action, {**protocol, 'episode': {'idea_id': 'known'}})
        action['research_notes']['test_plan']['evidence_ids'] = ['unseen']
        with self.assertRaises(AgentError):
            validate_research_notes(action['research_notes'], {}, set(), set())
        validate_test_plan({'action': 'catalog'}, {})  # Old contexts are readable.

    def test_episode_accounts_all_related_trials_and_ignores_oof_and_refit(self):
        decisions = [{'researcher_id': 'r', 'idea_id': 'same', 'question': 'q', 'test_plan': {},
                      'evidence_ids': [], 'observations': [{'experiment_id': f'e{i}', 'state': 'evaluated', 'primary_metric': {'value': i}}]}
                     for i in range(20)]
        jobs = [SimpleNamespace(reported=True, phase='train', diagnostic=False, experiment_id=f'e{i}',
                                event={'duration_seconds': 2}) for i in range(20)]
        jobs += [SimpleNamespace(reported=True, phase='train', diagnostic=True, experiment_id='e1', event={'duration_seconds': 99}),
                 SimpleNamespace(reported=True, phase='refit', diagnostic=False, experiment_id='e1', event={'duration_seconds': 99})]
        episode = episode_context({'decisions': decisions}, 'r', jobs, 60)
        self.assertEqual(episode['charged_training_seconds'], 40)
        self.assertEqual(episode['remaining_seconds'], 20)
        self.assertEqual(len(episode['measured_experiment_ids']), 20)

    def test_diagnostic_process_never_mutates_candidate_state_or_evaluation(self):
        for guarded in (False, True):
            with self.subTest(guarded=guarded):
                self.run_diagnostic_process(guarded)

    def run_diagnostic_process(self, guarded):
        h = Harness()
        try:
            cfg = replace(h.config, time_budget_seconds=300, research_mode='adaptive' if guarded else 'catalog')
            c = Orchestrator.__new__(Orchestrator)
            c.config, c.artifacts, c.store, c.bus, c.log, c.graph = cfg, h.artifacts, h.store, h.bus, h.log, None
            c.budget = Budget(cfg, time.monotonic(), h.log)
            c.runner = Runner(h.artifacts, cfg, c.budget, h.log)
            plan, code = h.plan(), h.code
            if guarded:
                from tests.test_adaptive_research import SOURCE
                from tools.code import write_code
                code = write_code({'researcher_id': 'test', 'filename': 'guarded.py', 'content': SOURCE,
                                   'declared_entrypoint': 'make_pipeline'}, h.artifacts)['artifact']
                plan.update(recipe_id='python-pipeline-v1', code_artifact_id=code['artifact_id'],
                            model={'family': 'custom', 'hyperparameters': {'alpha': .03}})
            eid = h.store.reserve(plan, code['sha256'])['experiment_id']
            fit = h.artifacts.write_json('diag-fit.json', list(range(60)))
            unused = h.artifacts.write_json('diag-unused.json', [])
            frame = pd.read_csv(h.task.parent/'train.csv')
            folds = inner_folds(frame.iloc[:60].rename(columns={'label': 'y'}),
                                {'task_kind': 'multiclass_classification', 'target_column': 'y'},
                                {'strategy': 'stratified_holdout', 'parameters': {}}, 42)
            lease = c.lease('diagnostic-owner', 'train', c.resources(20))['lease']
            job = c.runner.start_diagnostics({'lease': lease, 'lease_owner_id': 'diagnostic-owner',
                'train_rows_artifact_id': fit['artifact_id'], 'valid_rows_artifact_id': unused['artifact_id'],
                'diagnostic': {'folds': folds}}, h.spec, h.store.get(eid))
            with patch.object(h.store, 'apply_event', side_effect=AssertionError('OOF must not call C15')):
                event = c.wait_jobs([job])[job.job_id]
            self.assertEqual(event['state'], 'succeeded', event)
            self.assertEqual(h.store.get(eid)['state'], 'queued')
            self.assertIsNone(h.store.experiment_evaluation(eid))
            self.assertEqual(c.budget.leases, {})
            self.assertTrue(job.reported)
            self.assertEqual(pd.read_csv(job.prediction_path)['__row_index'].tolist(), list(range(60)))
            self.assertIn('fit_oof_predictions', [h.artifacts.meta(i)['kind'] for i in event['output_artifact_ids']])
        finally:
            if 'c' in locals():
                for job in c.runner.active():
                    c.cancel_job(job, 'test cleanup', 'cancelled'); c.wait_jobs([job])
            h.close()

    def test_deep_dialogue_is_funded_and_keeps_provider_application_unverified(self):
        h = Harness()
        try:
            cfg = replace(h.config, research_mode='adaptive', time_budget_seconds=3000,
                llm_max_output_tokens=256, llm_max_output_tokens_limit=1024,
                llm_research_dialogue_seconds=30, research_deep_dialogue_seconds=60)
            budget = Budget(cfg, time.monotonic(), h.log)
            requests = []
            malformed = False
            class Transport:
                def complete(self, request, timeout, context):
                    requests.append(request)
                    value = decision()
                    if malformed:
                        value.pop('research_notes')
                    return {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant',
                        'content': json.dumps(value), 'reasoning_content': 'Synthetic fixture field.'}}],
                        'usage': {'prompt_tokens': 20, 'completion_tokens': 20, 'completion_tokens_details': {'reasoning_tokens': 7}}}
                def close(self):
                    pass
            user = {'scientific_protocol': {'version': 'scientific-v1', 'deliberation': {'extended_requested': True}}}
            art = h.artifacts.write_json('deep-context.json', {'messages': [{'role': 'system', 'content': 'Synthetic test.'},
                {'role': 'user', 'content': json.dumps(user)}], 'metadata': {'schema_id': 'ResearchAction-v1'}})
            gateway = Gateway(cfg, budget, h.artifacts, h.bus, h.log, output_schemas(h.bus), transport=Transport())
            answer = gateway.complete({'purpose': 'research_proposal', 'context_artifact_id': art['artifact_id'],
                'response_schema_id': 'ResearchAction-v1', 'tool_schema_ids': [], 'max_output_tokens': 256,
                'deadline_seconds': 90, 'attempt_limit': 1})
            self.assertEqual(requests[0]['max_tokens'], 512)
            self.assertNotIn('extra_body', requests[0])
            self.assertIn('test_plan', answer['parsed_payload']['research_notes'])
            self.assertEqual(budget.token_snapshot()['output_reserved'], 0)
            malformed = True
            with self.assertRaises(AgentError):
                gateway.complete({'purpose': 'research_proposal', 'context_artifact_id': art['artifact_id'],
                    'response_schema_id': 'ResearchAction-v1', 'tool_schema_ids': [], 'max_output_tokens': 256,
                    'deadline_seconds': 90, 'attempt_limit': 1})
            self.assertEqual(budget.token_snapshot()['output_reserved'], 0)
            gateway.close()
        finally:
            h.close()


class ScientificReviewTests(unittest.TestCase):
    def test_expensive_preflight_needs_the_displayed_measured_reference(self):
        h = Harness()
        try:
            feedback, events = [], []
            c = SimpleNamespace(artifacts=h.artifacts,
                budget=SimpleNamespace(duration=SimpleNamespace(estimate=lambda plan: 45)),
                log=SimpleNamespace(event=lambda *a, **k: events.append(k)),
                record_research_feedback=lambda *a, **k: feedback.append(a))
            science = ScientificResearch(c)
            plan = h.plan()
            intent = {'research_notes': decision()['research_notes'], 'evidence_ids': [], 'tuning': None,
                      'strongest_measured_reference': 'measured-base', 'scientific_protocol': {}}
            self.assertIsNotNone(science.review(plan, intent, 'branch'))
            self.assertTrue(feedback)
            intent['evidence_ids'] = ['measured-base']
            self.assertIsNone(science.review(plan, intent, 'branch'))
            self.assertEqual(events[-1]['admission'], 'admitted')
        finally:
            h.close()

    def test_oof_visibility_does_not_import_sibling_reports(self):
        from agent.fit_diagnostics import FitDiagnostics
        diagnostics = FitDiagnostics(SimpleNamespace())
        diagnostics.records = [{'artifact_id': 'a', 'experiment_id': 'own'}, {'artifact_id': 'b', 'experiment_id': 'sibling'}]
        result = diagnostics.visible(['own'])
        self.assertEqual(result, [{'artifact_id': 'a', 'experiment_id': 'own'}])
        result[0]['artifact_id'] = 'changed'
        self.assertEqual(diagnostics.records[0]['artifact_id'], 'a')
        from agent.scientific_research import compact_protocol
        full = {'episode': {'remaining_seconds': 23, 'last_test_plan': {'prediction': 'x'*1000}},
            'fit_diagnostics': [{'artifact_id': 'a', 'experiment_id': 'own',
                'report': {'hard_groups': [{'rows': i} for i in range(8)]}}]}
        reduced = compact_protocol(full)
        self.assertEqual(reduced['episode']['remaining_seconds'], 23)
        self.assertTrue(reduced['episode']['last_test_plan_is_excerpt'])
        self.assertEqual(len(reduced['fit_diagnostics'][0]['report']['hard_groups']), 4)
        self.assertEqual(reduced['fit_diagnostics'][0]['report']['omitted_hard_groups'], 4)
        self.assertEqual(len(full['fit_diagnostics'][0]['report']['hard_groups']), 8)

    def test_spent_episode_review_uses_new_cost_and_requires_new_evidence(self):
        h = Harness()
        try:
            notes = decision()['research_notes']
            notes['idea_id'] = 'idea'
            notes['test_plan'].update(continuation='followup', evidence_ids=['old'])
            memory = {'decisions': [{'idea_id': 'idea', 'researcher_id': 'researcher-1', 'question': 'q',
                'observations': [{'experiment_id': 'e', 'state': 'evaluated', 'primary_metric': {'value': 1}}],
                'evidence_ids': ['old'], 'configuration': {'plan_id': 'previous'}}]}
            job = SimpleNamespace(reported=True, phase='train', diagnostic=False, experiment_id='e', event={'duration_seconds': 50})
            c = SimpleNamespace(artifacts=h.artifacts, runner=SimpleNamespace(jobs={'j': job}),
                graph=SimpleNamespace(research_memory=lambda branch, donors: deepcopy(memory)),
                budget=SimpleNamespace(duration=SimpleNamespace(estimate=lambda plan: 10)),
                log=SimpleNamespace(event=lambda *a, **k: None), record_research_feedback=lambda *a, **k: None)
            science = ScientificResearch(c)
            intent = {'research_notes': notes, 'evidence_ids': ['old'], 'tuning': None,
                'scientific_protocol': {'episode': {'idea_id': 'idea', 'allowance_seconds': 40, 'remaining_seconds': 40}}}
            self.assertIsNotNone(science.review(h.plan(), intent, 'branch'))
            notes['test_plan'].update(continuation='reconsider', switch_reason='A newly observed report changes the priority of this question.')
            self.assertIsNotNone(science.review(h.plan(), intent, 'branch'))
            notes['test_plan']['evidence_ids'].append('new')
            self.assertIsNone(science.review(h.plan(), intent, 'branch'))
        finally:
            h.close()


if __name__ == '__main__':
    unittest.main()
