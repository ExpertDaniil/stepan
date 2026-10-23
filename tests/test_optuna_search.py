"""Synthetic optimizer/dispatcher checks; injected scores do not measure ML quality."""
from copy import deepcopy
import json
import time
import unittest
from unittest.mock import patch

from agent.context import output_schemas
from agent.errors import AgentError
from agent.llm.gateway import Gateway
from agent.local_search import LocalSearch
from agent.research_actions import ACTION_SCHEMA, validate_action
from agent.researcher import Researcher
from agent.tuning_space import normalize_space
from tests.graph_helpers import GraphHarness
from tests.test_adaptive_research import action, SOURCE
from tests.test_research_memory import notes
from tools.code import write_code


def request(space, max_trials=3):
    return {'search_space': space, 'max_trials': max_trials, 'budget_seconds': 300}


class OptunaSearchTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800)
        self.g, self.c = self.h.g, self.h.controller
        self.c.graph = self.g
        self.c.llm_available = lambda: True
        self.c.resources = lambda: {'cpu_threads': 1, 'memory_limit_mb': 1024, 'timeout_seconds': 30}
        self.local = LocalSearch(self.c)
        self.c.local_search = self.local
        self.opt = self.local.optuna

    def tearDown(self):
        self.h.close()

    def anchor(self, recipe=None, params=None, source=None):
        ticket = self.h.issue()
        payload = self.h.proposal(ticket, params, recipe=recipe, source=source)
        return ticket, payload['proposal']['plan']

    def test_declared_increase_region_is_sampled_jointly_and_anchor_is_unchanged(self):
        _, plan = self.anchor('hist-gradient-regressor-v1')
        original = deepcopy(plan)
        space = {'max_iter': {'type': 'int', 'low': 500, 'high': 1500, 'step': 100},
                 'max_leaf_nodes': {'type': 'int', 'low': 31, 'high': 63}}
        self.local.start_series('researcher-1', self.h.root, plan, request=request(space))
        for _ in range(3):
            candidate = self.opt.next_plan('researcher-1', self.h.root)
            params = candidate['model']['hyperparameters']
            self.assertTrue(500 <= params['max_iter'] <= 1500)
            self.assertEqual((params['max_iter']-500) % 100, 0)
            self.assertTrue(31 <= params['max_leaf_nodes'] <= 63)
            self.assertEqual(params['target_transform'], plan['model']['hyperparameters']['target_transform'])
        self.assertEqual(plan, original)
        self.assertFalse(self.opt.has_series('researcher-1', self.h.root))

    def test_verified_history_warms_only_the_same_fixed_pipeline_and_resumes_study(self):
        self.h.result('researcher-1', {'alpha': 1.0}, 10)
        self.h.result('researcher-1', {'alpha': 2.0, 'target_transform': 'shifted_log1p'}, 5)
        _, plan = self.anchor(params={'alpha': 1.0})
        bounds = request({'alpha': {'type': 'float', 'low': .25, 'high': 4.0, 'log': True}})
        before = deepcopy(plan)
        self.opt.start_series('researcher-1', self.h.root, plan, bounds)
        session = self.opt.active[(self.h.root, 'researcher-1')]
        self.assertEqual(session['warm_start_trials'], 1)
        self.assertEqual(session['study_record']['study'].trials[0].value, 10)
        study_id = session['study_id']
        self.opt.start_series('researcher-1', self.h.root, plan, bounds)
        self.assertEqual(self.opt.active[(self.h.root, 'researcher-1')]['study_id'], study_id)
        self.assertEqual(plan, before)

    def test_custom_source_is_preserved_and_never_replaced_by_a_catalogue_factory(self):
        _, plan = self.anchor('python-pipeline-v1', {'alpha': 1.0}, SOURCE)
        self.opt.start_series('researcher-1', self.h.root, plan,
            request({'alpha': {'type': 'float', 'low': .01, 'high': 3.0, 'log': True}}))
        candidate = self.opt.next_plan('researcher-1', self.h.root)
        self.assertEqual(candidate['code_artifact_id'], plan['code_artifact_id'])
        self.assertEqual(candidate['feature_steps'], plan['feature_steps'])
        self.assertEqual(candidate['model']['family'], 'custom')
        self.assertEqual(self.g.artifacts.resolve(candidate['code_artifact_id']).read_text(), SOURCE)

    def test_study_identity_changes_with_fixed_parameters_source_features_and_split(self):
        _, plan = self.anchor('python-pipeline-v1', {'alpha': 1.0, 'fixed': 2}, SOURCE)
        space = {'alpha': {'type': 'float', 'low': .01, 'high': 3.0, 'log': True}}
        original = self.opt._scope(plan, space, self.h.root)
        variants = []
        changed = deepcopy(plan);changed['validation_id'] += '-different';variants.append(changed)
        changed = deepcopy(plan);changed['feature_steps'] = [{'step': 'different_joint_features'}];variants.append(changed)
        changed = deepcopy(plan);changed['model']['hyperparameters']['fixed'] = 3;variants.append(changed)
        code = write_code({'researcher_id': 'researcher-1', 'filename': 'variant.py', 'content': SOURCE+'\n# distinct implementation\n',
                           'declared_entrypoint': 'make_pipeline'}, self.g.artifacts)['artifact']
        changed = deepcopy(plan);changed['code_artifact_id'] = code['artifact_id'];variants.append(changed)
        for candidate in variants:
            self.assertNotEqual(self.opt._scope(candidate, space, self.h.root), original)

    def test_researcher_can_authorize_tuning_of_the_current_custom_base_without_rewriting_code(self):
        ticket, plan = self.anchor('python-pipeline-v1', {'alpha': 1.0}, SOURCE)
        payload = self.h.proposal(ticket, {'alpha': 1.0}, recipe='python-pipeline-v1', source=SOURCE)
        self.h.commit(self.g.invoke('G04', payload), 10)
        value = action('tune', recipe_id='python-pipeline-v1', hyperparameters={},
            tuning=request({'alpha': {'type': 'float', 'low': .01, 'high': 3.0, 'log': True}}))
        class Transport:
            def complete(transport, request, timeout, context):
                return {'choices': [{'message': {'role': 'assistant', 'content': json.dumps(value)}, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 100, 'completion_tokens': 100}}
            def close(transport):
                pass
        gateway = Gateway(self.g.config, self.g.budget, self.g.artifacts, self.g.bus, self.g.log,
                          output_schemas(self.g.bus), Transport())
        researcher = Researcher(self.g.bus, gateway, self.g.artifacts, self.g.spec, self.g.catalog, self.g.config, self.g.log)
        researcher.validation_id = self.g.validation['validation_id']
        proposal = researcher.propose(self.h.issue()['researcher_request']['payload'])
        tuned = proposal['plan']
        self.assertEqual(tuned['model']['hyperparameters'], {'alpha': 1.0})
        self.assertEqual(self.g.artifacts.resolve(tuned['code_artifact_id']).read_text(), SOURCE)
        self.assertEqual(tuned['feature_steps'], plan['feature_steps'])

    def test_pending_duplicate_does_not_release_another_trials_claim(self):
        _, plan = self.anchor(params={'alpha': 1.0})
        bounds = request({'alpha': {'type': 'float', 'low': 2.0, 'high': 2.0}}, max_trials=2)
        self.opt.start_series('researcher-1', self.h.root, plan, bounds)
        candidate = self.opt.next_plan('researcher-1', self.h.root)
        self.assertIsNotNone(candidate)
        second_plan = deepcopy(plan);second_plan['researcher_id'] = 'researcher-2'
        self.opt.start_series('researcher-2', self.h.root, second_plan, bounds)
        self.assertIsNone(self.opt.next_plan('researcher-2', self.h.root))
        first = self.opt.active[(self.h.root, 'researcher-1')]
        self.assertEqual(len(first['study_record']['claimed']), 1)
        self.assertFalse(first['trials'][0]['finished'])

    def test_completed_trial_reaches_optuna_and_research_memory(self):
        ticket, plan = self.anchor(params={'alpha': 1.0})
        bounds = request({'alpha': {'type': 'float', 'low': 2.0, 'high': 2.0}}, max_trials=2)
        researcher = Researcher(self.g.bus, None, self.g.artifacts, self.g.spec, self.g.catalog, self.g.config, self.g.log)
        context = self.g.artifacts.read_json(ticket['work_order']['context_id'])
        value = action('tune', hyperparameters={'alpha': 1.0}, tuning=bounds, research_notes=notes())
        researcher.save_decision(value, plan, ticket['researcher_request']['payload'], context,
                                {'output_artifact_id': plan['code_artifact_id']})
        self.opt.start_series('researcher-1', self.h.root, plan, bounds)
        candidate = self.opt.next_plan('researcher-1', self.h.root)
        payload = self.h.proposal(ticket, candidate['model']['hyperparameters'])
        payload['proposal']['plan'] = candidate
        payload['manifest'] = self.g.constructor.build(candidate)
        reply = self.g.invoke('G04', payload)
        eid = reply['lead_submission']['experiment_id']
        self.opt.bind(candidate, eid, ticket['work_order']['work_order_id'])
        self.h.commit(reply, 7)
        self.opt.sync()
        memory = self.g.research_memory(self.h.root, [])
        decision = memory['decisions'][0]
        self.assertEqual(decision['observations'][0]['primary_metric']['value'], 7)
        series = decision['tuning_series'][0]
        self.assertEqual(series['completed_trials'], 1)
        self.assertEqual(series['best_experiment_id'], eid)
        self.assertEqual(series['trials'][0]['experiment_id'], eid)
        self.assertEqual(series['trials'][0]['primary_metric']['value'], 7)
        self.assertEqual(series['trials'][0]['duration_seconds'], 1.0)
        self.assertIn(eid, memory['open_ideas'][0]['tested_experiment_ids'])
        trial = next(iter(self.opt.sessions.values()))['study_record']['study'].trials[-1]
        self.assertEqual(trial.value, 7)

    def test_optimizer_tell_failure_preserves_the_committed_evaluation(self):
        ticket, plan = self.anchor(params={'alpha': 1.0})
        self.opt.start_series('researcher-1', self.h.root, plan,
            request({'alpha': {'type': 'float', 'low': 2.0, 'high': 2.0}}))
        candidate = self.opt.next_plan('researcher-1', self.h.root)
        payload = self.h.proposal(ticket, candidate['model']['hyperparameters'])
        payload['proposal']['plan'] = candidate
        payload['manifest'] = self.g.constructor.build(candidate)
        reply = self.g.invoke('G04', payload)
        eid = reply['lead_submission']['experiment_id']
        self.opt.bind(candidate, eid, ticket['work_order']['work_order_id'])
        self.h.commit(reply, 7)
        study = next(iter(self.opt.studies.values()))['study']
        with patch.object(study, 'tell', side_effect=RuntimeError('Injected bookkeeping failure')):
            self.opt.sync()
        self.assertEqual(self.g.store.get(eid)['state'], 'evaluated')
        self.assertEqual(self.g.store.experiment_evaluation(eid)['primary_metric']['value'], 7)
        self.assertFalse(self.opt.active)
        self.assertIn('Injected bookkeeping failure', self.opt.disabled_error)

    def test_expired_authorization_proposes_no_new_work(self):
        _, plan = self.anchor(params={'alpha': 1.0})
        self.opt.start_series('researcher-1', self.h.root, plan,
            request({'alpha': {'type': 'float', 'low': .01, 'high': 3.0}}))
        self.opt.active[(self.h.root, 'researcher-1')]['deadline'] = time.monotonic()-1
        self.assertIsNone(self.opt.next_plan('researcher-1', self.h.root))

    def test_bad_spaces_are_rejected_without_silent_domain_changes(self):
        _, plan = self.anchor('hist-gradient-regressor-v1')
        cases = [
            {'max_iter': {'type': 'int', 'low': 500, 'high': 1499, 'step': 100}},
            {'max_iter': {'type': 'float', 'low': 500, 'high': 1000}},
            {'max_iter': {'type': 'int', 'low': 500, 'high': 1600}},
            {'learning_rate': {'type': 'float', 'low': .01, 'high': .1, 'step': .01, 'log': True}},
            {'learning_rate': {'type': 'float', 'low': .2, 'high': .1}},
            {'missing_parameter': {'type': 'int', 'low': 1, 'high': 3}},
            {'early_stopping': {'type': 'categorical', 'choices': [True, 1]}}]
        for space in cases:
            with self.subTest(space=space), self.assertRaises(AgentError):
                normalize_space(self.g.catalog, plan, request(space))
        value = action('tune', tuning=request({'alpha': {'type': 'float', 'low': .01, 'high': 3.0}}))
        self.g.schemas._check(value, ACTION_SCHEMA, '$')
        validate_action(value, self.g.catalog)
        self.assertEqual(ACTION_SCHEMA, self.g.schemas.schema['$defs']['ResearchAction'])
