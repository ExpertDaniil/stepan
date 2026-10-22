"""Regression scenarios for the run review. Not executed during the requested audit."""
import json
import time
import unittest
import uuid
from unittest.mock import patch

from agent.context import output_schemas
from agent.graph.controller import apply_switch, issue
from agent.llm.gateway import Gateway
from agent.local_search import LocalSearch
from agent.researcher import Researcher
from tests.graph_helpers import GraphHarness


class SearchConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness()
        self.c, self.g = self.h.controller, self.h.g
        self.c.graph = self.g
        self.c.resources = lambda: {'cpu_threads': 1, 'memory_limit_mb': 1024, 'timeout_seconds': 30}

    def tearDown(self):
        self.h.close()

    def test_partial_search_proposal_preserves_measured_feature_switches(self):
        self.h.result('researcher-1', {'alpha': 7, 'target_transform': 'shifted_log1p', 'numeric_interactions': True}, 10)
        ticket = self.h.issue()
        gateway = Gateway(self.g.config, self.g.budget, self.g.artifacts, self.g.bus, self.g.log, output_schemas(self.g.bus))
        researcher = Researcher(self.g.bus, gateway, self.g.artifacts, self.g.spec, self.g.catalog, self.g.config, self.g.log)
        researcher.validation_id = self.g.validation['validation_id']
        plan = researcher.propose(ticket['researcher_request']['payload'])['plan']
        self.assertEqual(plan['model']['hyperparameters']['target_transform'], 'shifted_log1p')
        self.assertTrue(plan['model']['hyperparameters']['numeric_interactions'])

    def test_local_variants_cover_several_axes_before_second_iteration_neighbour(self):
        local = LocalSearch(self.c)
        recipe = 'hist-gradient-regressor-v1'
        params = self.g.catalog.normalize(recipe, {})
        variants = list(local.variants(recipe, params))
        changed = [next(k for k in params if params[k] != candidate[k]) for candidate in variants[:5]]
        self.assertEqual(len(set(changed)), 5)
        self.assertFalse(any(candidate['max_iter'] in {299, 301} for candidate in variants))

    def test_final_variants_keep_features_and_allow_enum_depth(self):
        local = LocalSearch(self.c)
        recipe = 'random-forest-regressor-v1'
        params = self.g.catalog.normalize(recipe, {'target_transform': 'shifted_log1p'})
        variants = list(local.variants(recipe, params, numeric_only=True))
        self.assertTrue(any(v['max_depth'] != params['max_depth'] for v in variants))
        self.assertTrue(all(v['target_transform'] == 'shifted_log1p' and not v['numeric_interactions'] for v in variants))

    def test_local_series_rebases_on_measured_family_best(self):
        _, best, _, _ = self.h.result('researcher-1', {'alpha': 2}, 10)
        _, _, bad, _ = self.h.result('researcher-1', {'alpha': 8}, 12)
        local = LocalSearch(self.c)
        local.start_series('researcher-1', self.h.root, bad['proposal']['plan'])
        params = local.measured_base('researcher-1', self.h.root, 'linear-regressor-v1', {'alpha': 8})
        self.assertEqual(params, self.g.store.get(best['lead_submission']['experiment_id'])['plan']['model']['hyperparameters'])

    def test_old_model_proposal_survives_a_long_run_of_local_facts(self):
        _, old, _, _ = self.h.result('researcher-1', {'alpha': 2}, 12)
        self.h.result('researcher-1', {'alpha': 1}, 10)
        for alpha in range(3, 18):
            ticket = self.h.issue()
            payload = self.h.proposal(ticket, {'alpha': alpha})
            payload['proposal']['proposal_id'] = 'trusted-' + str(alpha)
            reply = self.g.invoke('G04', payload)
            self.h.commit(reply, 15)
        ticket = self.h.issue()
        user = json.loads(self.g.artifacts.read_json(ticket['work_order']['context_id'])['messages'][1]['content'])
        records = user['profile_summary']['measured_search']['entries']
        self.assertIn(old['lead_submission']['experiment_id'], [r['experiment_id'] for r in records])

    def test_four_trials_by_only_one_peer_are_not_a_team_plateau(self):
        self.h.result('researcher-1', {'alpha': 1}, 10)
        for alpha in range(2, 6):
            self.h.result('researcher-1', {'alpha': alpha}, 12)
        assess = lambda: self.g.invoke('G07', {'branch_id': self.h.root, 'expected_branch_revision': 0})['assessment']
        self.assertNotEqual(assess()['status'], 'plateau')
        for alpha in range(6, 10):
            self.h.result('researcher-2', {'alpha': alpha}, 12)
        self.assertEqual(assess()['status'], 'plateau')

    def test_assignment_ttl_covers_queue_beyond_response_timeout(self):
        self.c.budget.deadline = time.monotonic() + 2000
        ticket = issue(self.c, self.h.root, 'researcher-1')
        order = ticket['work_order']
        response_limit = self.c.config.llm_operation_timeout_seconds + 31
        with patch.object(self.g, 'elapsed', return_value=order['issued_elapsed_ms']+response_limit*1000):
            self.g.check_order(order['work_order_id'])

    def test_switching_at_the_live_limit_keeps_the_previous_branch_paused(self):
        fork, _, _ = self.h.make_fork()
        out = self.g.invoke('G02', fork)
        self.g.policy['max_live_branches'] = 2
        accept, reject = out['accept_branch']['branch_id'], out['reject_branch']['branch_id']
        decision = {'decision_id': 'fixture-switch', 'reason': 'fixture', 'parameters': {'target_branch_id': reject}}
        apply_switch(self.c, accept, decision)
        self.assertEqual(self.g.branch(accept)['state'], 'paused')
        self.assertEqual(self.g.branch(reject)['state'], 'active')

    def test_nested_reject_uses_the_accepted_ancestor_donor(self):
        fork, _, idea_a = self.h.make_fork()
        first = self.g.invoke('G02', fork)
        accept = first['accept_branch']['branch_id']
        before, source, payload, _ = self.h.result('researcher-1',
            {'alpha': 1, 'target_transform': 'shifted_log1p', 'numeric_interactions': True}, 7, branch=accept)
        source_id = source['lead_submission']['experiment_id']
        _, pilot, _, _ = self.h.result('researcher-2',
            {'alpha': 2, 'target_transform': 'shifted_log1p', 'numeric_interactions': True}, 8,
            branch=accept, action='transfer', donors=[source_id])
        component = next(c for c in payload['manifest']['core']['components'] if c['capability_id'] == 'feature.numeric_interactions')
        patch_body = {'patch_version': 'transfer-1.0', 'idea_id': 'idea-'+uuid.uuid4().hex,
            'supported_recipe_ids': list(self.g.catalog.recipes), 'required_base_capability_ids': [],
            'operations': [{'operation': 'add_component', 'target_component_id': None, 'component': self.g.constructor.external_component(component)}],
            'expected_added_capability_ids': ['feature.numeric_interactions'], 'expected_removed_capability_ids': []}
        artifact = self.g.files.put_json(patch_body, 'transfer_patch')
        evidence = {'title': 'Nested B', 'hypothesis': 'Synthetic transfer', 'source_experiment_id': source_id,
            'capability_ids': ['feature.numeric_interactions'], 'fork_checkpoint_id': before['work_order']['base_team_checkpoint_id'],
            'baseline_experiment_ids': [before['work_order']['base_experiment_id']], 'candidate_experiment_ids': [source_id],
            'transfer_pilot_experiment_ids': [pilot['lead_submission']['experiment_id']], 'transfer_patch_artifact_id': artifact['artifact_id']}
        idea_b = self.g.invoke('G06', evidence)['idea']
        second = self.g.invoke('G02', {'parent_branch_id': accept, 'expected_branch_revision': 0,
            'checkpoint_id': evidence['fork_checkpoint_id'], 'idea_id': idea_b['idea_id']})
        reject = second['reject_branch']
        self.g.invoke('G08', {'branch_id': reject['branch_id'], 'expected_branch_revision': 0, 'action': 'resume', 'reason': 'fixture'})
        donors = self.g.required_donors(reject)
        self.assertEqual(donors, [idea_a['source_experiment_id']])
        ticket = issue(self.c, reject['branch_id'], 'researcher-1', donors=donors)
        self.assertNotIn(source_id, ticket['work_order']['allowed_donor_experiment_ids'])

    def test_phase_time_cap_blocks_new_fit_and_is_released_for_refinement(self):
        ticket = self.h.issue()
        plan = self.h.proposal(ticket)['proposal']['plan']
        self.g.budget.deadline = time.monotonic()+100
        self.g.budget.set_phase_caps(deadline=time.monotonic()+.5)
        self.assertFalse(self.g.budget.can_train(plan)['allowed'])
        self.g.budget.set_phase_caps()
        self.assertTrue(self.g.budget.can_train(plan)['allowed'])


if __name__ == '__main__':
    unittest.main()
