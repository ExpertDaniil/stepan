from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from agent.config import Config
from agent.errors import AgentError
from agent.local_search import LocalSearch
from agent.llm.gateway import Gateway
from agent.researcher import Researcher
from agent.context import output_schemas
from tests.graph_helpers import GraphHarness


class FinalRefinementTests(unittest.TestCase):
    def harness(self, **config):
        return GraphHarness(graph_refinement_enabled=True, **config)

    def decide(self, h, branch=None, assessment=None):
        branch = branch or h.root
        cp = h.g.invoke('G01', {'branch_id': branch, 'expected_branch_revision': h.g.branch(branch)['branch_revision'], 'reason': 'manual'})['checkpoint']
        return h.g.invoke('G14', {'checkpoint_id': cp['checkpoint_id'], 'budget': h.g.budget.snapshot(), 'plateau_assessment_id': assessment})['decision']

    def start(self, h, reason='search_time_share_reached', key=None):
        return h.g.invoke('G15', {'source_experiment_id': h.g.store.best(), 'expected_graph_revision': h.g.revision(), 'reason': reason}, key=key)

    def test_defaults_and_time_boundary_use_85_percent_after_publication_reserve(self):
        self.assertEqual(Config().graph_plateau_seconds, 900)
        h = self.harness(time_budget_seconds=10800, max_research_experiments=100, llm_max_calls=96)
        try:
            h.result('researcher-1', {'alpha': 1}, 10)
            cutoff = (10800-h.g.budget.output_reserve())*.85
            with patch.object(h.g, 'elapsed', return_value=int((cutoff-300)*1000)):
                self.assertIsNone(h.g.refinement_due())
            with patch.object(h.g, 'elapsed', return_value=int((cutoff-1.5)*1000)):
                self.assertEqual(h.g.refinement_due(), 'next_research_batch_would_cross_phase_boundary')
            with patch.object(h.g, 'elapsed', return_value=int((cutoff+1)*1000)):
                decision = self.decide(h)
            self.assertEqual(decision['action'], 'REFINE_BEST')
            self.assertEqual(decision['reason'], 'search_time_share_reached')
            self.assertEqual(h.g.phase_limits()['call_limit'], 81)
        finally: h.close()

    def test_legacy_trial_count_does_not_end_search_before_time_boundary(self):
        h = self.harness(max_research_experiments=1, catalog_warmup_experiments=0)
        try:
            for alpha in range(1, 6): h.result('researcher-1', {'alpha': alpha}, 10+alpha)
            decision = self.decide(h)
            self.assertEqual(decision['action'], 'CONTINUE')
            self.assertIsNone(h.g.refinement_due())
            self.assertEqual(h.g.research_slots(h.g.branch(h.root), 1), 1)
            self.start(h)
            self.assertEqual(h.g.research_slots(h.g.branch(h.root), 1), 1)
        finally: h.close()

    def test_phase_transition_does_not_skip_the_other_peers_last_turn(self):
        h = self.harness(max_research_experiments=2, catalog_warmup_experiments=0)
        try:
            first = self.decide(h)
            self.assertEqual(first['parameters']['researcher_id'], 'researcher-1')
            h.result('researcher-1', {'alpha': 1}, 10)
            cutoff=(h.g.config.time_budget_seconds-h.g.budget.output_reserve())*.85
            with patch.object(h.g,'elapsed',return_value=int((cutoff+1)*1000)):
                self.assertEqual(self.decide(h)['action'], 'REFINE_BEST')
            self.start(h)
            final = self.decide(h)
            self.assertEqual(final['parameters']['researcher_id'], 'researcher-2')
        finally: h.close()

    def test_shared_final_seed_remains_explicit_history_after_own_improvement(self):
        h = self.harness()
        try:
            h.result('researcher-1', {'alpha': 1}, 10)
            source = h.g.store.best();self.start(h)
            h.result('researcher-2', {'alpha': 2}, 9)
            self.assertIn(source, h.g.visible_history(h.root, 'researcher-2'))
        finally: h.close()

    def test_api_retry_cannot_consume_reserved_final_attempts(self):
        h = self.harness(llm_max_calls=10)
        try:
            h.result('researcher-1', {'alpha': 1}, 10)
            for _ in range(8):
                number = h.g.budget.begin_llm_call(128)
                h.g.budget.finish_llm_call(number, {'completion_tokens': 1})
            with self.assertRaises(AgentError) as caught: h.g.budget.begin_llm_call(128)
            self.assertEqual(caught.exception.code, 'SEARCH_PHASE_BUDGET_EXHAUSTED')
            self.assertEqual(h.g.budget.calls, 8)
            self.start(h)
            for _ in range(2):
                number = h.g.budget.begin_llm_call(128);h.g.budget.finish_llm_call(number, {'completion_tokens': 1})
            self.assertEqual(h.g.budget.calls, 10)
            with self.assertRaises(AgentError) as caught: h.g.budget.begin_llm_call(128)
            self.assertEqual(caught.exception.code, 'BUDGET_EXHAUSTED')
        finally: h.close()

    def test_output_token_reserve_survives_search_and_is_released_for_refinement(self):
        h = self.harness(llm_max_total_output_tokens=40000, llm_bootstrap_output_tokens=1024)
        try:
            h.result('researcher-1', {'alpha': 1}, 10)
            for _ in range(2):
                number = h.g.budget.begin_llm_call(16000);h.g.budget.finish_llm_call(number, {'completion_tokens': 16000})
            with self.assertRaises(AgentError) as caught: h.g.budget.begin_llm_call(4096)
            self.assertEqual(caught.exception.code, 'SEARCH_PHASE_BUDGET_EXHAUSTED')
            self.assertEqual(h.g.budget.token_snapshot()['output_available_global'], 8000)
            self.start(h)
            number = h.g.budget.begin_llm_call(4096);h.g.budget.finish_llm_call(number, {'completion_tokens': 4096})
            self.assertEqual(h.g.budget.token_snapshot()['output_charged'], 36096)
        finally: h.close()

    def test_refinement_waits_for_an_existing_reasoning_order(self):
        h = self.harness(llm_max_calls=4)
        try:
            h.result('researcher-1', {'alpha': 1}, 10)
            h.g.budget.calls = 2
            first = self.decide(h);ticket = h.issue(first['parameters']['researcher_id'])
            self.assertEqual(self.decide(h)['action'], 'CONTINUE')
            with self.assertRaises(AgentError) as caught: self.start(h)
            self.assertEqual(caught.exception.code, 'PEER_BUSY')
            self.assertEqual(h.g.order(ticket['work_order']['work_order_id'])['state'], 'issued')
            reply, _ = h.submit(ticket, {'alpha': 3});h.commit(reply, 11)
            self.start(h)
            self.assertIsNotNone(h.g.refinement())
        finally: h.close()

    def test_best_in_sealed_ancestor_is_reactivated_without_new_tree_children(self):
        h = self.harness()
        try:
            fork, _, _ = h.make_fork();out = h.g.invoke('G02', fork)
            self.assertEqual(h.g.branch(h.root)['state'], 'sealed')
            source = h.g.store.best();count = h.g.db.execute('SELECT count(*) FROM g_branches').fetchone()[0]
            result = self.start(h)
            self.assertEqual(result['branch']['branch_id'], h.root)
            self.assertEqual(result['branch']['state'], 'active')
            self.assertTrue(all(x['experiment_id'] == source for x in h.g.heads(h.root)))
            self.assertEqual(h.g.db.execute('SELECT count(*) FROM g_branches').fetchone()[0], count)
            self.assertTrue(all(h.g.branch(out[k]['branch_id'])['state'] == 'archived' for k in ['accept_branch','reject_branch']))
            with self.assertRaises(AgentError) as caught: h.g.invoke('G02', fork)
            self.assertEqual(caught.exception.code, 'FINAL_REFINEMENT_ONLY')
        finally: h.close()

    def test_best_in_archived_reject_keeps_its_constraints(self):
        h = self.harness()
        try:
            fork, _, _ = h.make_fork();out = h.g.invoke('G02', fork);reject=out['reject_branch']['branch_id']
            h.g.invoke('G08', {'branch_id': reject, 'expected_branch_revision': 0, 'action': 'resume', 'reason': 'fixture'})
            h.result('researcher-1', {'alpha': 7}, 7, branch=reject)
            constraints = h.g.branch(reject)['constraints']
            h.g.invoke('G08', {'branch_id': reject, 'expected_branch_revision': 1, 'action': 'archive', 'reason': 'fixture'})
            result = self.start(h)
            self.assertEqual(result['branch']['branch_id'], reject)
            self.assertEqual(result['branch']['constraints'], constraints)
            self.assertIn('feature.target_transform', constraints['forbidden_capability_ids'])
        finally: h.close()

    def test_refinement_rejects_stale_graph_and_nonbest_source(self):
        h = self.harness()
        try:
            h.result('researcher-1', {'alpha': 1}, 10)
            _, worse, _, _ = h.result('researcher-2', {'alpha': 2}, 12)
            with self.assertRaises(AgentError) as caught:
                h.g.invoke('G15', {'source_experiment_id': worse['lead_submission']['experiment_id'], 'expected_graph_revision': h.g.revision(), 'reason': 'fixture'})
            self.assertEqual(caught.exception.code, 'BEST_CHANGED')
            with self.assertRaises(AgentError) as caught:
                h.g.invoke('G15', {'source_experiment_id': h.g.store.best(), 'expected_graph_revision': h.g.revision()-1, 'reason': 'fixture'})
            self.assertEqual(caught.exception.code, 'GRAPH_REVISION_CONFLICT')
            self.assertIsNone(h.g.refinement())
        finally: h.close()

    def test_idempotent_refinement_does_not_reset_newer_progress(self):
        h = self.harness()
        try:
            h.result('researcher-1', {'alpha': 1}, 10)
            payload = {'source_experiment_id': h.g.store.best(), 'expected_graph_revision': h.g.revision(), 'reason': 'fixture'}
            result = h.g.invoke('G15', payload, key='final-phase')
            h.result('researcher-1', {'alpha': 2}, 9)
            best = h.g.store.best();head = h.g.heads(h.root)
            replay = h.g.invoke('G15', payload, key='final-phase')
            self.assertEqual(replay, result)
            self.assertEqual(h.g.heads(h.root), head)
            self.assertEqual(h.g.store.best(), best)
        finally: h.close()

    def test_final_phase_continues_on_plateau_and_uses_only_exploit(self):
        h = self.harness()
        try:
            h.result('researcher-1', {'alpha': 1}, 10);self.start(h)
            for alpha in range(2, 6): h.result('researcher-1', {'alpha': alpha}, 11)
            for alpha in range(6, 10): h.result('researcher-2', {'alpha': alpha}, 11)
            report = h.g.invoke('G07', {'branch_id': h.root, 'expected_branch_revision': h.g.branch(h.root)['branch_revision']})['assessment']
            self.assertEqual(report['status'], 'plateau')
            self.assertEqual(report['recommended_action'], 'continue')
            self.assertEqual(self.decide(h, assessment=report['assessment_id'])['action'], 'CONTINUE')
            for peer in ['researcher-1','researcher-2']:
                ticket = h.issue(peer)
                context = h.g.artifacts.read_json(ticket['work_order']['context_id'])
                user = json.loads(context['messages'][1]['content'])
                self.assertEqual(user['research_phase'], 'refine')
                self.assertEqual(user['research_strategy'], 'exploit_incumbent')
                self.assertEqual(context['metadata']['recipe_ids'], ['linear-regressor-v1'])
                self.assertIn('Small real improvements', user['assignment'])
                h.g.abandon(ticket['work_order']['work_order_id'], 'fixture complete')
        finally: h.close()

    def test_tiny_gain_is_kept_and_worse_trials_cannot_replace_it(self):
        h = self.harness()
        try:
            h.result('researcher-1', {'alpha': 1}, 10);self.start(h)
            _, tiny, _, _ = h.result('researcher-1', {'alpha': 2}, 9.9999)
            h.result('researcher-2', {'alpha': 3}, 11)
            self.assertEqual(h.g.store.best(), tiny['lead_submission']['experiment_id'])
        finally: h.close()

    def test_refinement_rejects_switch_to_another_family(self):
        h = self.harness()
        try:
            h.result('researcher-1', {'alpha': 1}, 10);self.start(h)
            ticket = h.issue();proposal = h.proposal(ticket, {}, recipe='hist-gradient-regressor-v1')
            with self.assertRaises(AgentError) as caught: h.g.invoke('G04', proposal)
            self.assertEqual(caught.exception.code, 'REFINEMENT_FAMILY_CHANGED')
        finally: h.close()

    def test_partial_llm_tuning_keeps_the_measured_feature_parameters(self):
        h = self.harness()
        try:
            h.result('researcher-1', {'alpha': 1, 'target_transform': 'shifted_log1p'}, 10)
            self.start(h);h.g.context.round_no=1
            ticket=h.issue('researcher-2')
            gateway=Gateway(h.g.config,h.g.budget,h.g.artifacts,h.g.bus,h.g.log,output_schemas(h.g.bus))
            researcher=Researcher(h.g.bus,gateway,h.g.artifacts,h.g.spec,h.g.catalog,h.g.config,h.g.log)
            researcher.validation_id=h.g.validation['validation_id']
            plan=researcher.propose(ticket['researcher_request']['payload'])['plan']
            self.assertEqual(plan['model']['hyperparameters']['target_transform'],'shifted_log1p')
        finally: h.close()

    def test_explicit_feature_switch_is_rejected_during_final_tuning(self):
        h = self.harness()
        try:
            h.result('researcher-1', {'alpha': 1, 'target_transform': 'shifted_log1p'}, 10)
            self.start(h);ticket=h.issue('researcher-2')
            with self.assertRaises(AgentError) as caught:h.submit(ticket, {'alpha': 2})
            self.assertEqual(caught.exception.code,'REFINEMENT_COMPONENT_CHANGED')
        finally: h.close()

    def test_local_fallback_stays_on_best_family_features_and_drops_old_series(self):
        h = self.harness()
        try:
            h.controller.graph=h.g
            h.controller.resources=lambda:{'cpu_threads':1,'memory_limit_mb':1024,'timeout_seconds':30}
            local=LocalSearch(h.controller);h.controller.local_search=local
            h.result('researcher-1', {'alpha': 1, 'target_transform': 'shifted_log1p'}, 10)
            base=h.g.store.get(h.g.store.best())['plan']
            local.start_series('researcher-1',h.root,base)
            self.start(h);self.assertEqual(local.series,{})
            for _ in range(16):
                recipe,params=local.next_choice('researcher-1',h.root,base)
                self.assertEqual(recipe,base['recipe_id'])
                self.assertEqual(params['target_transform'],'shifted_log1p')
                self.assertNotEqual(params,base['model']['hyperparameters'])
        finally: h.close()

    def test_refinement_record_is_exported_and_original_caps_remain_global(self):
        h = self.harness()
        try:
            h.result('researcher-1', {'alpha': 1}, 10);self.start(h)
            output = h.g.invoke('G11', {'task_spec_id': h.g.spec['task_spec_id'], 'validation_id': h.g.validation['validation_id'], 'selection_policy_id': 'same-protocol-primary-v1', 'scope': 'final'})
            graph = h.g.artifacts.read_json(output['graph_artifact']['artifact_id'])
            self.assertEqual(graph['refinement']['source_experiment_id'], h.g.store.best())
            self.assertIsNone(h.g.budget.phase_call_limit)
            self.assertIsNone(h.g.budget.phase_output_limit)
            h.g.schemas.validate(graph, 'GGraphExport')
        finally: h.close()
