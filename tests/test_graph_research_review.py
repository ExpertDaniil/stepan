from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest

from agent.errors import AgentError
from agent.graph.controller import research_batch, transfer_ticket, lineage_for
from agent.researcher import Researcher
from scripts.configure import configure
from tests.graph_helpers import GraphHarness


class GraphResearchReviewTests(unittest.TestCase):
    def context_user(self, h, ticket):
        context = h.g.artifacts.read_json(ticket['work_order']['context_id'])
        return json.loads(context['messages'][1]['content'])

    def decide(self, h, branch=None):
        branch = branch or h.root
        current = h.g.branch(branch)
        cp = h.g.invoke('G01', {'branch_id': branch, 'expected_branch_revision': current['branch_revision'], 'reason': 'manual'})['checkpoint']
        return h.g.invoke('G14', {'checkpoint_id': cp['checkpoint_id'], 'budget': h.g.budget.snapshot(), 'plateau_assessment_id': None})['decision']

    def test_large_profile_preserves_measured_incumbent_and_unsuccessful_attempt(self):
        h = GraphHarness()
        try:
            _, best, _, _ = h.result('researcher-1', {'alpha': 1}, 10)
            _, worse, _, _ = h.result('researcher-1', {'alpha': 2}, 12)
            h.g.context.profile['test_dtypes'] = {f'x{i}': 'wide redundant type description '*100 for i in range(40)}
            ticket = h.issue()
            user = self.context_user(h, ticket)
            self.assertEqual(user['base_result']['experiment_id'], best['lead_submission']['experiment_id'])
            self.assertEqual(user['base_result']['metric']['value'], 10)
            self.assertEqual(user['base_result']['hyperparameters']['alpha'], 1)
            self.assertIn(worse['lead_submission']['experiment_id'], [f['source_id'] for f in user['facts']])
            self.assertNotIn('test_dtypes', user['profile_summary']['data'])
            self.assertEqual(user['research_strategy'], 'exploit_incumbent')
            self.assertIn('exploit_incumbent', user['assignment'])
            h.g.schemas.validate(user, 'GContextUser')
        finally:
            h.close()

    def test_exploration_gets_one_least_observed_family_and_roles_rotate(self):
        h = GraphHarness()
        try:
            h.result('researcher-1', {'alpha': 1}, 10)
            h.result('researcher-2', {'alpha': 2}, 12)
            h.controller.context.round_no = 1
            ticket = h.issue('researcher-1')
            user = self.context_user(h, ticket)
            self.assertEqual(user['research_strategy'], 'explore_alternative')
            self.assertEqual(len(user['profile_summary']['recipes']), 1)
            self.assertNotEqual(user['profile_summary']['recipes'][0]['family'], 'linear')
            self.assertIn('explore_alternative', user['assignment'])
            other = self.context_user(h, h.issue('researcher-2'))
            self.assertEqual(other['research_strategy'], 'exploit_incumbent')
            self.assertEqual(other['profile_summary']['recipes'][0]['family'], 'linear')
        finally:
            h.close()

    def test_failure_details_reach_next_context_and_other_peers_do_not_leak(self):
        h = GraphHarness()
        try:
            h.result('researcher-1', {'alpha': 1}, 10)
            _, other, _, _ = h.result('researcher-2', {'alpha': 2}, 12)
            ticket = h.issue();reply, _ = h.submit(ticket, {'alpha': 3})
            event = h.train_event(reply, state='failed')
            event['error'] = AgentError('SYNTHETIC_FAILURE', 'Synthetic estimator rejected this configuration').as_dict()
            h.g.store.apply_event(event, 'train')
            user = self.context_user(h, h.issue())
            facts = {f['source_id']: f for f in user['facts']}
            failed = reply['lead_submission']['experiment_id']
            self.assertEqual(facts[failed]['kind'], 'failure')
            self.assertIn('Synthetic estimator', json.loads(facts[failed]['summary'])['error'])
            self.assertNotIn(other['lead_submission']['experiment_id'], facts)
        finally:
            h.close()

    def test_reject_context_recovers_own_prefork_failure_but_excludes_later_parent_results(self):
        h = GraphHarness()
        try:
            h.result('researcher-1', {'alpha': 4}, 20)
            fork, _, _ = h.make_fork()
            _, later, _, _ = h.result('researcher-1', {'alpha': 5}, 30)
            out = h.g.invoke('G02', fork)
            reject = out['reject_branch']['branch_id']
            h.g.invoke('G08', {'branch_id': reject, 'expected_branch_revision': 0, 'action': 'resume', 'reason': 'counterfactual'})
            user = self.context_user(h, h.issue(branch=reject))
            facts = [json.loads(f['summary']) for f in user['facts']]
            self.assertTrue(any(f['hyperparameters']['alpha'] == 4 for f in facts))
            self.assertNotIn(later['lead_submission']['experiment_id'], [f['experiment_id'] for f in facts])
            self.assertTrue(all(f['hyperparameters'].get('target_transform') != 'shifted_log1p' for f in facts))
        finally:
            h.close()

    def test_catalogue_domains_match_accept_and_reject_constraints(self):
        h = GraphHarness()
        try:
            fork, _, _ = h.make_fork();out = h.g.invoke('G02', fork)
            for kind, transform in [('accept_branch', 'shifted_log1p'), ('reject_branch', 'identity')]:
                branch = out[kind]['branch_id']
                if kind == 'reject_branch':
                    h.g.invoke('G08', {'branch_id': branch, 'expected_branch_revision': 0, 'action': 'resume', 'reason': 'counterfactual'})
                ticket = h.issue(branch=branch)
                for recipe in self.context_user(h, ticket)['profile_summary']['recipes']:
                    self.assertEqual(recipe['parameters']['target_transform']['enum'], [transform])
                    self.assertEqual(recipe['parameters']['target_transform']['default'], transform)
                h.g.abandon(ticket['work_order']['work_order_id'], 'test assignment complete')
            self.assertEqual(h.g.catalog.recipes['linear-regressor-v1']['parameters']['target_transform']['enum'], ['identity', 'shifted_log1p'])
        finally:
            h.close()

    def test_new_branch_and_resumed_branch_do_not_inherit_old_wall_time_as_plateau(self):
        h = GraphHarness()
        try:
            fork, _, _ = h.make_fork()
            h.g.policy['plateau']['min_elapsed_seconds'] = 300
            h.g.budget.started -= 1000
            out = h.g.invoke('G02', fork);accept = out['accept_branch']['branch_id']
            h.result('researcher-1', {'alpha': 7, 'target_transform': 'shifted_log1p'}, 20, branch=accept)
            for alpha in [4, 5, 6]:
                h.result('researcher-1', {'alpha': alpha, 'target_transform': 'shifted_log1p'}, 20+alpha, branch=accept)
            p = {'branch_id': accept, 'expected_branch_revision': 0}
            first = h.g.invoke('G07', p)['assessment']
            self.assertEqual(first['status'], 'searching')
            self.assertLess(first['elapsed_seconds'], 300)
            h.g.invoke('G08', {**p, 'action': 'pause', 'reason': 'temporary scheduling pause'})
            h.g.budget.started -= 1000
            h.g.invoke('G08', {'branch_id': accept, 'expected_branch_revision': 1, 'action': 'resume', 'reason': 'resume'})
            second = h.g.invoke('G07', {'branch_id': accept, 'expected_branch_revision': 2})['assessment']
            self.assertEqual(second['status'], 'searching')
            self.assertLess(second['elapsed_seconds'], 300)
        finally:
            h.close()

    def test_expired_assignments_release_peers_before_scheduling_and_are_journaled(self):
        h = GraphHarness()
        try:
            tickets = [h.issue(peer) for peer in ['researcher-1', 'researcher-2']]
            with h.g.store.transaction():
                h.g.db.execute("UPDATE g_work_orders SET expires_elapsed_ms=1 WHERE state='issued'")
            decision = self.decide(h)
            self.assertEqual(decision['action'], 'CONTINUE')
            for t in tickets:
                self.assertEqual(h.g.order(t['work_order']['work_order_id'])['state'], 'expired')
            events = [json.loads(r[0]) for r in h.g.db.execute("SELECT record_json FROM g_events WHERE json_extract(record_json,'$.record.state')='expired'")]
            self.assertEqual(len(events), 2)
        finally:
            h.close()

    def test_last_llm_call_is_not_assigned_to_two_peers(self):
        h = GraphHarness(llm_max_calls=1)
        try:
            decision = self.decide(h)
            self.assertEqual(decision['action'], 'CONTINUE')
            h.issue(decision['parameters']['researcher_id'])
            second = self.decide(h)
            self.assertEqual(second['action'], 'FINALIZE')
            self.assertEqual(second['parameters']['reason'], 'llm_call_limit')
        finally:
            h.close()

    def test_last_two_trials_are_split_between_accept_and_reject_even_after_a_failure(self):
        h = GraphHarness(max_research_experiments=6)
        try:
            fork, _, _ = h.make_fork();out = h.g.invoke('G02', fork);accept = out['accept_branch']
            self.assertEqual(h.g.research_slots(accept, 2), 1)
            ticket = h.issue(branch=accept['branch_id'])
            reply, _ = h.submit(ticket, {'alpha': 4, 'target_transform': 'shifted_log1p'})
            event = h.train_event(reply, state='failed')
            event['error'] = AgentError('SYNTHETIC_FAILURE', 'A failed attempt still consumes the experiment ceiling').as_dict()
            h.g.store.apply_event(event, 'train')
            decision = self.decide(h, accept['branch_id'])
            self.assertEqual(decision['action'], 'BACKTRACK')
            self.assertEqual(decision['parameters']['target_branch_id'], out['reject_branch']['branch_id'])
        finally:
            h.close()

    def test_scheduler_does_not_fork_when_only_one_training_trial_remains(self):
        h = GraphHarness(max_research_experiments=5)
        try:
            h.make_fork()
            self.assertEqual(h.g.attempted_research_count(), 4)
            self.assertEqual(self.decide(h)['action'], 'CONTINUE')
            self.assertEqual(h.g.db.execute('SELECT count(*) FROM g_branches').fetchone()[0], 1)
        finally:
            h.close()

    def test_output_quota_exhaustion_is_a_clean_search_stop(self):
        h = GraphHarness(llm_max_total_output_tokens=1024)
        try:
            decision = self.decide(h)
            self.assertEqual(decision['action'], 'FINALIZE')
            self.assertEqual(decision['parameters']['reason'], 'llm_output_budget')
            self.assertEqual(h.g.budget.calls, 0)
        finally:
            h.close()

    def test_proposal_rejection_is_remembered_without_inventing_an_evaluation(self):
        h = GraphHarness()
        try:
            h.result('researcher-1', {'alpha': 1}, 10)
            ticket = h.issue()
            h.g.abandon(ticket['work_order']['work_order_id'], 'Unknown hyperparameter: synthetic_invalid')
            user = self.context_user(h, h.issue())
            fact = next(f for f in user['facts'] if f['source_id'] == ticket['work_order']['work_order_id'])
            self.assertEqual(fact['kind'], 'failure')
            self.assertEqual(json.loads(fact['summary'])['stage'], 'proposal_rejected')
            self.assertNotIn('metric', json.loads(fact['summary']))
        finally:
            h.close()

    def test_acceptance_applies_confirmed_feature_without_llm_or_retraining_cached_results(self):
        h = GraphHarness()
        try:
            fork, _, _ = h.make_fork();out = h.g.invoke('G02', fork);accept = out['accept_branch']['branch_id']
            h.controller.graph = h.g
            h.controller.call = lambda cid, payload, fn: h.g.bus.call(cid, payload, fn)['data']
            before = h.g.db.execute('SELECT count(*) FROM experiments').fetchone()[0]
            for _ in range(2):
                decision = self.decide(h, accept)
                self.assertEqual(decision['action'], 'TRANSFER_COMPONENT')
                plan, ticket = transfer_ticket(h.controller, decision)
                self.assertEqual(plan['model']['hyperparameters']['target_transform'], 'shifted_log1p')
                payload = {'work_order_id': ticket['work_order']['work_order_id'],
                           'proposal': {'proposal_id': plan['plan_id'], 'researcher_id': plan['researcher_id'], 'plan': plan, 'decision': 'propose', 'stop_reason': None},
                           'manifest': h.g.constructor.build(plan), 'lineage': lineage_for(ticket['work_order'])}
                reply = h.g.invoke('G04', payload)
                self.assertEqual(reply['disposition'], 'duplicate')
                h.commit(reply, 999)
            self.assertEqual(h.g.db.execute('SELECT count(*) FROM experiments').fetchone()[0], before)
            self.assertEqual(h.g.budget.calls, 0)
            self.assertTrue(all(h.g.head_satisfies(head, out['accept_branch']['constraints']) for head in h.g.heads(accept)))
            self.assertEqual(self.decide(h, accept)['action'], 'CONTINUE')
        finally:
            h.close()

    def test_preidea_reference_does_not_block_valid_branch_head_or_hide_its_progress(self):
        h = GraphHarness()
        try:
            fork, _, _ = h.make_fork();out = h.g.invoke('G02', fork);accept = out['accept_branch']['branch_id']
            global_best = h.g.store.best()
            latest = None
            for alpha, score in [(4, 20), (5, 19), (6, 18), (7, 17)]:
                _, latest, _, result = h.result('researcher-1', {'alpha': alpha, 'target_transform': 'shifted_log1p'}, score, branch=accept)
                self.assertEqual(result['head_results'][0]['disposition'], 'advanced')
            self.assertEqual(h.g.heads(accept)[0]['experiment_id'], latest['lead_submission']['experiment_id'])
            self.assertEqual(h.g.store.best(), global_best)
            assessment = h.g.invoke('G07', {'branch_id': accept, 'expected_branch_revision': 0})['assessment']
            self.assertEqual(assessment['status'], 'searching')
            self.assertEqual(assessment['utility_gain'], 3)
        finally:
            h.close()

    def test_constraint_override_is_disclosed_in_effective_plan_and_journal(self):
        h = GraphHarness()
        try:
            fork, _, _ = h.make_fork();out = h.g.invoke('G02', fork)
            ticket = h.issue(branch=out['accept_branch']['branch_id'])
            data = deepcopy(h.g.schemas.operations['C10']['response_example']['data'])
            data['parsed_payload'] = {'recipe_id': 'linear-regressor-v1', 'hyperparameters': {'alpha': 4, 'target_transform': 'identity'},
                                      'hypothesis': 'Test the raw target.', 'expected_effect': 'Compare on the same validation.'}
            gateway = SimpleNamespace(complete=lambda _: data)
            researcher = Researcher(h.g.bus, gateway, h.g.artifacts, h.g.spec, h.g.catalog, h.g.config, h.g.log)
            researcher.validation_id = h.g.validation['validation_id']
            plan = researcher.propose(ticket['researcher_request']['payload'])['plan']
            self.assertEqual(plan['model']['hyperparameters']['target_transform'], 'shifted_log1p')
            self.assertIn('Effective branch constraints override', plan['hypothesis'])
            self.assertIn('shifted_log1p', plan['hypothesis'])
            events = [json.loads(line) for line in (h.h.run/'events.jsonl').read_text().splitlines()]
            event = next(e for e in events if e['event'] == 'research.constraints_applied')
            self.assertEqual(event['changes']['target_transform'], {'proposed': 'identity', 'effective': 'shifted_log1p'})
        finally:
            h.close()

    def test_extended_profile_preserves_output_cap_credentials_and_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'.env'
            original = 'OPENAI_API_KEY=synthetic-placeholder\nWORK_DIR=custom\nTIME_BUDGET_SECONDS=10800\nLLM_MAX_TOTAL_OUTPUT_TOKENS=64000\nMAX_RESEARCH_EXPERIMENTS=12\n'
            path.write_text(original)
            configure(path, profile='extended')
            self.assertEqual(path.read_text(), original)
            configure(path, apply=True, profile='extended')
            text = path.read_text()
            for line in ['OPENAI_API_KEY=synthetic-placeholder', 'WORK_DIR=custom', 'TIME_BUDGET_SECONDS=10800', 'LLM_MAX_TOTAL_OUTPUT_TOKENS=64000', 'MAX_RESEARCH_EXPERIMENTS=100', 'LLM_MAX_CALLS=96']:
                self.assertIn(line, text)


class ResearchBatchSchedulingTests(unittest.TestCase):
    def test_fast_proposal_trains_while_other_peer_is_still_thinking_on_dispatcher(self):
        started, release_slow = threading.Event(), threading.Event()
        dispatcher = threading.get_ident()
        trained = []
        def propose(payload):
            peer = payload['researcher_id']
            if peer == 'researcher-2':
                started.set()
                if not release_slow.wait(3):
                    raise AssertionError('Fast proposal was needlessly blocked by the slow peer')
            else:
                if not started.wait(3):
                    raise AssertionError('Second researcher was not started concurrently')
            return {'decision': 'propose', 'plan': {'researcher_id': peer}}
        def train(plans, tickets):
            self.assertEqual(threading.get_ident(), dispatcher)
            for plan, ticket in zip(plans, tickets):
                self.assertEqual(plan['researcher_id'], ticket['work_order']['researcher_id'])
                trained.append(plan['researcher_id'])
                if plan['researcher_id'] == 'researcher-1':
                    release_slow.set()
            return len(plans)
        graph = SimpleNamespace(policy={'max_train_workers': 1}, metadata=lambda _: ['researcher-1', 'researcher-2'])
        controller = SimpleNamespace(graph=graph, researcher=SimpleNamespace(propose=propose), run_training=train, degraded=False,
                                     call=lambda _cid, payload, fn: fn(payload), log=SimpleNamespace(exception=lambda *a, **k: self.fail(str(a))))
        tickets = [{'work_order': {'work_order_id': peer, 'researcher_id': peer}, 'researcher_request': {'payload': {'researcher_id': peer}}} for peer in ['researcher-1', 'researcher-2']]
        accepted, had_plans = research_batch(controller, [], [], tickets)
        self.assertEqual(accepted, 2);self.assertTrue(had_plans)
        self.assertEqual(trained, ['researcher-1', 'researcher-2'])
        self.assertFalse(controller.degraded)
