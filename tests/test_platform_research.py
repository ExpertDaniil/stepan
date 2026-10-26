"""Offline budget, transcript and rolling-dispatch regressions. No provider calls."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
import os
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.budget import Budget
from agent.config import Config
from agent.context import output_schemas
from agent.errors import AgentError
from agent.graph.controller import research_batch
from agent.llm.gateway import Gateway
from agent.research_feedback import FeedbackMailbox
from tests.graph_helpers import GraphHarness
from tests.test_adaptive_research import action


class PlatformBudgetTests(unittest.TestCase):
    def test_platform_factory_has_no_development_or_phase_numeric_caps(self):
        with patch.dict(os.environ, {}, clear=True):
            config = Config.from_env(profile='platform')
            development = Config.from_env(profile='development')
        self.assertIsNone(config.llm_max_calls)
        self.assertIsNone(config.llm_max_total_output_tokens)
        self.assertFalse(config.graph_refinement_enabled)
        self.assertEqual(development.llm_max_calls, 32)
        budget = Budget(config, time.monotonic(), SimpleNamespace())
        for _ in range(40):
            ident, slots = budget.reserve_dialogue(16384)
            number = budget.begin_llm_call(16384, ident)
            budget.finish_llm_call(number, {'completion_tokens': 8000})
            budget.release_dialogue(ident)
        self.assertEqual(budget.calls, 40)
        self.assertGreater(budget.output_tokens_charged, 262144)
        self.assertIsNone(budget.snapshot()['llm_calls_remaining'])
        self.assertIsNone(budget.llm_unavailable_reason())

    def test_concurrent_finite_reservations_cannot_spend_another_final(self):
        cfg = Config(llm_max_calls=4, llm_max_total_output_tokens=4000)
        budget = Budget(cfg, time.monotonic(), SimpleNamespace())
        barrier = threading.Barrier(2)
        def dialogue():
            ident, slots = budget.reserve_dialogue(1000)
            barrier.wait(3)
            numbers = []
            for _ in range(slots):
                self.assertTrue(budget.prepare_dialogue_turn(ident, 1000, allow_tools=True))
                number = budget.begin_llm_call(1000, ident)
                budget.finish_llm_call(number, {'completion_tokens': 1000})
                numbers.append(number)
            budget.release_dialogue(ident)
            return numbers
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(dialogue) for _ in range(2)]
            calls = [number for future in futures for number in future.result()]
        self.assertEqual(sorted(calls), [1, 2, 3, 4])
        self.assertEqual(budget.token_snapshot()['output_reserved'], 0)

    def test_reasoning_control_cannot_be_enabled_in_compatible_mode(self):
        with patch.dict(os.environ, {'LLM_REASONING_EFFORT': 'xhigh'}, clear=True):
            with self.assertRaises(AgentError):
                Config.from_env()


class ActualRequestTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800,
                              llm_max_calls=None, llm_max_total_output_tokens=None)

    def tearDown(self):
        self.h.close()

    def complete(self, transport, *, protocol='compatible', turns=10):
        g = self.h.g
        config = replace(g.config, llm_reasoning_protocol=protocol)
        gateway = Gateway(config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), transport)
        gateway.feedback = FeedbackMailbox(g.artifacts)
        transport.gateway = gateway
        class Analysis:
            def analyze(service, args, context=None):
                art = g.artifacts.write_json('analysis/test-'+args['columns'][0]+'.json', args, 'research_analysis')
                return {'artifact_id': art['artifact_id'], 'report': {'scope': 'fit_only', 'columns': args['columns']}}
        gateway.analysis = Analysis()
        ticket = self.h.issue()
        transport.branch = ticket['work_order']['branch_id']
        result = gateway.complete({'purpose': 'research_proposal', 'context_artifact_id': ticket['work_order']['context_id'],
            'response_schema_id': 'ResearchAction-v1', 'tool_schema_ids': ['analyze_data'], 'max_output_tokens': 1024,
            'deadline_seconds': 300, 'attempt_limit': turns})
        return result

    @staticmethod
    def tool(number):
        return {'role': 'assistant', 'content': 'Visible observation '+str(number), 'reasoning_content': 'provider field',
            'tool_calls': [{'id': 'inspect-'+str(number), 'type': 'function', 'function': {'name': 'analyze_data',
                'arguments': json.dumps({'kind': 'summary', 'columns': ['column_'+str(number)], 'group_by': None, 'include_target': False})}}]}

    def test_eight_tool_turns_leave_a_final_action_and_preserve_content_and_call_ids(self):
        outer = self
        class Transport:
            def __init__(self):
                self.requests = []
            def complete(self, request, timeout, context):
                self.requests.append(deepcopy(request));n = len(self.requests)
                message = outer.tool(n) if n <= 8 else {'content': json.dumps(action('catalog'))}
                return {'choices': [{'message': message, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 100}}
        transport = Transport()
        result = self.complete(transport)
        self.assertEqual(result['attempts_used'], 9)
        self.assertNotIn('tools', transport.requests[-1])
        history = transport.requests[-1]['messages']
        tool_ids = [m['tool_call_id'] for m in history if m['role'] == 'tool']
        declared_ids = [call['id'] for m in history if m['role'] == 'assistant' for call in m.get('tool_calls', [])]
        self.assertEqual(len(set(tool_ids)), 8)
        self.assertEqual(tool_ids, declared_ids)
        self.assertTrue(any(m.get('content') == 'Visible observation 1' for m in history))
        self.assertFalse(any('reasoning_content' in m for m in history))
        self.assertNotIn('extra_body', transport.requests[0])
        self.assertFalse(self.h.g.budget.llm_dialogues)

    def test_opted_in_provider_field_survives_a_tool_turn(self):
        outer = self
        class Transport:
            def __init__(self):
                self.requests = []
            def complete(self, request, timeout, context):
                self.requests.append(deepcopy(request));n = len(self.requests)
                return {'choices': [{'message': outer.tool(n) if n == 1 else {'content': json.dumps(action('catalog'))},
                                     'finish_reason': 'stop'}]}
        transport = Transport()
        self.complete(transport, protocol='qwen_chat_template')
        assistant = next(m for m in transport.requests[1]['messages'] if m['role'] == 'assistant')
        self.assertEqual(assistant['reasoning_content'], 'provider field')
        self.assertEqual(assistant['content'], 'Visible observation 1')
        self.assertTrue(transport.requests[1]['extra_body']['chat_template_kwargs']['preserve_thinking'])

    def test_completed_series_is_in_the_next_sent_request_and_evidence_trace(self):
        outer = self
        class Transport:
            def __init__(self):
                self.requests = []
            def complete(self, request, timeout, context):
                self.requests.append(deepcopy(request))
                if len(self.requests) == 1:
                    art = outer.h.g.artifacts.write_json('optuna/test-series.json', {'status': 'completed'}, 'optuna_series')
                    self.gateway.feedback.publish(self.branch, {'branch_id': self.branch, 'results': [],
                        **{k: context['metadata']['detail_scope'][k] for k in ('dataset_id', 'task_spec_id', 'validation_id')},
                        'series': [{'series_id': 'series-finished', 'artifact_id': art['artifact_id'], 'status': 'completed'}]})
                    message = outer.tool(1)
                else:
                    message = {'content': json.dumps(action('catalog'))}
                return {'choices': [{'message': message, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 100}}
        transport = Transport()
        result = self.complete(transport)
        updates = [json.loads(m['content'])['research_updates'] for m in transport.requests[1]['messages']
                   if m['role'] == 'user' and m['content'].startswith('{"research_updates"')]
        self.assertEqual(updates[0]['series'][0]['series_id'], 'series-finished')
        delivery = next(c for c in result['tool_calls'] if c['name'] == 'research_updates')
        self.assertIn(updates[0]['series'][0]['artifact_id'], delivery['evidence_ids'])

    def test_cancelled_late_response_is_charged_but_never_accepted_as_a_decision(self):
        class Transport:
            def complete(self, request, timeout, context):
                self.gateway.cancelled.set()
                return {'choices': [{'message': {'content': json.dumps(action('catalog'))}, 'finish_reason': 'stop'}],
                        'usage': {'completion_tokens': 123}}
        with self.assertRaises(AgentError) as caught:
            self.complete(Transport())
        self.assertEqual(caught.exception.code, 'RESEARCH_CANCELLED')
        self.assertEqual(self.h.g.budget.output_tokens_charged, 123)
        self.assertFalse(self.h.g.budget.llm_dialogues)
        self.assertEqual(self.h.g.store.db.execute('SELECT count(*) FROM experiments').fetchone()[0], 0)

    def test_nonretryable_provider_failure_enters_explicit_fallback_cooldown(self):
        class Transport:
            def complete(self, request, timeout, context):
                error = RuntimeError('Synthetic unauthorized endpoint')
                error.status_code = 401
                raise error
        transport = Transport()
        with self.assertRaises(AgentError):
            self.complete(transport)
        with self.assertRaises(AgentError) as caught:
            transport.gateway.check_availability()
        self.assertEqual(caught.exception.code, 'LLM_COOLDOWN')
        self.assertEqual(self.h.g.budget.calls, 1)


class RollingDispatcherTests(unittest.TestCase):
    def test_free_peer_starts_another_dialogue_before_slow_peer_finishes(self):
        h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800,
            graph_train_workers=1,research_protocol='legacy',background_tuning_enabled=False)
        release = threading.Event()
        try:
            c, g = h.controller, h.g
            c.graph, c.degraded, c.research_stalls = g, False, {}
            c.call = lambda cid, payload, fn, key=None: fn(payload)
            c.llm_available = lambda pending=0: True
            c.set_fallback = lambda reason=None: None
            c.prepare_adaptive_proposal = lambda plan, *args: plan
            c.local_search = SimpleNamespace(has_series=lambda *args: False, optuna=None)
            gateway = SimpleNamespace(feedback=FeedbackMailbox(g.artifacts), cancelled=threading.Event())
            tickets = [h.issue(peer) for peer in ('researcher-1', 'researcher-2')]
            proposal = h.proposal(tickets[0])['proposal']
            calls, dispatcher = {}, threading.get_ident()
            def propose(payload):
                peer = payload['researcher_id'];calls[peer] = calls.get(peer, 0)+1
                if peer == 'researcher-1' and calls[peer] == 1:
                    return proposal
                if peer == 'researcher-1':
                    release.set()
                elif not release.wait(5):
                    raise AssertionError('Fast peer was held behind the slow dialogue')
                return {'decision': 'stop', 'plan': None, 'stop_reason': 'Synthetic stop'}
            c.researcher = SimpleNamespace(propose=propose, gateway=gateway)
            def train(plans, work):
                self.assertEqual(threading.get_ident(), dispatcher)
                for plan, ticket in zip(plans, work):
                    payload = h.proposal(ticket)
                    payload['proposal']['plan'] = plan
                    payload['manifest'] = g.constructor.build(plan)
                    h.commit(g.invoke('G04', payload), 10)
                    gateway.feedback.publish(h.root, {'branch_id': h.root, 'results': [],
                        'task_spec_id': g.spec['task_spec_id'], 'dataset_id': g.spec['dataset_id'],
                        'validation_id': g.validation['validation_id'], 'series': [], 'revision_marker': 1})
                return len(plans)
            c.run_training = train
            accepted, _ = research_batch(c, [], [], tickets)
            self.assertEqual(accepted, 1)
            self.assertEqual(calls['researcher-1'], 2)
            self.assertTrue(release.is_set())
        finally:
            release.set();h.close()


class FramedUpdatesTests(unittest.TestCase):
    """Background tuning updates reach the dialogue without resetting its question or base."""

    def setUp(self):
        self.h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800,
                              llm_max_calls=None, llm_max_total_output_tokens=None)

    def tearDown(self):
        self.h.close()

    def publish(self, gateway, branch, scope, measured, best, pending=None):
        from agent.context_details import solution_record
        g = self.h.g
        results = [{'experiment_id': eid, 'primary_metric': g.store.experiment_evaluation(eid)['primary_metric'],
                    'evaluation_id': g.store.experiment_evaluation(eid)['evaluation_id'], 'recipe_id': 'linear-regressor-v1',
                    'diagnostics_artifact_id': g.store.experiment_evaluation(eid)['diagnostics_artifact_id']}
                   for eid in measured]
        gateway.feedback.publish(branch, {'branch_id': branch, **{k: scope[k] for k in ('dataset_id', 'task_spec_id', 'validation_id')},
            'results': results, 'admission_feedback': [], 'pending_experiments': pending or [], 'series': [], 'fit_diagnostics': [],
            'solution_catalog': {eid: solution_record(g.store, g.artifacts, eid) for eid in measured},
            'best_solution': {'experiment_id': best, 'recipe_id': 'linear-regressor-v1',
                              'metric': g.store.experiment_evaluation(best)['primary_metric']}})

    def test_irrelevant_trials_keep_the_view_and_a_new_best_is_read_afresh(self):
        g, outer = self.h.g, self
        _, base_reply, _, _ = self.h.result('researcher-1', {'alpha': 1}, 10)
        _, noise_reply, _, _ = self.h.result('researcher-2', {'alpha': 2}, 14)
        _, best_reply, _, _ = self.h.result('researcher-2', {'alpha': 3}, 7)
        base, noise, best = (r['lead_submission']['experiment_id'] for r in (base_reply, noise_reply, best_reply))
        read_best = {'name': 'read_solution', 'arguments': json.dumps({'experiment_id': 'best', 'view': 'result'})}
        class Transport:
            def __init__(self):
                self.requests = []
            def complete(self, request, timeout, context):
                self.requests.append(deepcopy(request));n = len(self.requests)
                scope = context['metadata']['detail_scope']
                if n == 1:   # a worse background trial arrives
                    outer.publish(self.gateway, self.branch, scope, [base, noise], base,
                        [{'experiment_id': 'next-background-trial', 'state': 'running', 'execution_kind': 'background_tuning'}])
                elif n == 2:  # background tuning finds a better solution
                    outer.publish(self.gateway, self.branch, scope, [base, noise, best], best)
                if n <= 3:
                    calls = [{'id': 'read-'+str(n), 'type': 'function', 'function': read_best}]
                    if n == 2:  # an unanswered question keeps the dialogue moving
                        calls.append({'id': 'code-2', 'type': 'function', 'function': {'name': 'read_solution',
                            'arguments': json.dumps({'experiment_id': base, 'view': 'code'})}})
                    message = {'content': None, 'tool_calls': calls}
                else:
                    message = {'content': json.dumps(action('catalog'))}
                return {'choices': [{'message': message, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 100}}
        transport = Transport()
        gateway = Gateway(g.config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), transport)
        gateway.feedback = FeedbackMailbox(g.artifacts)
        transport.gateway = gateway
        ticket = self.h.issue()
        transport.branch = ticket['work_order']['branch_id']
        scope = g.artifacts.read_json(ticket['work_order']['context_id'])['metadata']['detail_scope']
        self.publish(gateway, transport.branch, scope, [base], base)
        result = gateway.complete({'purpose': 'research_proposal', 'context_artifact_id': ticket['work_order']['context_id'],
            'response_schema_id': 'ResearchAction-v1', 'tool_schema_ids': ['read_solution'], 'max_output_tokens': 1024,
            'deadline_seconds': 300, 'attempt_limit': 9})
        updates = [next(json.loads(m['content'])['research_updates'] for m in r['messages']
                        if m['role'] == 'user' and m['content'].startswith('{"research_updates"')) for r in transport.requests]
        frames = [u['research_frame'] for u in updates]
        # One stable question/base; only the utility reference moves with background tuning.
        self.assertEqual(len({json.dumps({k: f[k] for k in ('question', 'research_base_experiment_id')}) for f in frames}), 1)
        self.assertEqual([f['utility_reference']['experiment_id'] for f in frames], [base, base, best, best])
        self.assertEqual([u['changed_since_previous_turn'] for u in updates], [True, False, True, False])
        self.assertEqual(updates[1]['results'], updates[0]['results'])
        self.assertEqual(updates[2]['changes'][0]['to']['experiment_id'], best)
        # "best" names the delivered incumbent: the first read is reused, the new best is read afresh.
        tool_texts = [json.loads(m['content']) for m in transport.requests[-1]['messages'] if m['role'] == 'tool']
        self.assertEqual([t['experiment_id'] for t in tool_texts], [base, base, base, best])
        self.assertEqual([bool(t.get('reused')) for t in tool_texts], [False, True, False, False])
        self.assertEqual(result['parsed_payload']['action'], 'catalog')
        self.assertEqual(result['attempts_used'], 4)

    def test_background_queue_churn_is_quiet_but_a_losing_research_candidate_is_delivered(self):
        from agent.research_feedback import frame_updates
        context = {'_research_frame': {'question': 'Check representation', 'research_base_experiment_id': 'base'}}
        original = {'best_solution': {'experiment_id': 'leader', 'metric': {'value': 9}},
                    'results': [], 'candidate_results': [], 'pending_experiments': []}
        first = frame_updates(original, context)
        self.assertTrue(first['changed_since_previous_turn'])
        for state, eid in [('running', 'trial-a'), ('trained', 'trial-a'), ('running', 'trial-b')]:
            incoming = {**original, 'results': [{'experiment_id': eid}],
                        'pending_experiments': [{'experiment_id': eid, 'state': state, 'execution_kind': 'background_tuning'}]}
            reply = frame_updates(incoming, context)
            self.assertFalse(reply['changed_since_previous_turn'])
            self.assertEqual(reply['results'], first['results'])
            self.assertEqual(reply['pending_experiments'], [])
        pending = {**original, 'pending_experiments': [
            {'experiment_id': 'research-candidate', 'state': 'running', 'execution_kind': 'candidate'}]}
        self.assertTrue(frame_updates(pending, context)['changed_since_previous_turn'])
        outcome = {**original, 'candidate_results': [{'experiment_id': 'research-candidate', 'primary_metric': {'value': 12}}]}
        reply = frame_updates(outcome, context)
        self.assertTrue(reply['changed_since_previous_turn'])
        self.assertIn({'kind': 'candidate_result'}, reply['changes'])
        self.assertEqual(reply['candidate_results'], outcome['candidate_results'])
        self.assertEqual(reply['research_frame']['research_base_experiment_id'], 'base')
        self.assertEqual(original['pending_experiments'], [])
