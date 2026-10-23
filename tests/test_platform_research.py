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
            def analyze(service, args):
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
        self.assertEqual([m['tool_call_id'] for m in history if m['role'] == 'tool'], ['inspect-'+str(i) for i in range(1, 9)])
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


class RollingDispatcherTests(unittest.TestCase):
    def test_free_peer_starts_another_dialogue_before_slow_peer_finishes(self):
        h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800)
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
                    gateway.feedback.publish(h.root, {'branch_id': h.root, 'results': [], 'series': [], 'revision_marker': 1})
                return len(plans)
            c.run_training = train
            accepted, _ = research_batch(c, [], [], tickets)
            self.assertEqual(accepted, 1)
            self.assertEqual(calls['researcher-1'], 2)
            self.assertTrue(release.is_set())
        finally:
            release.set();h.close()
