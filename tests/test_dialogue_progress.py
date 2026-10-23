"""Synthetic budget and dispatcher regressions; no network or estimator training."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import threading
import time
from types import MethodType, SimpleNamespace
import unittest

from agent.budget import Budget
from agent.config import Config
from agent.context import output_schemas
from agent.errors import AgentError
from agent.graph.controller import research_batch
from agent.llm.gateway import Gateway
from agent.local_search import LocalSearch
from agent.orchestrator import Orchestrator
from tests.graph_helpers import GraphHarness
from tests.test_adaptive_research import action
from tools.analysis import AnalysisService


class DialogueBudgetTests(unittest.TestCase):
    def budget(self, calls=5):
        return Budget(Config(llm_max_calls=calls, llm_max_total_output_tokens=10000), time.monotonic(), SimpleNamespace())

    def spend(self, budget, ident):
        number = budget.begin_llm_call(100, ident)
        budget.finish_llm_call(number, {'completion_tokens': 10})
        return number

    def test_another_request_cannot_spend_protected_final_or_repair_capacity(self):
        budget = self.budget()
        first, slots = budget.reserve_dialogue(100)
        second, _ = budget.reserve_dialogue(100)
        self.assertEqual(slots, 2)
        self.assertFalse(budget.prepare_dialogue_turn(first, 100, allow_tools=True))
        self.spend(budget, first)
        self.assertEqual(budget.snapshot()['llm_calls_remaining'], 0)
        with self.assertRaises(AgentError):
            budget.begin_llm_call(100)
        for ident in (second, first):
            self.assertTrue(budget.prepare_dialogue_turn(ident, 100, allow_tools=True))
            self.spend(budget, ident)
            self.assertTrue(budget.prepare_dialogue_turn(ident, 100, allow_tools=False))
            self.spend(budget, ident)
            budget.release_dialogue(ident)
        self.assertEqual(budget.calls, 5)
        self.assertEqual(budget.token_snapshot()['output_reserved'], 0)

    def test_concurrent_dialogues_keep_their_own_terminal_requests(self):
        budget, barrier = self.budget(4), threading.Barrier(2)
        def complete():
            ident, slots = budget.reserve_dialogue(100)
            barrier.wait(timeout=3)
            numbers = []
            for _ in range(slots):
                self.assertTrue(budget.prepare_dialogue_turn(ident, 100, allow_tools=True))
                numbers.append(self.spend(budget, ident))
            budget.release_dialogue(ident)
            return numbers
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(complete) for _ in range(2)]
            numbers = [n for future in futures for n in future.result()]
        self.assertEqual(sorted(numbers), [1, 2, 3, 4])
        self.assertFalse(budget.llm_dialogues)

    def test_only_one_available_request_goes_directly_to_final(self):
        budget = self.budget(1)
        ident, slots = budget.reserve_dialogue(100)
        self.assertEqual(slots, 1)
        self.assertTrue(budget.prepare_dialogue_turn(ident, 100, allow_tools=True))
        self.spend(budget, ident)
        budget.release_dialogue(ident)
        self.assertEqual(budget.calls, 1)

    def test_phase_caps_and_output_reservations_are_not_bypassed(self):
        budget = self.budget(10)
        budget.set_phase_caps(calls=2, output_tokens=200)
        ident, slots = budget.reserve_dialogue(100)
        self.assertEqual(slots, 2)
        self.assertEqual(budget.token_snapshot()['output_reserved_dialogues'], 200)
        with self.assertRaises(AgentError):
            budget.reserve_dialogue(100)
        budget.release_dialogue(ident)
        self.assertEqual(budget.calls, 0)
        self.assertEqual(budget.snapshot()['llm_calls_remaining'], 2)


class DialogueGatewayTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800, llm_max_calls=3)

    def tearDown(self):
        self.h.close()

    def complete(self, transport):
        g = self.h.g
        gateway = Gateway(g.config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), transport)
        gateway.analysis = AnalysisService(g.spec, g.validation, g.artifacts, g.budget, g.log)
        return gateway.complete(dict(purpose='research_proposal', context_artifact_id=self.h.issue()['work_order']['context_id'],
            response_schema_id='ResearchAction-v1', tool_schema_ids=['analyze_data'], max_output_tokens=1024,
            deadline_seconds=300, attempt_limit=8))

    def test_last_free_tool_turn_is_followed_by_final_and_long_json_repair(self):
        good = action('catalog', hypothesis='Synthetic complete decision. '+('detail '*400))
        malformed = json.dumps(good)[:-1]
        class Transport:
            def __init__(transport):
                transport.requests = []
            def complete(transport, request, timeout, context):
                transport.requests.append(deepcopy(request))
                if len(transport.requests) == 1:
                    message = {'tool_calls': [{'id': 'inspect', 'type': 'function', 'function': {'name': 'analyze_data',
                        'arguments': json.dumps(dict(kind='summary', columns=['x_numeric'], group_by=None, include_target=False))}}]}
                else:
                    message = {'content': malformed if len(transport.requests) == 2 else json.dumps(good)}
                return {'choices': [{'message': message, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 100}}
        transport = Transport()
        result = self.complete(transport)
        self.assertEqual(result['parsed_payload'], good)
        self.assertEqual(result['attempts_used'], 3)
        self.assertIn('tools', transport.requests[0])
        self.assertNotIn('tools', transport.requests[1])
        self.assertNotIn('tools', transport.requests[2])
        self.assertTrue(any(m.get('role') == 'assistant' and m.get('content') == malformed for m in transport.requests[2]['messages']))
        self.assertEqual(self.h.g.budget.calls, 3)
        self.assertFalse(self.h.g.budget.llm_dialogues)
        self.assertEqual(self.h.g.budget.token_snapshot()['output_reserved'], 0)

    def test_provider_failure_releases_unused_terminal_capacity(self):
        class Transport:
            def complete(transport, request, timeout, context):
                error = RuntimeError('synthetic unauthorized')
                error.status_code = 401
                raise error
        with self.assertRaises(AgentError):
            self.complete(Transport())
        self.assertFalse(self.h.g.budget.llm_dialogues)
        self.assertFalse(self.h.g.budget.pending_llm_tokens)
        self.assertEqual(self.h.g.budget.calls, 1)


class TuningProgressTests(unittest.TestCase):
    def test_multiple_authorized_trials_finish_while_the_other_peer_is_still_reasoning(self):
        h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800)
        release = threading.Event()
        try:
            c, g = h.controller, h.g
            c.graph, c.degraded, c.research_feedback = g, False, []
            c.resources = lambda: {'cpu_threads': 1, 'memory_limit_mb': 1024, 'timeout_seconds': 30}
            c.llm_available = lambda pending=0: True
            c.call = lambda cid, payload, fn, key=None: fn(payload)
            c.local_search = LocalSearch(c)
            for name in ('local_plan', 'authorize_research', 'prepare_adaptive_proposal', 'record_research_feedback'):
                setattr(c, name, MethodType(getattr(Orchestrator, name), c))
            tickets = [h.issue(peer) for peer in ('researcher-1', 'researcher-2')]
            proposal = h.proposal(tickets[0])['proposal']
            plan = proposal['plan']
            tuning = {'search_space': {'alpha': {'type': 'float', 'low': 2.0, 'high': 4.0}}, 'max_trials': 3, 'budget_seconds': 300}
            g.artifacts.write_json('research/unit-intent.json', {'action': 'tune', 'tuning': tuning, 'evidence_ids': []},
                                   'research_intent', ident='intent-'+plan['plan_id'])
            def propose(payload):
                if payload['researcher_id'] == 'researcher-1':
                    return proposal
                if not release.wait(3):
                    raise AssertionError('Authorized continuations waited for the slow peer')
                return {'decision': 'stop', 'plan': None, 'stop_reason': 'Synthetic peer completed'}
            c.researcher = SimpleNamespace(propose=propose)
            dispatcher, trained = threading.get_ident(), []
            def train(plans, work):
                self.assertEqual(threading.get_ident(), dispatcher)
                for candidate, ticket in zip(plans, work):
                    payload = h.proposal(ticket, candidate['model']['hyperparameters'])
                    payload['proposal']['plan'] = candidate
                    payload['manifest'] = g.constructor.build(candidate)
                    reply = g.invoke('G04', payload)
                    eid = reply['lead_submission']['experiment_id']
                    c.local_search.optuna.bind(candidate, eid, ticket['work_order']['work_order_id'])
                    h.commit(reply, 10-len(trained))
                    trained.append(eid)
                    c.local_search.optuna.sync()
                    if len(trained) == 3:
                        release.set()
                return len(plans)
            c.run_training = train
            accepted, active = research_batch(c, [], [], tickets)
            self.assertEqual(accepted, 3)
            self.assertTrue(active)
            self.assertEqual(len(set(trained)), 3)
            self.assertFalse(c.degraded)
        finally:
            release.set()
            h.close()
