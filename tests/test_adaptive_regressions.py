"""Regressions from the first live adaptive run; no paid API calls or real task data."""
from contextlib import redirect_stderr
from copy import deepcopy
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

from agent.context import output_schemas
from agent.errors import AgentError
from agent.llm.gateway import Gateway, MockTransport
from agent.local_search import LocalSearch
from agent.orchestrator import Orchestrator
from agent.runtime import main
from scripts.make_demo import make_demo
from tests.graph_helpers import GraphHarness
from tests.test_adaptive_research import action
from tools.analysis import AnalysisService


class AdaptiveToolRegressions(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness('regression', research_mode='adaptive')
        self.g = self.h.g
        self.service = AnalysisService(self.g.spec, self.g.validation, self.g.artifacts, self.g.budget, self.g.log)

    def tearDown(self):
        self.h.close()

    def test_tool_error_can_be_corrected_without_replaying_provider_reasoning(self):
        requests = []
        target = self.g.spec['target_column']
        class Transport:
            def complete(transport, request, timeout, context):
                requests.append(deepcopy(request))
                number = len(requests)
                if number < 3:
                    query = dict(kind='group', columns=['x_numeric', target] if number == 1 else ['x_numeric'],
                                 group_by='x_numeric', include_target=True)
                    message = {'role': 'assistant', 'content': None, 'reasoning_content': 'provider reasoning '*6000,
                               'tool_calls': [{'id': 'analysis-'+str(number), 'type': 'function', 'function': {
                                   'name': 'analyze_data', 'arguments': json.dumps(query)}}]}
                else:
                    message = {'role': 'assistant', 'content': json.dumps(action('catalog'))}
                return {'choices': [{'message': message, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 100, 'completion_tokens': 100}}
            def close(transport):
                pass
        gateway = Gateway(self.g.config, self.g.budget, self.g.artifacts, self.g.bus, self.g.log,
                          output_schemas(self.g.bus), Transport())
        gateway.analysis = self.service
        ticket = self.h.issue()
        result = gateway.complete(dict(purpose='research_proposal', context_artifact_id=ticket['work_order']['context_id'],
            response_schema_id='ResearchAction-v1', tool_schema_ids=['analyze_data'], max_output_tokens=1024,
            deadline_seconds=30, attempt_limit=4))
        self.assertEqual(result['attempts_used'], 3)
        error = json.loads(requests[1]['messages'][-1]['content'])['error']
        self.assertIn('include_target', error['message'])
        report = json.loads(next(m['content'] for m in reversed(requests[2]['messages']) if m['role'] == 'tool'))
        self.assertTrue(report['report']['groups'])
        self.assertEqual(len(self.service.recent()), 1)
        for request in requests[1:]:
            self.assertFalse(any('reasoning_content' in m for m in request['messages']))
            self.assertLessEqual(len(json.dumps(request, ensure_ascii=False).encode()), self.g.config.graph_context_bytes)

    def test_cached_analysis_cannot_be_mutated_by_a_caller(self):
        query = dict(kind='summary', columns=['x_numeric'], group_by=None, include_target=True)
        first = self.service.analyze(query)
        expected = deepcopy(first)
        first['report']['columns'].clear()
        self.assertEqual(self.service.analyze(query), expected)

    def test_expired_tuning_authorization_does_not_poison_catalogue_fallback(self):
        c = self.h.controller
        c.llm_available = lambda: False
        c.resources = lambda: {'cpu_threads': 1, 'memory_limit_mb': 1024, 'timeout_seconds': 30}
        local = LocalSearch(c)
        plan = self.h.proposal(self.h.issue())['proposal']['plan']
        local.start_series('researcher-1', self.h.root, plan,
                           request={'parameters': ['alpha'], 'max_trials': 3, 'budget_seconds': 30})
        local.authorizations[(self.h.root, 'researcher-1')]['deadline'] = time.monotonic()-1
        self.assertFalse(local.has_series('researcher-1', self.h.root))
        self.assertNotIn((self.h.root, 'researcher-1'), local.authorizations)
        self.assertIsNotNone(local.next_choice('researcher-1', self.h.root, plan))

    def test_tuning_rebase_preserves_axes_that_were_not_authorized(self):
        c = self.h.controller
        c.llm_available = lambda: True
        c.resources = lambda: {'cpu_threads': 1, 'memory_limit_mb': 1024, 'timeout_seconds': 30}
        _, _, base, _ = self.h.result('researcher-1', {'alpha': 1}, 10)
        self.h.result('researcher-1', {'alpha': 2, 'target_transform': 'shifted_log1p'}, 8)
        local = LocalSearch(c)
        plan = base['proposal']['plan']
        local.start_series('researcher-1', self.h.root, plan,
                           request={'parameters': ['alpha'], 'max_trials': 3, 'budget_seconds': 30})
        choice = local.next_choice('researcher-1', self.h.root, plan)
        self.assertIsNotNone(choice)
        self.assertEqual(choice[1]['target_transform'], plan['model']['hyperparameters']['target_transform'])


class AdaptiveSearchRegressions(unittest.TestCase):
    def run_search(self, graph, scripted, extra=None):
        original = MockTransport.complete
        def transport(adapter, request, timeout, context):
            if context['metadata']['schema_id'] != 'ResearchAction-v1':
                return original(adapter, request, timeout, context)
            payload = scripted(context)
            return {'choices': [{'message': {'role': 'assistant', 'content': json.dumps(payload)}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 100}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = make_demo(root/'task', 'regression', rows=100, test_rows=12)
            settings = dict(LLM_MODE='mock', RESEARCH_MODE='adaptive', TIME_BUDGET_SECONDS='120',
                # Isolate tuning/recovery here; mandatory timing has dedicated enabled checks.
                DATA_RESEARCH_MIN_SECONDS='0',
                GRAPH_ENABLED=str(int(graph)), GRAPH_REFINEMENT_ENABLED='0', MAX_WORKERS='1', GRAPH_PEERS='1',
                GRAPH_TRAIN_WORKERS='1', CATALOG_WARMUP_EXPERIMENTS='0', LLM_MAX_CALLS='10', LOG_LEVEL='ERROR',
                MODEL_SNAPSHOT_MAX_BYTES='0', GRAPH_MIN_FREE_DISK_BYTES='0', FEATURE_RESEARCH_ENABLED='0')
            settings.update(extra or {})
            with patch.dict(os.environ, settings), patch.object(MockTransport, 'complete', transport), redirect_stderr(io.StringIO()) as stderr:
                code = main([str(task), '--llm-mode', 'mock', '--work-dir', str(root/'runs')])
            self.assertEqual(code, 0, stderr.getvalue()[-7000:])
            run = next((root/'runs').glob('run-*'))
            events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
            db = sqlite3.connect(run/'state.sqlite')
            plans = [(state, json.loads(plan)) for state, plan in db.execute('SELECT state,plan FROM experiments ORDER BY rowid')]
            db.close()
            return events, plans

    def test_tune_of_a_measured_base_runs_new_trials_in_both_controllers(self):
        def scripted(context):
            number = context['metadata']['round_no']
            if number == 0:
                return action('catalog', hyperparameters={'alpha': 1})
            if number == 1:
                return action('tune', hyperparameters={'alpha': 1},
                              tuning={'parameters': ['alpha'], 'max_trials': 3, 'budget_seconds': 30})
            return action('stop', hyperparameters={})
        for graph in (False, True):
            with self.subTest(graph=graph):
                events, plans = self.run_search(graph, scripted)
                self.assertTrue(any(e['event'] == 'research.action_admitted' and e['action'] == 'tune' for e in events))
                novel = [p for state, p in plans if state == 'evaluated' and p['recipe_id'] == 'linear-regressor-v1'
                         and p['model']['hyperparameters']['alpha'] != 1]
                self.assertTrue(novel, 'The cached base must not cancel the authorized tuning series')

    def test_bad_generated_code_does_not_end_search_before_a_corrected_proposal(self):
        def scripted(context):
            number = context['metadata']['round_no']
            if number == 0:
                return action(code='def make_pipeline(task_spec, params):\n    return open("train.csv")\n')
            if number == 1:
                return action('catalog')
            return action('stop', hyperparameters={})
        for graph in (False, True):
            with self.subTest(graph=graph):
                events, plans = self.run_search(graph, scripted)
                self.assertTrue(any(e['event'] == 'research.proposal_failed' for e in events))
                self.assertTrue(any(state == 'evaluated' and p['model']['family'] != 'dummy' for state, p in plans))
                self.assertFalse(any(e['event'] == 'search.finished' and e.get('stop_reason') == 'no_affordable_candidates'
                                     for e in events))

    def test_failed_refit_does_not_block_a_later_publishable_candidate(self):
        failed = []
        original = Orchestrator._finalize_best
        def fail_once(controller):
            candidate = controller.final_candidate()
            if candidate and not failed and controller.store.get(candidate)['plan']['model']['family'] != 'dummy':
                failed.append(candidate)
                raise AgentError('REFIT_FAILED', 'Synthetic full-train failure after a successful validation fit')
            return original(controller)
        def scripted(context):
            number = context['metadata']['round_no']
            return (action('catalog', hyperparameters={'alpha': .03 if number == 0 else 100})
                    if number < 2 else action('stop', hyperparameters={}))
        with patch.object(Orchestrator, '_finalize_best', fail_once):
            events, plans = self.run_search(True, scripted)
        publications = [e for e in events if e['event'] == 'submission.published']
        self.assertEqual(len(failed), 1)
        self.assertGreaterEqual(len(publications), 2)
        self.assertNotEqual(publications[-1]['experiment_id'], failed[0])
        self.assertLess(publications[-1]['score'], publications[0]['score'])


if __name__ == '__main__':
    unittest.main()
