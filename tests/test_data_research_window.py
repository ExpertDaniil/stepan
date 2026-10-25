"""Clock-controlled policy checks, plus a real short adaptive run."""
from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

from agent.budget import Budget
from agent.config import Config
from agent.data_research import attach_context, require_action, require_feature_parameters
from agent.errors import AgentError
from agent.local_search import LocalSearch
from agent.orchestrator import Orchestrator
from tests.graph_helpers import GraphHarness
from tests.test_adaptive_research import action


class DataResearchWindowTests(unittest.TestCase):
    def test_custom_data_base_keeps_timing_identity_with_refinement_and_old_contexts(self):
        from agent.context import output_schemas
        from agent.llm.gateway import Gateway, MockTransport
        from tests.test_adaptive_research import SOURCE
        h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800, data_research_min_seconds=None)
        try:
            reply, payload = h.submit(h.issue(), {'alpha': 1.}, recipe='python-pipeline-v1', source=SOURCE)
            h.commit(reply, 7)
            g = h.g
            g.context.data_research = g.budget.data_research_snapshot
            g.budget.start_data_research()
            context_id = h.issue()['work_order']['context_id']
            context = g.artifacts.read_json(context_id)
            user = json.loads(context['messages'][1]['content'])
            source_id = user['base_result']['code_artifact_id']
            self.assertEqual(source_id, payload['proposal']['plan']['code_artifact_id'])
            g.budget.phase_deadline = time.monotonic()+9000
            gateway = Gateway(g.config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), MockTransport())
            args = {'purpose': 'research_proposal', 'context_artifact_id': context_id,
                'response_schema_id': 'ResearchAction-v1', 'tool_schema_ids': [],
                'max_output_tokens': 1024, 'deadline_seconds': 900, 'attempt_limit': 2}
            for legacy in (False, True):
                if legacy:
                    user['base_result'].pop('code_artifact_id')
                    context['messages'][1]['content'] = json.dumps(user)
                    args['context_artifact_id'] = g.artifacts.write_json('contexts/old-data-base.json', context)['artifact_id']
                with self.subTest(legacy=legacy), patch.object(g.budget.duration, 'estimate', wraps=g.budget.duration.estimate) as estimate:
                    self.assertEqual(gateway.complete(args)['parsed_payload']['action'], 'features')
                    probes = [call.args[0] for call in estimate.call_args_list if call.args[0]['model']['family'] == 'custom']
                    self.assertTrue(probes)
                    self.assertTrue(all(p['code_artifact_id'] == ('unresolved-source' if legacy else source_id) for p in probes))
        finally:
            h.close()

    def test_opt_in_auto_three_hour_window_starts_once_and_unlocks_at_thirty_minutes(self):
        with patch('agent.budget.time.monotonic', return_value=100) as clock:
            budget = Budget(Config(time_budget_seconds=10800, data_research_min_seconds=None), 100, None)
            self.assertEqual(budget.data_research_snapshot()['stage'], 'not_started')
            budget.start_data_research()
            self.assertEqual(budget.data_research_snapshot()['minimum_seconds'], 1800)
            clock.return_value = 1899.999
            budget.start_data_research()  # A second peer or a new branch must not restart it.
            self.assertTrue(budget.data_research_snapshot()['active'])
            clock.return_value = 1900
            self.assertTrue(budget.data_research_snapshot()['tuning_allowed'])
            self.assertEqual(budget.deadline, 10900)

    def test_short_runs_scale_and_explicit_disable_and_legacy_are_supported(self):
        for config, expected in [(Config(time_budget_seconds=60, data_research_min_seconds=None), 10),
                                 (Config(time_budget_seconds=90, data_research_min_seconds=None), 15),
                                 (Config(data_research_min_seconds=37), 37),
                                 (Config(data_research_min_seconds=0), 0),
                                 (Config(research_mode='catalog'), 0)]:
            with self.subTest(config=config):
                budget = Budget(config, time.monotonic(), None)
                budget.start_data_research()
                self.assertEqual(budget.data_research_snapshot()['minimum_seconds'], expected)
                self.assertEqual(budget.data_research_snapshot()['active'], expected > 0)

    def test_invalid_explicit_windows_are_rejected_by_runtime_configuration(self):
        for value in ('-1', '2.5', '60', '1800'):
            with self.subTest(value=value), patch.dict(os.environ, {
                    'TIME_BUDGET_SECONDS': '60', 'DATA_RESEARCH_MIN_SECONDS': value}, clear=True):
                with self.assertRaises(AgentError):
                    Config.from_env(profile='development')

    def test_runtime_defaults_enable_the_window_in_both_profiles(self):
        for profile in ('platform', 'development'):
            with self.subTest(profile=profile), patch.dict(os.environ, {}, clear=True):
                config = Config.from_env(profile=profile)
                self.assertIsNone(config.data_research_min_seconds)
                budget = Budget(config, time.monotonic(), None)
                budget.start_data_research()
                self.assertEqual(budget.data_research_snapshot()['minimum_seconds'], 1800)
                self.assertTrue(budget.data_research_snapshot()['active'])

    def test_data_window_never_extends_the_global_deadline_or_output_reserve(self):
        with patch('agent.budget.time.monotonic', return_value=100) as clock:
            budget = Budget(Config(time_budget_seconds=60, data_research_min_seconds=50), 100, None)
            clock.return_value = 120
            budget.start_data_research()
            clock.return_value = 159
            self.assertTrue(budget.data_research_snapshot()['active'])
            self.assertFalse(budget.snapshot()['search_allowed'])
            self.assertEqual(budget.deadline, 160)

    def test_only_data_actions_are_permitted_then_all_actions_reopen(self):
        budget = Budget(Config(data_research_min_seconds=None), time.monotonic(), None)
        budget.start_data_research()
        active = budget.data_research_snapshot()
        for kind in ['catalog', 'tune', 'stop']:
            with self.subTest(action=kind), self.assertRaises(AgentError) as caught:
                require_action(action(kind), active)
            self.assertEqual(caught.exception.code, 'DATA_RESEARCH_REQUIRED')
        require_action(action('features'), active)
        require_action(action('pipeline'), active)
        with self.assertRaises(AgentError):
            require_action(action('pipeline', feature_steps=[]), active)
        budget.data_research_started -= 1800
        for kind in ['catalog', 'tune', 'stop']:
            require_action(action(kind), budget.data_research_snapshot())

    def test_final_training_does_not_complete_an_interrupted_data_window(self):
        with patch('agent.budget.time.monotonic', return_value=100) as clock:
            budget = Budget(Config(data_research_min_seconds=None), 100, None)
            budget.start_data_research()
            clock.return_value = 1000
            budget.end_data_research('llm_call_limit')
            clock.return_value = 2000
            state = budget.data_research_snapshot()
            self.assertEqual(state['stage'], 'interrupted')
            self.assertEqual(state['elapsed_seconds'], 900)
            self.assertFalse(state['tuning_allowed'])

    def test_context_reports_remaining_wall_time_and_preserves_task_facts(self):
        user = {'profile_summary': {'data': {'meaning': 'unchanged'}}, 'assignment': 'old'}
        window = {'active': True, 'remaining_seconds': 1800}
        attach_context(user, window)
        self.assertEqual(user['profile_summary']['data'], {'meaning': 'unchanged'})
        self.assertTrue(user['profile_summary']['feature_focus'])
        self.assertEqual(user['profile_summary']['data_research'], window)
        self.assertIn('DATA RESEARCH', user['assignment'])
        attach_context(user, {'active': False, 'remaining_seconds': 0})
        self.assertNotIn('unavailable', user['assignment'])

    def test_feature_actions_cannot_hide_a_model_parameter_sweep(self):
        from agent.catalog import Catalog
        catalog = Catalog('regression', Config())
        window = {'active': True, 'remaining_seconds': 1800}
        value = action('features', hyperparameters={})
        require_feature_parameters(value, catalog, window)
        with self.assertRaises(AgentError):
            require_feature_parameters({**value, 'hyperparameters': {'alpha': 2.}}, catalog, window)
        require_feature_parameters({**value, 'hyperparameters': {'target_transform': 'shifted_log1p'}}, catalog, window)
        require_feature_parameters({**value, 'hyperparameters': {'alpha': 2.}}, catalog, {'active': False})

    def test_request_compaction_keeps_the_mandatory_time_policy(self):
        from agent.context_budget import compact_research_request, request_bytes
        user = {'profile_summary': {'analysis_history': [{'details': 'large '*10000}],
                                  'core_profile': {'columns': []}}, 'assignment': 'old'}
        window = {'active': True, 'remaining_seconds': 100, 'minimum_seconds': 1800}
        attach_context(user, window)
        request = {'messages': [{'role': 'system', 'content': 'Context'},
                                {'role': 'user', 'content': json.dumps(user)}]}
        before = request_bytes(request)
        compact_research_request(request, 5000)
        self.assertLess(request_bytes(request), before)
        packed = json.loads(request['messages'][1]['content'])
        self.assertEqual(packed['profile_summary']['data_research'], window)

    def test_optuna_and_emergency_parameter_search_cannot_bypass_the_window(self):
        h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800, data_research_min_seconds=None)
        try:
            c = h.controller
            c.graph = h.g
            c.resources = lambda: {'cpu_threads': 1, 'memory_limit_mb': 1024, 'timeout_seconds': 30}
            c.llm_available = lambda: False
            local = LocalSearch(c)
            c.local_search = local
            plan = h.proposal(h.issue(), {'alpha': 1.0})['proposal']['plan']
            request = {'search_space': {'alpha': {'type': 'float', 'low': .1, 'high': 3.}},
                       'max_trials': 2, 'budget_seconds': 300}
            c.budget.start_data_research()
            with patch.object(local.optuna, '_library', side_effect=AssertionError('Optimizer must not be loaded')):
                with self.assertRaises(AgentError) as caught:
                    local.start_series('researcher-1', h.root, plan, request=request)
                self.assertEqual(caught.exception.code, 'DATA_RESEARCH_REQUIRED')
            self.assertIsNone(local.next_choice('researcher-1', h.root, plan))
            self.assertIsNone(Orchestrator.local_plan(c, 'researcher-1', h.root, plan))
            c.budget.data_research_started -= 1800
            local.start_series('researcher-1', h.root, plan, request=request)
            self.assertIsNotNone(local.optuna.next_plan('researcher-1', h.root))
        finally:
            h.close()

    def test_an_existing_series_is_paused_without_consuming_a_trial(self):
        h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800, data_research_min_seconds=None)
        try:
            c = h.controller
            c.graph = h.g
            c.resources = lambda: {'cpu_threads': 1, 'memory_limit_mb': 1024, 'timeout_seconds': 30}
            local = LocalSearch(c)
            plan = h.proposal(h.issue(), {'alpha': 1.0})['proposal']['plan']
            local.start_series('researcher-1', h.root, plan, request={
                'search_space': {'alpha': {'type': 'float', 'low': .1, 'high': 3.}},
                'max_trials': 2, 'budget_seconds': 300})
            c.budget.start_data_research()
            self.assertFalse(local.has_series('researcher-1', h.root))
            self.assertIsNone(local.optuna.next_plan('researcher-1', h.root))
            self.assertEqual(next(iter(local.optuna.active.values()))['trials'], [])
        finally:
            h.close()

    def test_controller_rechecks_data_changes_before_launching_a_training_process(self):
        h = GraphHarness('regression', research_mode='adaptive', data_research_min_seconds=None)
        try:
            c = h.controller
            plan = h.proposal(h.issue())['proposal']['plan']
            c.budget.start_data_research()
            self.assertFalse(Orchestrator.admit_training(c, plan, 'not-launched'))
            self.assertFalse(c.budget.leases)
            self.assertFalse(c.budget.admissions)
        finally:
            h.close()

    def test_context_and_gateway_expose_the_same_restricted_action_schema(self):
        from agent.context import output_schemas
        from agent.llm.gateway import Gateway, MockTransport
        h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800, data_research_min_seconds=None)
        try:
            g = h.g
            g.context.data_research = g.budget.data_research_snapshot
            g.budget.start_data_research()
            context_id = h.issue()['work_order']['context_id']
            context = g.artifacts.read_json(context_id)
            self.assertTrue(json.loads(context['messages'][1]['content'])['profile_summary']['data_research']['active'])
            requests = []
            original = MockTransport.complete
            class Transport(MockTransport):
                def complete(self, request, timeout, context):
                    requests.append(deepcopy(request))
                    return original(self, request, timeout, context)
            schemas = output_schemas(g.bus)
            gateway = Gateway(g.config, g.budget, g.artifacts, g.bus, g.log, schemas, Transport())
            result = gateway.complete({'purpose': 'research_proposal', 'context_artifact_id': context_id,
                'response_schema_id': 'ResearchAction-v1', 'tool_schema_ids': [],
                'max_output_tokens': 1024, 'deadline_seconds': 900, 'attempt_limit': 2})
            self.assertEqual(result['parsed_payload']['action'], 'features')
            schema_message = next(m['content'] for m in requests[0]['messages'] if m['content'].startswith('Final output JSON schema:'))
            schema = json.loads(schema_message.split('\n', 1)[1])
            self.assertEqual(schema['properties']['action']['enum'], ['features', 'pipeline', 'research'])
            self.assertIn('tune', schemas['ResearchAction-v1']['properties']['action']['enum'])
        finally:
            h.close()


class DataResearchExecutionTests(unittest.TestCase):
    def test_api_exhaustion_keeps_verified_answer_without_early_parameter_search(self):
        from agent.runtime import main
        from scripts.make_demo import make_demo
        for graph in (False, True):
            with self.subTest(graph=graph), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                task = make_demo(root/'task', 'regression', rows=90, test_rows=12)
                settings = {'AUTOML_PROFILE': 'development', 'LLM_MODE': 'mock', 'RESEARCH_MODE': 'adaptive',
                    'TIME_BUDGET_SECONDS': '90', 'DATA_RESEARCH_MIN_SECONDS': '60', 'LLM_MAX_CALLS': '2',
                    'GRAPH_ENABLED': str(int(graph)), 'GRAPH_PEERS': '1', 'GRAPH_TRAIN_WORKERS': '1',
                    'GRAPH_REFINEMENT_ENABLED': '0', 'MAX_WORKERS': '1', 'TRAIN_CPU_THREADS': '1',
                    'CATALOG_WARMUP_EXPERIMENTS': '0', 'FEATURE_SEED_BUDGET_FRACTION': '0',
                    'RESEARCH_OOF_ENABLED': '0', 'LOG_LEVEL': 'ERROR', 'GRAPH_MIN_FREE_DISK_BYTES': '0'}
                with patch.dict(os.environ, settings), redirect_stderr(io.StringIO()) as stderr:
                    code = main([str(task), '--llm-mode', 'mock', '--work-dir', str(root/'runs')])
                self.assertEqual(code, 0, stderr.getvalue()[-6000:])
                run = next((root/'runs').glob('run-*'))
                events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
                phase = next(e for e in events if e['event'] == 'research.data_phase_interrupted')
                self.assertEqual(phase['reason'], 'llm_call_limit')
                self.assertFalse(phase['tuning_allowed'])
                self.assertFalse(any(e['event'] in {'optuna.series_started', 'search.local_trial',
                                                  'research.data_phase_completed'} for e in events))
                import pandas as pd
                self.assertEqual(len(pd.read_csv(task.parent/'submission.csv')), 12)
                self.assertEqual(events[-1]['state'], 'completed')

    def test_real_features_run_before_tuning_and_the_checked_answer_is_retained(self):
        for graph, refinement in ((False, False), (True, False), (True, True)):
            with self.subTest(graph=graph, refinement=refinement):
                self.execute(graph, refinement)

    def execute(self, graph, refinement):
        from agent.llm.gateway import MockTransport
        from agent.runtime import main
        from agent.scientific_research import synthetic_notes
        from scripts.make_demo import make_demo
        calls = []
        original = MockTransport.complete
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = make_demo(root/'task', 'regression', rows=120, test_rows=16)
            def complete(transport, request, timeout, context):
                response = original(transport, request, timeout, context)
                if context['metadata']['schema_id'] != 'ResearchAction-v1':
                    return response
                user = json.loads(context['messages'][1]['content'])
                window = user.get('profile_summary', user)['data_research']
                value = json.loads(response['choices'][0]['message']['content'])
                if window['active']:
                    value.update(action='features', recipe_id='linear-regressor-v1', hyperparameters={'alpha': 1.},
                        feature_program=[{'op': 'product', 'name': 'joint_'+str(len(calls)), 'columns': ['x_numeric', 'z_numeric']}],
                        feature_steps=[], code=None, tuning=None)
                elif not any(kind == 'tune' for kind, _ in calls):
                    value.pop('feature_program', None)
                    value.update(action='tune', recipe_id='linear-regressor-v1', hyperparameters={'alpha': 1.},
                        feature_steps=[], code=None, tuning={'search_space': {
                            'alpha': {'type': 'float', 'low': .2, 'high': 2.}}, 'max_trials': 2, 'budget_seconds': 30})
                else:
                    value.pop('feature_program', None)
                    value.update(action='stop', code=None, tuning=None)
                value['research_notes'] = synthetic_notes(value, user)
                value['investigation']['columns'] = ['x_numeric', 'z_numeric'] if value['action'] == 'features' else []
                calls.append((value['action'], window['active']))
                response['choices'][0]['message']['content'] = json.dumps(value)
                return response
            settings = {'AUTOML_PROFILE': 'development', 'LLM_MODE': 'mock', 'RESEARCH_MODE': 'adaptive',
                'TIME_BUDGET_SECONDS': '90', 'DATA_RESEARCH_MIN_SECONDS': '4', 'MAX_WORKERS': '1',
                'GRAPH_ENABLED': str(int(graph)), 'GRAPH_PEERS': '1', 'GRAPH_TRAIN_WORKERS': '1',
                'GRAPH_REFINEMENT_ENABLED': str(int(refinement)),
                'TRAIN_CPU_THREADS': '1', 'LLM_MAX_CALLS': '20', 'CATALOG_WARMUP_EXPERIMENTS': '0',
                'FEATURE_SEED_BUDGET_FRACTION': '0', 'RESEARCH_OOF_ENABLED': '0', 'LOG_LEVEL': 'ERROR',
                'MODEL_SNAPSHOT_MAX_BYTES': '0', 'GRAPH_MIN_FREE_DISK_BYTES': '0'}
            with patch.dict(os.environ, settings), patch.object(MockTransport, 'complete', complete), redirect_stderr(io.StringIO()) as stderr:
                code = main([str(task), '--work-dir', str(root/'runs')])
            self.assertEqual(code, 0, stderr.getvalue()[-6000:])
            run = next((root/'runs').glob('run-*'))
            events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['event'] == 'research.data_phase_completed' for e in events),
                            stderr.getvalue()[-12000:])
            started = next(i for i,e in enumerate(events) if e['event'] == 'research.data_phase_started')
            opened = next(i for i,e in enumerate(events) if e['event'] == 'research.data_phase_completed')
            series = next(i for i,e in enumerate(events) if e['event'] == 'optuna.series_started')
            self.assertLess(started, opened)
            self.assertLess(opened, series)
            self.assertTrue(any(kind == 'features' and active for kind,active in calls))
            self.assertFalse(any(kind == 'tune' and active for kind,active in calls))
            with sqlite3.connect(run/'state.sqlite') as db:
                self.assertTrue(any(json.loads(plan)['model']['family'] == 'custom' and state == 'evaluated'
                                    for state,plan in db.execute('SELECT state,plan FROM experiments')))
                self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])
                self.assertEqual(db.execute("SELECT count(*) FROM experiments WHERE state IN ('queued','running')").fetchone()[0], 0)
            import pandas as pd
            self.assertEqual(len(pd.read_csv(task.parent/'submission.csv')), 16)
            self.assertEqual(events[-1]['state'], 'completed', stderr.getvalue()[-6000:])
