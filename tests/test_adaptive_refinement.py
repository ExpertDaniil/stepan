"""Refinement must execute adaptive choices under the normal checked lifecycle."""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agent.errors import AgentError
from agent.graph.service import GraphService
from agent.llm.gateway import MockTransport
from agent.runtime import main
from scripts.make_demo import make_demo
from tests.graph_helpers import GraphHarness
from tests.test_adaptive_research import SOURCE, action


class AdaptiveRefinementAdmissionTests(unittest.TestCase):
    def harness(self):
        h = GraphHarness('regression', research_mode='adaptive', graph_refinement_enabled=True)
        self.addCleanup(h.close)
        return h

    def refine(self, h):
        return h.g.invoke('G15', {'source_experiment_id': h.g.store.best(),
            'expected_graph_revision': h.g.revision(), 'reason': 'search_time_share_reached'})

    def test_custom_source_and_its_primary_parent_are_kept_in_refinement(self):
        h = self.harness()
        h.result('researcher-1', {'alpha': 1}, 10)
        parent = h.g.store.best()
        self.refine(h)
        ticket = h.issue()
        context = h.g.artifacts.read_json(ticket['work_order']['context_id'])
        user = json.loads(context['messages'][1]['content'])
        self.assertEqual(user['research_phase'], 'refine')
        self.assertIn('python-pipeline-v1', context['metadata']['recipe_ids'])
        payload = h.proposal(ticket, {'alpha': .03}, recipe='python-pipeline-v1', source=SOURCE)
        reply = h.g.invoke('G04', payload)
        experiment = h.g.store.get(reply['lead_submission']['experiment_id'])
        self.assertEqual(experiment['state'], 'queued')
        self.assertEqual(experiment['plan']['parent_ids'], [parent])
        self.assertEqual(h.g.artifacts.resolve(experiment['plan']['code_artifact_id']).read_text(), SOURCE)

    def test_owned_feature_switch_and_recipe_remain_revisable(self):
        for recipe, params in [('linear-regressor-v1', {'alpha': 2, 'target_transform': 'shifted_log1p'}),
                               ('hist-gradient-regressor-v1', {})]:
            with self.subTest(recipe=recipe):
                h = self.harness()
                h.result('researcher-1', {'alpha': 1}, 10)
                self.refine(h)
                reply, _ = h.submit(h.issue(), params, recipe=recipe)
                self.assertEqual(reply['disposition'], 'submitted')

    def test_custom_source_still_cannot_bypass_explicit_branch_restrictions(self):
        h = self.harness()
        fork, _, _ = h.make_fork()
        reject = h.g.invoke('G02', fork)['reject_branch']['branch_id']
        h.g.invoke('G08', {'branch_id': reject, 'expected_branch_revision': 0,
                          'action': 'resume', 'reason': 'restricted refinement fixture'})
        h.result('researcher-1', {'alpha': 7}, 7, branch=reject)
        self.refine(h)
        payload = h.proposal(h.issue(branch=reject), {'alpha': .03}, recipe='python-pipeline-v1', source=SOURCE)
        with self.assertRaises(AgentError) as caught:
            h.g.invoke('G04', payload)
        self.assertEqual(caught.exception.code, 'UNVERIFIABLE_CONSTRAINT')

    def test_generated_manifest_is_still_independently_verified(self):
        h = self.harness()
        h.result('researcher-1', {'alpha': 1}, 10)
        self.refine(h)
        payload = h.proposal(h.issue(), {'alpha': .03}, recipe='python-pipeline-v1', source=SOURCE)
        payload['manifest']['solution_hash'] = '0' * 64
        with self.assertRaises(AgentError) as caught:
            h.g.invoke('G04', payload)
        self.assertEqual(caught.exception.code, 'MANIFEST_MISMATCH')


class AdaptiveRefinementExecutionTests(unittest.TestCase):
    def test_custom_pipeline_is_actually_trained_and_evaluated_in_final_phase(self):
        self.execute(SOURCE, expected_state='evaluated')

    def test_failed_custom_training_preserves_the_previous_checked_answer(self):
        self.execute(SOURCE.replace('Ridge(alpha=params["alpha"])', 'Ridge(alpha=-1.0)'), expected_state='failed')

    def execute(self, source, expected_state):
        original_complete = MockTransport.complete
        original_due = GraphService.refinement_due
        phases, previous_answer = [], []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = make_demo(root/'task', 'regression', rows=120, test_rows=16)
            def due(graph, pending=None):
                if graph.refinement():
                    return None
                best = graph.store.best()
                if best and graph.store.get(best)['plan']['model']['family'] != 'dummy':
                    return 'search_time_share_reached'
                return original_due(graph, pending)
            def complete(transport, request, timeout, context):
                if context['metadata']['schema_id'] != 'ResearchAction-v1':
                    return original_complete(transport, request, timeout, context)
                user = json.loads(context['messages'][1]['content'])
                phases.append(user['research_phase'])
                if len(phases) == 1:
                    previous_answer.append((task.parent/'submission.csv').read_bytes())
                    value = action(code=source)
                else:
                    value = action('stop', code=None, hyperparameters={}, feature_steps=[])
                return {'choices': [{'message': {'role': 'assistant', 'content': json.dumps(value)},
                                     'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 100, 'completion_tokens': 100}}
            settings = {'LLM_MODE': 'mock', 'RESEARCH_MODE': 'adaptive', 'TIME_BUDGET_SECONDS': '120',
                'RESEARCH_PROTOCOL':'legacy','BACKGROUND_TUNING_ENABLED':'0','DATA_RESEARCH_MIN_SECONDS':'0',
                'GRAPH_ENABLED': '1', 'GRAPH_REFINEMENT_ENABLED': '1', 'GRAPH_PEERS': '1',
                'GRAPH_TRAIN_WORKERS': '1', 'MAX_WORKERS': '1', 'TRAIN_CPU_THREADS': '1',
                'CATALOG_WARMUP_EXPERIMENTS': '1', 'LLM_MAX_CALLS': '8', 'LOG_LEVEL': 'ERROR',
                'MODEL_SNAPSHOT_MAX_BYTES': '0', 'GRAPH_MIN_FREE_DISK_BYTES': '0', 'FEATURE_RESEARCH_ENABLED': '0'}
            with patch.dict(os.environ, settings), patch.object(MockTransport, 'complete', complete), \
                    patch.object(GraphService, 'refinement_due', due), redirect_stderr(io.StringIO()) as stderr:
                result = main([str(task), '--llm-mode', 'mock', '--work-dir', str(root/'runs')])
            self.assertEqual(result, 0, stderr.getvalue()[-7000:])
            self.assertTrue(phases, stderr.getvalue()[-10000:])
            self.assertEqual(set(phases), {'refine'})
            run = next((root/'runs').glob('run-*'))
            with sqlite3.connect(run/'state.sqlite') as db:
                experiments = [(state, json.loads(plan)) for state, plan in db.execute('SELECT state,plan FROM experiments')]
                custom = [(state, plan) for state, plan in experiments if plan['model']['family'] == 'custom']
                self.assertEqual(len(custom), 1, stderr.getvalue()[-7000:])
                self.assertEqual(custom[0][0], expected_state, stderr.getvalue()[-7000:])
                self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])
                self.assertEqual(db.execute("SELECT count(*) FROM experiments WHERE state IN ('queued','running')").fetchone()[0], 0)
            events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
            self.assertFalse(any(e['event'] == 'graph.proposal_rejected' for e in events))
            self.assertEqual(sum(e['event'] == 'job.started' for e in events), sum(e['event'] == 'job.finished' for e in events))
            if expected_state == 'failed':
                self.assertEqual((task.parent/'submission.csv').read_bytes(), previous_answer[0])
            import pandas as pd
            output = pd.read_csv(task.parent/'submission.csv')
            self.assertEqual(len(output), 16)
            self.assertTrue(output['label'].notna().all())
