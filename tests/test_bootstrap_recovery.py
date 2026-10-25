import io
import json
import os
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.errors import AgentError
from agent.llm.gateway import MockTransport
from agent.runtime import main
from agent.submission import Submission
from agent.task_interpreter import recover_contract
from scripts.make_demo import make_demo
from tests.helpers import Harness


class BootstrapRecoveryTests(unittest.TestCase):
    def recovery(self, h, text=None):
        return recover_contract({'task_text': text or h.task.read_text(), 'task_root': str(h.task.parent),
            'task_text_artifact_id': h.code['artifact_id'], 'candidate_paths': {
                'train': str(h.task.parent/'train.csv'), 'test': str(h.task.parent/'test.csv'),
                'sample_submission': str(h.task.parent/'sample_submission.csv')}}, h.profile, h.artifacts)

    def test_unscored_atomic_output_preserves_literal_ids_and_labels(self):
        for kind in ('regression', 'classification'):
            h = Harness(kind)
            try:
                r = self.recovery(h)
                self.assertIsNotNone(r)
                s = Submission(r['spec'], h.artifacts, h.store, h.log)
                s.publish_emergency(r)
                self.assertTrue(s.require_emergency())
                self.assertIsNone(h.store.champion())
                self.assertIsNone(s.emergency['evaluation'])
                self.assertIn('000090', Path(r['spec']['output']['path']).read_text())
                self.assertFalse(list(h.task.parent.glob('.submission-*.tmp')))
            finally:
                h.close()

    def test_output_contract_can_precede_metric(self):
        h = Harness('regression')
        try:
            text = h.task.read_text().split('```agent-demo')[0].replace('Metric: RMSE, lower is better.', 'Regression task; metric unspecified.')
            r = self.recovery(h, text)
            self.assertIsNotNone(r)
            self.assertIsNone(r['draft'])
        finally:
            h.close()

    def test_ambiguous_output_or_target_is_not_guessed(self):
        h = Harness('regression')
        try:
            self.assertIsNone(self.recovery(h, h.task.read_text()+'\nWrite alternative.csv too.'))
            profile = h.artifacts.read_json(h.profile['profile_artifact_id'])
            profile['data_dictionary'] = [{'feature': 'wrong_target', 'role': 'target'}]
            h.artifacts.write_json('task/conflict.json', profile, ident='conflicting-profile')
            h.profile['profile_artifact_id'] = 'conflicting-profile'
            self.assertIsNone(self.recovery(h))
        finally:
            h.close()

    def test_early_provider_failure_still_trains_without_llm(self):
        for kind in ('regression', 'classification'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                task = make_demo(Path(temp)/'task', kind, rows=120, test_rows=20)
                env = {'LLM_MODE': 'mock', 'LOG_LEVEL': 'ERROR', 'TIME_BUDGET_SECONDS': '60',
                    'MAX_RESEARCH_EXPERIMENTS': '1', 'CATALOG_WARMUP_EXPERIMENTS': '1',
                    'RESEARCH_MODE': 'catalog', 'GRAPH_ENABLED': '0'}
                with patch.dict(os.environ, env), patch.object(MockTransport, 'complete', side_effect=AgentError('INVALID_LLM_OUTPUT', 'invalid JSON')), redirect_stderr(io.StringIO()):
                    code = main([str(task), '--work-dir', str(Path(temp)/'runs')])
                self.assertEqual(code, 0)
                self.assertTrue((task.parent/'submission.csv').is_file())
                events = [json.loads(l) for l in next((Path(temp)/'runs').glob('run-*/events.jsonl')).read_text().splitlines()]
                emergency = next(i for i,e in enumerate(events) if e['event'] == 'submission.emergency_published')
                api = next(i for i,e in enumerate(events) if e['event'] == 'llm.request_started') if any(e['event'] == 'llm.request_started' for e in events) else len(events)
                self.assertLess(emergency, api)
                self.assertTrue(any(e['event'] == 'experiment.evaluated' for e in events))
