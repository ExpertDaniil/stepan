"""Exercise authority boundaries with real generated workers and issued leases."""
from copy import deepcopy
from pathlib import Path
import io
import json
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from agent.budget import Budget
from agent.config import ROOT
from agent.errors import AgentError
from agent.orchestrator import Orchestrator
from agent.observability.console import ConciseConsole
from agent.runner import Runner
from agent.submission import Submission
from tests.helpers import Harness
from tools.code import validate_code


class ExecutionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness('regression')

    def tearDown(self):
        self.h.close()

    def worker(self, source, phase='train'):
        validate_code(source)
        code = self.h.run/'candidate.py';code.write_text(source)
        fit = self.h.run/'fit.json';fit.write_text(json.dumps(list(range(60))))
        valid = self.h.run/'valid.json';valid.write_text(json.dumps(list(range(60, 90))))
        job = dict(job_id='boundary', phase=phase, task_spec=self.h.spec, guard_generated=True,
                   train_path=str(self.h.task.parent/'train.csv'), test_path=str(self.h.task.parent/'test.csv'),
                   train_indices_path=str(fit), valid_indices_path=str(valid), code_path=str(code),
                   hyperparameters={}, seed=42, cpu_threads=1, memory_limit_mb=2048, timeout_seconds=20,
                   prediction_path=str(self.h.run/'prediction.csv'), model_path=str(self.h.run/'model.joblib'),
                   model_snapshot_max_bytes=1048576, result_path=str(self.h.run/'result.json'))
        request = self.h.run/'job.json';request.write_text(json.dumps(job))
        completed = subprocess.run([sys.executable, '-m', 'agent.worker', str(request)],
                                   cwd=ROOT, text=True, capture_output=True, timeout=30)
        return completed, json.loads(Path(job['result_path']).read_text()), Path(job['prediction_path'])

    def test_generated_pickle_hooks_are_never_called(self):
        marker = str(self.h.run/'untrusted-reduce.npy')
        source = f'''import numpy as np
class Model:
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.zeros(len(X))
    def __reduce__(self):
        np.lib.format.open_memmap({marker!r}, mode="w+", shape=(1,))
        return (float, (0,))
def make_pipeline(task_spec, params):
    return Model()
'''
        completed, result, prediction = self.worker(source, phase='refit')
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(result['model_snapshot']['status'], 'disabled_generated')
        self.assertIsNone(result['model_path'])
        self.assertFalse(Path(marker).exists())
        ready = next(json.loads(line[len('AUTOML_WORKER '):]) for line in completed.stdout.splitlines()
                     if line.startswith('AUTOML_WORKER ') and json.loads(line[len('AUTOML_WORKER '):])['stage'] == 'model_ready')
        self.assertEqual(ready['model_type'], 'generated_boundary.Model')
        self.assertEqual(ready['estimator_type'], ready['model_type'])
        import pandas as pd
        self.assertEqual(list(pd.read_csv(prediction)), self.h.spec['output']['columns'])

    def test_model_ready_reports_actual_pipeline_and_estimator_through_runner(self):
        source = '''from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
def make_pipeline(task_spec, params):
    features = ColumnTransformer([("numeric", SimpleImputer(), ["x_numeric", "z_numeric"])])
    return Pipeline([("features", features), ("model", Ridge())])
'''
        completed, result, prediction = self.worker(source)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        budget = Budget(self.h.config, time.monotonic(), self.h.log)
        runner = Runner(self.h.artifacts, self.h.config, budget, self.h.log)
        # Reuse the real worker stream and Runner's production event bridge.
        directory = self.h.run/'exp-observed'/'job-observed';directory.mkdir(parents=True)
        runner._drain(io.StringIO(completed.stdout), directory/'stdout.log')
        events = [json.loads(line) for line in (self.h.run/'events.jsonl').read_text().splitlines()]
        ready = next(e for e in events if e['event'] == 'worker.model_ready')
        self.assertEqual(ready['model_type'], 'sklearn.pipeline.Pipeline')
        self.assertEqual(ready['estimator_type'], 'sklearn.linear_model._ridge.Ridge')
        self.assertEqual(ready['experiment_id'], 'exp-observed')
        text = ConciseConsole().render(ready)
        self.assertIn('exp-observed', text)
        self.assertIn('sklearn.linear_model._ridge.Ridge', text)
        self.assertNotIn('—', text)

    def test_custom_constructor_cannot_mutate_trusted_output_schema(self):
        source = '''import numpy as np
class Model:
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.zeros(len(X))
def make_pipeline(task_spec, params):
    task_spec["output"]["columns"] = ["mutated"]
    return Model()
'''
        completed, result, prediction = self.worker(source, phase='refit')
        self.assertEqual(completed.returncode, 0, completed.stderr)
        import pandas as pd
        self.assertEqual(list(pd.read_csv(prediction)), self.h.spec['output']['columns'])

    def test_custom_prediction_conversion_stays_inside_io_guard(self):
        marker = str(self.h.run/'untrusted-conversion.npy')
        source = f'''import numpy as np
class Value:
    def __float__(self):
        np.lib.format.open_memmap({marker!r}, mode="w+", shape=(1,))
        return 0.0
    def __str__(self):
        np.lib.format.open_memmap({marker!r}, mode="w+", shape=(1,))
        return "0"
class Model:
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.array([Value() for i in X.index], dtype=object)
def make_pipeline(task_spec, params):
    return Model()
'''
        completed, result, prediction = self.worker(source)
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(result['error_type'], 'PermissionError', completed.stderr)
        self.assertFalse(Path(marker).exists())
        self.assertFalse(prediction.exists())

    def test_thread_name_cannot_grant_watchdog_io_privileges(self):
        secret = self.h.run/'secret.csv';secret.write_text('validation labels')
        script = '''from tools.runtime_guard import PipelineGuard
import threading, sys, sysconfig
from pathlib import Path
installed_source=Path(sysconfig.get_path('purelib'))/'numpy'/'__init__.py'
threading.current_thread().name='worker-watchdog'
guard=PipelineGuard(sys.argv[1]);guard.active=True
try:
    open(sys.argv[1]).read()
    installed_source.read_text(encoding='utf-8')
    print('installed library readable')
    open(sys.argv[2]).read()
except PermissionError:
    guard.active=False
    print('blocked')
else:
    guard.active=False
    raise AssertionError('thread name bypassed guard')
'''
        source = self.h.run/'source.py';source.write_text('')
        completed = subprocess.run([sys.executable, '-c', script, str(source), str(secret)],
                                   cwd=ROOT, text=True, capture_output=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('blocked', completed.stdout)
        self.assertIn('installed library readable', completed.stdout)

    def test_runner_rejects_mutated_allowance_and_wrong_owner_or_phase_before_spawn(self):
        budget = Budget(self.h.config, time.monotonic(), self.h.log)
        runner = Runner(self.h.artifacts, self.h.config, budget, self.h.log)
        experiment = {'experiment_id': 'exp-boundary'}
        lease = budget.lease(dict(operation='acquire', owner_id=experiment['experiment_id'], phase='train',
                                  requested=dict(cpu_threads=1, memory_limit_mb=512, timeout_seconds=5)))['lease']
        for field, replacement in [('cpu_threads', 6), ('memory_limit_mb', 20480), ('timeout_seconds', 1000)]:
            with self.subTest(field=field), patch('agent.runner.subprocess.Popen') as spawn:
                mutated = deepcopy(lease);mutated[field] = replacement
                with self.assertRaisesRegex(AgentError, 'resources differ'):
                    runner._start({'lease': mutated}, self.h.spec, experiment, 'train')
                spawn.assert_not_called()
        for owner, phase in [('another-owner', 'train'), (experiment['experiment_id'], 'refit')]:
            with self.subTest(owner=owner, phase=phase), patch('agent.runner.subprocess.Popen') as spawn:
                budget.leases[lease['lease_id']].update(owner_id=owner, phase=phase)
                with self.assertRaisesRegex(AgentError, 'owner or phase'):
                    runner._start({'lease': lease}, self.h.spec, experiment, 'train')
                spawn.assert_not_called()

    def test_refit_spawn_failure_releases_unstarted_lease_and_retains_published_csv(self):
        submission = Submission(self.h.spec, self.h.artifacts, self.h.store, self.h.log)
        evaluated = self.h.evaluation(2.0)
        candidate = self.h.candidate(label='0')
        report = submission.validate_output(dict(task_spec_id=self.h.spec['task_spec_id'],
            reference_test_artifact_id=self.h.spec['test_artifact_id'], candidate_artifact_id=candidate['artifact_id']))
        submission.publish(dict(validation_report_id=report['validation_report_id'],
            candidate_artifact_id=candidate['artifact_id'], candidate_sha256=report['candidate_sha256'],
            experiment_id=evaluated['experiment_id'], evaluation_id=evaluated['evaluation_id'], expected_champion_version=0))
        output = Path(self.h.spec['output']['path']);previous = output.read_bytes()
        self.h.evaluation(1.0)
        budget = Budget(self.h.config, time.monotonic(), self.h.log)
        runner = Runner(self.h.artifacts, self.h.config, budget, self.h.log)
        controller = Orchestrator.__new__(Orchestrator)
        for key, value in dict(config=self.h.config, bus=self.h.bus, artifacts=self.h.artifacts,
                               store=self.h.store, log=self.h.log, spec=self.h.spec, budget=budget,
                               runner=runner, submission=submission, graph=None).items():
            setattr(controller, key, value)
        with patch('agent.runner.subprocess.Popen', side_effect=OSError('synthetic refit spawn failure')):
            with self.assertRaises(AgentError):
                controller.finalize_best()
        self.assertFalse(runner.jobs)
        self.assertFalse(budget.leases)
        self.assertEqual(budget.snapshot()['cpu_threads_available'], 6)
        self.assertEqual(output.read_bytes(), previous)
        self.assertEqual(submission.require_output()['experiment_id'], evaluated['experiment_id'])
        self.assertEqual(next(reversed(submission.jobs.values()))['state'], 'failed')


if __name__ == '__main__':
    unittest.main()
