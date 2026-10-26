"""Sealed generated pipelines can train ctypes-based boosters, but still cannot load new natives."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = '''
import sys
from tools.runtime_guard import PipelineGuard, prepare_pipeline_imports
prepare_pipeline_imports(sys.argv[1])
guard = PipelineGuard(sys.argv[1]); guard.active = True
import numpy as np
X = np.random.RandomState(0).normal(size=(200, 4)); y = X[:, 0]*2 + X[:, 1]
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
for model in (LGBMRegressor(n_estimators=20, verbose=-1, n_jobs=1), XGBRegressor(n_estimators=20, n_jobs=1)):
    model.fit(X, y)
    assert np.isfinite(model.predict(X)).all()
print('boosters_trained')
import ctypes, ctypes.util
try:
    ctypes.CDLL(ctypes.util.find_library('c'))
except PermissionError:
    print('new_loader_blocked')
guard.active = False
'''


class NativeGuardTests(unittest.TestCase):
    def test_lightgbm_and_xgboost_train_inside_the_sealed_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'candidate.py'
            source.write_text('import lightgbm\nimport xgboost\n\ndef make_pipeline(task_spec, params):\n    return None\n')
            done = subprocess.run([sys.executable, '-c', SCRIPT, str(source)], cwd=ROOT,
                                  capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-3000:])
        self.assertIn('boosters_trained', done.stdout)
        self.assertIn('new_loader_blocked', done.stdout)


if __name__ == '__main__':
    unittest.main()


BLEND_SOURCE = '''import numpy as np
from agent.ml import build_pipeline

DEFAULT_PARAMS = {'strong_weight': 0.7}


class Blend:
    def __init__(self, task_spec, params):
        self.task_spec, self.params = task_spec, params

    def fit(self, X, y):
        owned = {'_cpu_threads': self.params.get('_cpu_threads', 1), '_seed': self.params.get('_seed', 0)}
        self.strong = build_pipeline('strong-regressor-v1', self.task_spec, {'bags': 1, **owned}).fit(X, y)
        self.catboost = build_pipeline('catboost-regressor-v1', self.task_spec, {'iterations': 20, **owned}).fit(X, y)
        return self

    def predict(self, X):
        weight = self.params['strong_weight']
        return weight*np.asarray(self.strong.predict(X))+(1-weight)*np.asarray(self.catboost.predict(X))


def make_pipeline(task_spec, params):
    return Blend(task_spec, params)
'''


class ComposedPipelineGuardTests(unittest.TestCase):
    """Owned wrappers load their backend lazily inside fit; the sealed worker must still run them."""

    def run_worker(self, h, source, hyperparameters, phase, folds=None):
        import json
        from agent.feature_schema import FACTORY_SOURCE
        from tools.code import write_code
        code = write_code({'researcher_id': 'researcher-1', 'filename': 'candidate.py',
                           'content': FACTORY_SOURCE if source is None else source,
                           'declared_entrypoint': 'make_pipeline'}, h.artifacts)['artifact']
        directory = h.run/('job-'+phase+'-'+code['sha256'][:8])
        directory.mkdir(parents=True, exist_ok=True)
        rows = list(range(90))
        (directory/'fit.json').write_text(json.dumps(rows[:60]))
        (directory/'valid.json').write_text(json.dumps(rows[60:]))
        job = {'job_id': 'guard-'+phase, 'phase': phase, 'task_spec': h.spec, 'guard_generated': True,
               'train_path': str(h.artifacts.resolve(h.spec['train_artifact_id'])),
               'test_path': str(h.artifacts.resolve(h.spec['test_artifact_id'])),
               'code_path': str(h.artifacts.resolve(code['artifact_id'])), 'code_sha256': code['sha256'],
               'hyperparameters': hyperparameters, 'seed': 7, 'cpu_threads': 1, 'memory_limit_mb': 4096,
               'timeout_seconds': 240, 'prediction_path': str(directory/'prediction.csv'),
               'model_path': str(directory/'pipeline.joblib'), 'model_snapshot_max_bytes': 64*1024*1024,
               'result_path': str(directory/'result.json'),
               'train_indices_path': str(directory/'fit.json'), 'valid_indices_path': str(directory/'valid.json')}
        if folds:
            (directory/'folds.json').write_text(json.dumps(folds))
            job['fold_plan_path'] = str(directory/'folds.json')
        (directory/'job.json').write_text(json.dumps(job))
        done = subprocess.run([sys.executable, str(ROOT/'agent/worker.py'), str(directory/'job.json')], cwd=ROOT,
                              env={'PYTHONPATH': str(ROOT), 'PATH': '/usr/bin:/bin'}, capture_output=True, text=True, timeout=300)
        result = json.loads((directory/'result.json').read_text())
        self.assertTrue(result['success'], (result, done.stderr[-3000:]))
        return directory, result

    def check_complete_path(self, source, hyperparameters):
        import json
        import joblib
        import pandas as pd
        from tests.helpers import Harness
        h = Harness('regression')
        try:
            folds = [{'fit': list(range(40)), 'valid': list(range(60, 75))},
                     {'fit': list(range(60)), 'valid': list(range(75, 90))}]
            directory, _ = self.run_worker(h, source, hyperparameters, 'train', folds)
            validation = pd.read_csv(directory/'prediction.csv')
            self.assertEqual(validation['__row_index'].tolist(), list(range(60, 90)))
            directory, result = self.run_worker(h, source, hyperparameters, 'refit')
            self.assertEqual(result['model_snapshot']['status'], 'saved')
            submission = pd.read_csv(directory/'prediction.csv')
            # Snapshots are diagnostics; a custom class unpickles once its source module is importable.
            import importlib.util
            job = json.loads((directory/'job.json').read_text())
            spec = importlib.util.spec_from_file_location('generated_guard_refit', job['code_path'])
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            model = joblib.load(directory/'pipeline.joblib')
            test = pd.read_csv(h.artifacts.resolve(h.spec['test_artifact_id']))
            features = [c for c in test.columns if c not in set(h.spec['excluded_features'])]
            reloaded = model.predict(test[features])
            column = h.spec['output']['prediction_columns'][0]
            self.assertTrue(pd.Series(reloaded).notna().all())
            self.assertEqual(len(reloaded), len(submission[column]))
        finally:
            h.close()

    def test_feature_program_over_strong_runs_sealed(self):
        self.check_complete_path(None, {'base_recipe_id': 'strong-regressor-v1', 'model_parameters': {'bags': 1},
            'feature_program': [{'op': 'missing', 'name': 'x_missing', 'columns': ['x_numeric']}]})

    def test_custom_blend_of_owned_recipes_runs_sealed(self):
        self.check_complete_path(BLEND_SOURCE, {'strong_weight': 0.7})


ORIGIN_SCRIPT = '''
import sys
from tools.runtime_guard import GuardViolation, PipelineGuard
guard = PipelineGuard(sys.argv[1]); guard.active = True
import importlib.util
spec = importlib.util.spec_from_file_location('candidate', sys.argv[1])
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
    module.run()
except GuardViolation as error:
    print('origin=' + error.origin)
guard.active = False
'''


class GuardOriginTests(unittest.TestCase):
    """A refusal names the nearest owned caller, so environment defects are not candidate verdicts."""

    def origin(self, source):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'candidate.py'
            path.write_text(source)
            done = subprocess.run([sys.executable, '-c', ORIGIN_SCRIPT, str(path)], cwd=ROOT,
                                  capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-3000:])
        return done.stdout.strip().splitlines()[-1]

    def test_unprepared_owned_backend_is_an_environment_refusal(self):
        # Simulates the pre-fix worker: strong loads LightGBM lazily inside fit.
        source = ('import numpy as np\nimport pandas as pd\n'
                  'def run():\n'
                  '    from agent.ml import build_pipeline\n'
                  '    X = pd.DataFrame({"a": np.arange(80.0)}); y = X["a"]*2\n'
                  '    task = {"task_kind": "regression", "target_columns": ["t"], "metric": {"metric_id": "rmse"}}\n'
                  '    build_pipeline("strong-regressor-v1", task, {"bags": 1}).fit(X, y)\n')
        self.assertEqual(self.origin(source), 'origin=owned_runtime')

    def test_generated_loader_call_is_a_candidate_refusal(self):
        source = 'import ctypes, ctypes.util\ndef run():\n    ctypes.CDLL(ctypes.util.find_library("c"))\n'
        self.assertEqual(self.origin(source), 'origin=generated_source')


class FailureScopeTests(unittest.TestCase):
    def test_scope_uses_codes_not_prose(self):
        import json
        from agent.errors import failure_scope
        self.assertEqual(failure_scope(json.dumps({'code': 'GUARD_ENVIRONMENT_BLOCKED', 'message': 'x'})), 'execution_environment')
        self.assertEqual(failure_scope({'code': 'GUARD_POLICY_BLOCKED'}), 'candidate')
        self.assertEqual(failure_scope({'code': 'JOB_TIMEOUT'}), 'resource_limit')
        self.assertEqual(failure_scope('Generated pipeline blocked audit event ctypes.dlopen'), 'unknown')
        self.assertEqual(failure_scope(None), 'unknown')
