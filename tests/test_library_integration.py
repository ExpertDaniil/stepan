"""Real CPU fits of the approved libraries; synthetic data, no provider calls."""
from copy import deepcopy
from contextlib import redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from agent.catalog import Catalog
from agent.config import Config, ROOT
from agent.contracts import Schemas
from agent.duration import DurationModel
from agent.ml import build_pipeline, enforce_threads
from tests.helpers import Harness


class SchemaBackendTests(unittest.TestCase):
    def test_reference_siblings_and_combinator_siblings_are_enforced(self):
        schemas = Schemas()
        cases = [({'$ref': '#/$defs/ResearchAction', 'properties': {'action': {'const': 'stop'}}}, {}),
                 ({'anyOf': [{'type': 'integer'}, {'type': 'string'}], 'enum': [1]}, 2),
                 ({'oneOf': [{'type': 'integer'}, {'type': 'string'}], 'enum': [1]}, 2)]
        # Both values satisfy the reference; its sibling distinguishes them.
        from tests.test_adaptive_research import action
        schemas._check(action('stop'), cases[0][0], '$')
        cases[0] = (cases[0][0], action('catalog'))
        for schema, value in cases:
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                schemas._check(value, schema, '$')

    def test_integer_tokens_and_finite_extensible_maps_remain_strict(self):
        schemas = Schemas()
        for value in (True, 3.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                schemas._check(value, {'type': 'integer'}, '$')
        schemas._check(3, {'type': 'integer'}, '$')
        with self.assertRaises(ValueError):
            schemas._check({'anything': [float('inf')]}, {'type': 'object'}, '$')

    def test_schema_references_never_retrieve_network_resources(self):
        schemas = Schemas()
        with patch('urllib.request.urlopen', side_effect=AssertionError('Network retrieval')) as network:
            with self.assertRaisesRegex(ValueError, 'local registry'):
                schemas._check(1, {'$ref': 'https://invalid.example/schema'}, '$')
        network.assert_not_called()


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lightgbm', 'xgboost')),
                     'Install the approved requirements.runtime.txt before library integration checks')
class BoosterIntegrationTests(unittest.TestCase):
    def parameters(self, catalog, recipe):
        params = {'n_estimators': 20}
        if recipe.startswith('lightgbm-'):
            params.update(num_leaves=7, min_child_samples=2)
        else:
            params.update(max_depth=2)
        return catalog.normalize(recipe, params)

    def test_cpu_models_fit_mixed_missing_features_and_preserve_class_labels(self):
        rng = np.random.default_rng(78)
        frame = pd.DataFrame({'measure': rng.normal(size=90), 'group': ['a', 'b', None]*30})
        frame.loc[0, 'measure'] = np.nan
        predict_frame = pd.DataFrame({'measure': [.3, np.nan, -.2], 'group': ['unseen', None, 'a']})
        labels = np.array(['third', 'first', 'second']*30)
        for kind, target in [('classification', labels), ('regression', rng.normal(size=90))]:
            catalog = Catalog(kind, Config())
            suffix = 'classifier' if kind == 'classification' else 'regressor'
            for family in ('lightgbm', 'xgboost'):
                with self.subTest(kind=kind, family=family), threadpool_limits(limits=1):
                    recipe = family+'-'+suffix+'-v1'
                    params = self.parameters(catalog, recipe)
                    model = clone(build_pipeline(recipe, {'task_kind': kind}, {**params, '_cpu_threads': 2}))
                    enforce_threads(model, 1)
                    self.assertTrue(all(value == 1 for key, value in model.get_params().items()
                                        if key.split('__')[-1] == 'n_jobs'))
                    model.fit(frame, target)
                    predictions = model.predict(predict_frame)
                    self.assertEqual(predictions.shape, (3,))
                    estimator = model.named_steps['model']
                    if kind == 'classification':
                        self.assertTrue(set(predictions) <= set(labels))
                        self.assertEqual(list(estimator.classes_), ['first', 'second', 'third'])
                        probabilities = model.predict_proba(predict_frame)
                        self.assertEqual(probabilities.shape, (3, 3))
                        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
                    else:
                        self.assertTrue(np.isfinite(predictions).all())
                    settings = (estimator.estimator_ if family == 'xgboost' and kind == 'classification'
                                else estimator).get_params()
                    self.assertEqual(settings['device'] if family == 'xgboost' else settings['device_type'], 'cpu')
                    self.assertEqual(settings['n_jobs'], 1)

    def test_duration_uses_each_family_history_and_tree_shape(self):
        duration = DurationModel(Config(time_budget_seconds=300))
        duration.rows = {'train': 2000, 'refit': 3000}
        duration.features = 12
        for family, shape, heavier in [('lightgbm', {'num_leaves': 7}, {'num_leaves': 31}),
                                       ('xgboost', {'max_depth': 2}, {'max_depth': 5})]:
            plan = {'recipe_id': family+'-regressor-v1', 'model': {'family': family,
                    'hyperparameters': {'n_estimators': 100, **shape}}, 'resources': {'cpu_threads': 1}}
            duration.observe(plan, 'train', 12, fit_seconds=10)
            larger = deepcopy(plan)
            larger['model']['hyperparameters'].update(heavier)
            self.assertGreater(duration.estimate(larger), duration.estimate(plan))
            larger['model']['hyperparameters'].update(n_estimators=200)
            self.assertGreater(duration.estimate(larger), duration.estimate(plan))
            self.assertGreater(duration.estimate(plan, 'refit'), duration.estimate(plan))
        self.assertEqual(set(duration.samples), {'lightgbm', 'xgboost'})

    def test_native_guard_keeps_library_loading_and_unknown_symbols_blocked(self):
        import lightgbm.basic
        import xgboost.core
        from tools.runtime_guard import PipelineGuard
        guard = PipelineGuard(Path(__file__))
        # Exercise the policy directly without activating a process-wide hook.
        guard.active = True
        try:
            guard.check('ctypes.dlsym', (lightgbm.basic._LIB, 'LGBM_DatasetCreateFromMat'))
            guard.check('ctypes.dlsym', (xgboost.core._LIB, 'XGBGetGlobalConfig'))
            for event, args in [('ctypes.dlopen', ('anything.so',)),
                                ('ctypes.dlsym', (object(), 'system')), ('ctypes.cdata', (123,)),
                                ('socket.connect', (None,)), ('subprocess.Popen', (None,))]:
                with self.subTest(event=event), self.assertRaises(PermissionError):
                    guard.check(event, args)
        finally:
            guard.active = False

    def test_guarded_feature_workers_train_and_refit_with_both_boosters(self):
        h = Harness('regression')
        try:
            catalog = Catalog('regression', h.config)
            source = h.run / 'features.py'
            source.write_text('from agent.feature_engineering import make_feature_pipeline\n'
                              'def make_pipeline(task_spec, params):\n'
                              '    return make_feature_pipeline(task_spec, params)\n', encoding='utf-8')
            fit_path, valid_path = h.run / 'fit.json', h.run / 'valid.json'
            fit_path.write_text(json.dumps(list(range(60))))
            valid_path.write_text(json.dumps(list(range(60, 90))))
            for family in ('lightgbm', 'xgboost'):
                recipe = family+'-regressor-v1'
                params = {'base_recipe_id': recipe, 'model_parameters': self.parameters(catalog, recipe),
                          'feature_program': [{'op': 'missing', 'name': 'measure_missing', 'columns': ['x_numeric']}]}
                for phase in ('train', 'refit'):
                    with self.subTest(family=family, phase=phase):
                        name = family+'-'+phase
                        job = {'job_id': name, 'phase': phase, 'task_spec': h.spec,
                               'train_path': str(h.task.parent / 'train.csv'), 'test_path': str(h.task.parent / 'test.csv'),
                               'train_indices_path': str(fit_path), 'valid_indices_path': str(valid_path),
                               'code_path': str(source), 'guard_generated': True, 'hyperparameters': params,
                               'cpu_threads': 1, 'seed': 42, 'memory_limit_mb': 2048, 'timeout_seconds': 30,
                               'prediction_path': str(h.run / (name+'.csv')), 'model_path': str(h.run / (name+'.joblib')),
                               'model_snapshot_max_bytes': 0, 'result_path': str(h.run / (name+'-result.json'))}
                        job_path = h.run / (name+'-job.json')
                        job_path.write_text(json.dumps(job), encoding='utf-8')
                        completed = subprocess.run([sys.executable, '-m', 'agent.worker', str(job_path)],
                                                   cwd=ROOT, capture_output=True, text=True, timeout=40)
                        self.assertEqual(completed.returncode, 0, completed.stderr)
                        self.assertTrue(json.loads(Path(job['result_path']).read_text())['success'])
                        result = pd.read_csv(job['prediction_path'])
                        self.assertEqual(len(result), 30 if phase == 'train' else 12)
                        self.assertIn('features_fitted', completed.stdout)
                        self.assertTrue(np.isfinite(result.iloc[:, -1]).all())
        finally:
            h.close()

    def test_adaptive_catalogue_runs_reach_independent_evaluation_and_frozen_reconstruction(self):
        from agent.llm.gateway import MockTransport
        from agent.runtime import main
        from scripts.make_demo import make_demo
        from scripts.reconstruct_solution import reconstruct
        from tests.test_adaptive_research import action
        original = MockTransport.complete
        for family in ('lightgbm', 'xgboost'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                task = make_demo(root / 'task', 'regression', rows=120, test_rows=16)
                catalog = Catalog('regression', Config())
                recipe = family+'-regressor-v1'
                calls = []
                def complete(transport, request, timeout, context):
                    if context['metadata']['schema_id'] != 'ResearchAction-v1':
                        return original(transport, request, timeout, context)
                    calls.append(1)
                    value = (action('catalog', recipe_id=recipe, hyperparameters=self.parameters(catalog, recipe))
                             if len(calls) == 1 else action('stop', hyperparameters={}))
                    return {'choices': [{'message': {'role': 'assistant', 'content': json.dumps(value)},
                                         'finish_reason': 'stop'}],
                            'usage': {'prompt_tokens': 100, 'completion_tokens': 100}}
                settings = {'AUTOML_PROFILE': 'development', 'LLM_MODE': 'mock', 'RESEARCH_MODE': 'adaptive',
                    'TIME_BUDGET_SECONDS': '120', 'GRAPH_ENABLED': '1', 'GRAPH_REFINEMENT_ENABLED': '0',
                    'GRAPH_PEERS': '1', 'GRAPH_TRAIN_WORKERS': '1', 'MAX_WORKERS': '1', 'TRAIN_CPU_THREADS': '1',
                    'CATALOG_WARMUP_EXPERIMENTS': '0', 'LLM_MAX_CALLS': '8', 'LOG_LEVEL': 'ERROR',
                    'MODEL_SNAPSHOT_MAX_BYTES': '0', 'GRAPH_MIN_FREE_DISK_BYTES': '0', 'FEATURE_RESEARCH_ENABLED': '0'}
                with patch.dict(os.environ, settings), patch.object(MockTransport, 'complete', complete), \
                        redirect_stderr(io.StringIO()) as stderr:
                    code = main([str(task), '--work-dir', str(root/'runs')])
                self.assertEqual(code, 0, stderr.getvalue()[-7000:])
                run = next((root/'runs').glob('run-*'))
                with sqlite3.connect(run/'state.sqlite') as db:
                    rows = [(eid, state) for eid, state, plan in db.execute('SELECT experiment_id,state,plan FROM experiments')
                            if json.loads(plan)['recipe_id'] == recipe]
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0][1], 'evaluated', stderr.getvalue()[-7000:])
                    self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(), [])
                    self.assertEqual(db.execute("SELECT count(*) FROM experiments WHERE state IN ('queued','running')").fetchone()[0], 0)
                output = pd.read_csv(task.parent/'submission.csv')
                self.assertEqual(len(output), 16)
                self.assertTrue(np.isfinite(output.label).all())
                rebuilt = root/'reconstructed'
                report = reconstruct(run, rows[0][0], rebuilt, task.parent)
                # Execute its frozen refit, rather than trusting the presence of files.
                command = report['run_command']
                replay = subprocess.run([sys.executable, *command[1:]], cwd=rebuilt,
                                        capture_output=True, text=True, timeout=40)
                self.assertEqual(replay.returncode, 0, replay.stderr)
                environment = json.loads((rebuilt/'environment.json').read_text())
                self.assertIn(family, environment['package_versions'])
                self.assertTrue(np.isfinite(pd.read_csv(rebuilt/'candidate.csv').label).all())
