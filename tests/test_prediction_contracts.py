"""Class/column alignment, numeric target shape and immutable refit regressions."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
import time

import numpy as np
import pandas as pd

from agent.context import demo_metadata
from agent.errors import AgentError
from agent.evaluation import Evaluator, score_predictions
from agent.predictions import checked_matrix, predict, targets, validation_columns
from agent.predictions import read_training
from agent.runner import frozen_execution
from agent.budget import Budget
from agent.orchestrator import Orchestrator
from agent.submission import Submission
from agent.task import resolve_spec
from agent.worker import execute
from tests.helpers import Harness
from tools.code import write_code
from tools.data import profile_data


class PredictionContractTests(unittest.TestCase):
    def test_fitted_class_order_and_submission_column_order_are_independent(self):
        spec = {'task_kind': 'multiclass_classification', 'target_column': 'y', 'allowed_labels': ['a', 'b', 'c'],
            'metric': {'prediction_kind': 'probabilities'},
            'output': {'prediction_kind': 'probabilities', 'prediction_columns': ['p_c', 'p_a', 'p_b'], 'class_labels': ['c', 'a', 'b']}}
        model = SimpleNamespace(classes_=np.array(['c', 'a', 'b']), predict_proba=lambda x: np.array([[.2, .5, .3]]))
        np.testing.assert_allclose(predict(model, [0], spec, 'train'), [[.5, .3, .2]])
        np.testing.assert_allclose(predict(model, [0], spec, 'refit'), [[.2, .5, .3]])

    def test_one_binary_probability_column_selects_the_declared_class(self):
        spec = {'target_column': 'y', 'allowed_labels': ['no', 'yes'], 'metric': {'prediction_kind': 'probabilities'},
            'output': {'prediction_columns': ['risk'], 'positive_label': 'yes'}}
        model = SimpleNamespace(classes_=['yes', 'no'], predict_proba=lambda x: np.array([[.8, .2], [.3, .7]]))
        np.testing.assert_allclose(predict(model, [0, 1], spec, 'refit'), [[.8], [.3]])

    def test_missing_fitted_classes_are_zero_filled_without_relabeling_columns(self):
        spec = {'target_column': 'y', 'allowed_labels': ['a', 'b', 'c'], 'metric': {'prediction_kind': 'probabilities'}}
        model = SimpleNamespace(classes_=['c', 'a'], predict_proba=lambda x: np.array([[.25, .75]]))
        np.testing.assert_allclose(predict(model, [0], spec, 'train'), [[.75, 0, .25]])
        del model.classes_
        with self.assertRaises(AgentError):
            predict(model, [0], spec, 'train')

    def test_multi_target_arrays_cannot_be_silently_flattened(self):
        with self.assertRaises(AgentError):
            checked_matrix([1, 2, 3, 4], 2, 2, 'values')
        spec = {'task_kind': 'regression', 'metric': {'metric_id': 'rmse', 'prediction_kind': 'values'}}
        self.assertAlmostEqual(score_predictions(spec, [[0, 0], [0, 0]], np.array([[0, 0], [0, 4]])), 2)

    def test_invalid_probabilities_are_rejected_before_csv_publication(self):
        for values in [[[.6, .6]], [[-.1, 1.1]], [[np.nan, 1]]]:
            with self.subTest(values=values), self.assertRaises(AgentError):
                checked_matrix(values, 1, 2, 'probabilities')


class FrozenExecutionTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()

    def tearDown(self):
        self.h.close()

    def test_refit_rejects_changed_source_or_parameters(self):
        h = self.h
        eid = h.store.reserve(h.plan(), h.code['sha256'])['experiment_id']
        original = h.store.get(eid)
        frozen_execution(h.artifacts, original, h.spec, 'train')
        frozen_execution(h.artifacts, original, h.spec, 'refit')
        changed = deepcopy(original)
        changed['plan']['model']['hyperparameters']['C'] = 999
        with self.assertRaises(AgentError):
            frozen_execution(h.artifacts, changed, h.spec, 'refit')
        path = h.artifacts.resolve(h.code['artifact_id'])
        path.write_text(path.read_text()+'\n# changed\n')
        with self.assertRaises(AgentError):
            frozen_execution(h.artifacts, original, h.spec, 'refit')

    def test_refit_requires_the_training_execution_snapshot(self):
        h = self.h
        eid = h.store.reserve(h.plan(), h.code['sha256'])['experiment_id']
        with self.assertRaises(AgentError) as caught:
            frozen_execution(h.artifacts, h.store.get(eid), h.spec, 'refit')
        self.assertEqual(caught.exception.code, 'MISSING_EXECUTION_SNAPSHOT')

    def test_textual_class_ids_preserve_leading_zeros(self):
        path = self.h.root/'class_ids.csv'
        path.write_text('feature,label\n1,001\n2,002\n')
        frame = read_training(path, {'task_kind': 'multiclass_classification', 'target_column': 'label'})
        self.assertEqual(frame['label'].tolist(), ['001', '002'])

    def test_refit_start_failure_releases_the_lease_before_retrying_a_candidate(self):
        h = self.h
        h.evaluation(.5)
        c = Orchestrator.__new__(Orchestrator)
        c.config, c.spec, c.artifacts, c.store, c.log = h.config, h.spec, h.artifacts, h.store, h.log
        c.graph, c.failed_refits = None, set()
        c.budget = Budget(h.config, time.monotonic(), h.log)
        c.submission = Submission(h.spec, h.artifacts, h.store, h.log)
        c.call = lambda cid, payload, handler, key=None: handler(payload)
        def fail(*args):
            raise AgentError('CANDIDATE_CHANGED', 'Synthetic refit start rejection')
        c.runner = SimpleNamespace(jobs={}, start_refit=fail)
        with self.assertRaises(AgentError):
            c._finalize_best()
        self.assertEqual(c.budget.leases, {})


class MultipleTargetPipelineTests(unittest.TestCase):
    """Small real fits when run locally: parsing → worker → evaluator → CSV validator."""
    def test_two_numeric_targets_keep_their_order_and_are_excluded_from_features(self):
        h = Harness('regression')
        try:
            train_path = h.task.parent/'train.csv'
            train = pd.read_csv(train_path, keep_default_na=False)
            train['second_target'] = 2*pd.to_numeric(train['label'])+3
            train.to_csv(train_path, index=False)
            test = pd.read_csv(h.task.parent/'test.csv', dtype=str, keep_default_na=False)
            pd.DataFrame({'second': 0, 'id': test['id'], 'first': 0}).to_csv(h.task.parent/'sample_submission.csv', index=False)
            draft = demo_metadata(h.task.read_text())['draft']
            draft['target_columns'] = ['label', 'second_target']
            draft['output'].update(columns=['second', 'id', 'first'], prediction_columns=['first', 'second'])
            profile = profile_data({'candidate_paths': {'train': str(train_path), 'test': str(h.task.parent/'test.csv'),
                'sample_submission': str(h.task.parent/'sample_submission.csv')}, 'max_report_bytes': 24000}, h.artifacts)
            spec = resolve_spec(draft, profile, h.task.parent, h.artifacts, h.schemas)
            self.assertEqual(targets(spec), ['label', 'second_target'])
            self.assertIn('second_target', spec['excluded_features'])
            evaluator = Evaluator(spec, h.artifacts, h.store, h.log)
            split = evaluator.create_protocol({'task_spec_id': spec['task_spec_id'], 'dataset_id': spec['dataset_id'],
                'proposal': {'strategy': 'random_holdout', 'parameters': {'valid_fraction': .2}, 'seed': 42, 'reason': 'Synthetic two-target regression'}})
            source = 'from agent.ml import build_pipeline\ndef make_pipeline(task_spec, params):\n    return build_pipeline("linear-regressor-v1", task_spec, params)\n'
            code = write_code({'researcher_id': 'researcher-1', 'filename': 'multi.py', 'content': source,
                               'declared_entrypoint': 'make_pipeline'}, h.artifacts)['artifact']
            job = {'job_id': 'unit-multi', 'task_spec': spec, 'phase': 'train', 'guard_generated': False,
                'train_path': str(train_path), 'test_path': str(h.task.parent/'test.csv'), 'code_path': str(h.artifacts.resolve(code['artifact_id'])),
                'hyperparameters': {'alpha': 1}, 'seed': 42, 'cpu_threads': 1, 'prediction_path': str(h.run/'validation.csv'),
                'model_path': str(h.run/'multi.joblib'), 'model_snapshot_max_bytes': 0,
                'train_indices_path': str(h.artifacts.resolve(split['train_rows_artifact_id'])),
                'valid_indices_path': str(h.artifacts.resolve(split['valid_rows_artifact_id']))}
            execute(job)
            frame = pd.read_csv(job['prediction_path'])
            self.assertEqual(list(frame.columns), ['__row_index', *validation_columns(spec)])
            plan = h.plan(task_spec_id=spec['task_spec_id'], dataset_id=spec['dataset_id'], validation_id=split['validation_id'],
                recipe_id='linear-regressor-v1', code_artifact_id=code['artifact_id'], model={'family': 'linear', 'hyperparameters': {'alpha': 1}})
            eid = h.store.reserve(plan, code['sha256'])['experiment_id']
            prediction = h.artifacts.register(job['prediction_path'], 'validation_predictions', 'text/csv')
            ev = evaluator.evaluate({'experiment_id': eid, 'task_spec_id': spec['task_spec_id'], 'dataset_id': spec['dataset_id'],
                'validation_id': split['validation_id'], 'predictions_artifact_id': prediction['artifact_id'], 'fidelity': 'full'})
            self.assertTrue(ev['evaluation']['valid'])
            job.update(phase='refit', prediction_path=str(h.run/'candidate.csv'))
            execute(job)
            candidate = h.artifacts.register(job['prediction_path'], 'submission', 'text/csv')
            output = pd.read_csv(job['prediction_path'], dtype={'id': str})
            self.assertEqual(list(output.columns), ['second', 'id', 'first'])
            np.testing.assert_allclose(output['second'], output['first']*2+3, atol=1e-8)
            report = Submission(spec, h.artifacts, h.store, h.log).validate_output({'task_spec_id': spec['task_spec_id'],
                'candidate_artifact_id': candidate['artifact_id'], 'reference_test_artifact_id': spec['test_artifact_id']})
            self.assertTrue(report['valid'])
        finally:
            h.close()
