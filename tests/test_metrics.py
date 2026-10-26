"""Metric registry: exact scoring, resolution without aborts, learner proxies, end-to-end runs."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn import metrics as sk

from agent import metrics as registry
from agent.config import ROOT
from agent.errors import AgentError
from agent.evaluation import score_predictions
from scripts.make_demo import make_demo


def spec(metric_id, task_kind, labels=(), **parameters):
    return {'task_kind': task_kind, 'allowed_labels': list(labels),
            'metric': {'metric_id': metric_id, 'direction': registry.get(metric_id).direction,
                       'prediction_kind': registry.get(metric_id).prediction_kind, 'parameters': parameters}}


class RegistryScoringTests(unittest.TestCase):
    """Every implemented metric agrees with an independent sklearn/numpy reference."""
    def setUp(self):
        rng = np.random.default_rng(7)
        self.y = rng.gamma(2.0, 3.0, 200)
        self.p = np.clip(self.y*rng.lognormal(0, .3, 200), 0, None)
        self.labels = ['a', 'b', 'c']
        self.truth = rng.choice(self.labels, 300, p=[.6, .3, .1])
        self.pred = np.where(rng.random(300) < .7, self.truth, rng.choice(self.labels, 300))
        raw = rng.random((300, 3))+np.eye(3)[[self.labels.index(v) for v in self.truth]]
        self.proba = raw/raw.sum(axis=1, keepdims=True)

    def test_regression_metrics_match_references(self):
        y, p = self.y, self.p
        expected = {'rmse': math.sqrt(sk.mean_squared_error(y, p)), 'mse': sk.mean_squared_error(y, p),
                    'mae': sk.mean_absolute_error(y, p), 'medae': sk.median_absolute_error(y, p), 'r2': sk.r2_score(y, p),
                    'rmsle': math.sqrt(sk.mean_squared_log_error(y, p)), 'msle': sk.mean_squared_log_error(y, p),
                    'mape': sk.mean_absolute_percentage_error(y, p),
                    'smape': float(np.mean(2*np.abs(y-p)/(np.abs(y)+np.abs(p))))}
        self.assertEqual(set(expected), {m for m, d in registry.METRICS.items() if d.task_kinds == registry.REGRESSION})
        for metric_id, value in expected.items():
            with self.subTest(metric_id=metric_id):
                self.assertAlmostEqual(score_predictions(spec(metric_id, 'regression'), y, p.reshape(-1, 1)), value, places=12)

    def test_smape_counts_zero_over_zero_as_exact(self):
        self.assertEqual(registry.score(spec('smape', 'regression'), [0., 2.], np.array([0., 2.])), 0.0)

    def test_log_metrics_refuse_negative_values_instead_of_repairing_them(self):
        with self.assertRaises(ValueError):
            registry.score(spec('rmsle', 'regression'), [1., 2.], np.array([-.5, 2.]))

    def test_label_metrics_match_references(self):
        y, p, labels = self.truth, self.pred, self.labels
        expected = {'accuracy': sk.accuracy_score(y, p), 'balanced_accuracy': sk.balanced_accuracy_score(y, p),
                    'mcc': sk.matthews_corrcoef(y, p), 'cohen_kappa': sk.cohen_kappa_score(y, p),
                    'qwk': sk.cohen_kappa_score(y, p, weights='quadratic', labels=labels)}
        for base, function in (('f1', sk.f1_score), ('precision', sk.precision_score), ('recall', sk.recall_score)):
            for average in ('macro', 'weighted', 'micro'):
                expected[f'{base}_{average}'] = function(y, p, average=average, labels=labels, zero_division=0)
        for metric_id, value in expected.items():
            with self.subTest(metric_id=metric_id):
                case = spec(metric_id, 'multiclass_classification', labels, label_order=labels)
                self.assertAlmostEqual(score_predictions(case, y, p.reshape(-1, 1)), value, places=12)

    def test_binary_label_metrics_use_the_positive_label(self):
        y = np.where(self.truth == 'a', 'yes', 'no')
        p = np.where(self.pred == 'a', 'yes', 'no')
        for metric_id, function in (('f1', sk.f1_score), ('precision', sk.precision_score), ('recall', sk.recall_score)):
            for positive in ('yes', 'no'):
                with self.subTest(metric_id=metric_id, positive=positive):
                    case = spec(metric_id, 'binary_classification', ['no', 'yes'], positive_label=positive)
                    self.assertAlmostEqual(registry.score(case, y, p), function(y, p, pos_label=positive), places=12)

    def test_probability_metrics_match_references(self):
        y, proba, labels = self.truth, self.proba, self.labels
        self.assertAlmostEqual(registry.score(spec('log_loss', 'multiclass_classification', labels), y, proba),
                               sk.log_loss(y, proba, labels=labels), places=12)
        self.assertAlmostEqual(registry.score(spec('roc_auc_ovr', 'multiclass_classification', labels), y, proba),
                               sk.roc_auc_score(y, proba, multi_class='ovr', average='macro', labels=labels), places=12)
        binary = np.where(y == 'a', 'pos', 'neg')
        two = np.column_stack([1-proba[:, 0], proba[:, 0]])
        self.assertAlmostEqual(registry.score(spec('roc_auc', 'binary_classification', ['neg', 'pos'], positive_label='pos'), binary, two),
                               sk.roc_auc_score(binary == 'pos', proba[:, 0]), places=12)
        self.assertAlmostEqual(registry.score(spec('pr_auc', 'binary_classification', ['neg', 'pos'], positive_label='pos'), binary, two),
                               sk.average_precision_score(binary == 'pos', proba[:, 0]), places=12)

    def test_ovr_auc_skips_classes_absent_from_a_validation_subset(self):
        y = np.array(['a', 'a', 'b', 'b'])
        proba = np.array([[.8, .1, .1], [.6, .3, .1], [.2, .7, .1], [.3, .6, .1]])
        self.assertEqual(registry.score(spec('roc_auc_ovr', 'multiclass_classification', self.labels), y, proba), 1.0)

    def test_quadratic_kappa_uses_the_declared_order_not_string_order(self):
        y = np.array(['1', '2', '10', '10'])
        p = np.array(['2', '2', '10', '1'])
        numeric = spec('qwk', 'multiclass_classification', ['1', '10', '2'], label_order=['1', '2', '10'])
        self.assertAlmostEqual(registry.score(numeric, y, p),
                               sk.cohen_kappa_score(y, p, weights='quadratic', labels=['1', '2', '10']), places=12)


class ResolutionTests(unittest.TestCase):
    def resolve(self, metric_id, task_kind='regression', labels=(), truth=None, **extra):
        return registry.resolve({'metric_id': metric_id, 'direction': 'maximize', 'prediction_kind': 'values', 'parameters': {}, **extra},
                                task_kind, list(labels), truth)

    def test_aliases_are_unique_and_normalized(self):
        cases = {'Root Mean Squared Error': 'rmse', 'RMSLE': 'rmsle', 'MAPE': 'mape', 'Matthews_CorrCoef': 'mcc',
                 'quadratic weighted kappa': 'qwk', 'AUC': 'roc_auc', 'Average Precision': 'pr_auc', 'R²': 'r2'}
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(registry.ALIASES[registry.normalize_name(name)], expected)

    def test_exact_metric_has_no_declared_record_and_registry_semantics(self):
        metric = registry.resolve({'metric_id': 'rmse', 'direction': 'minimize', 'prediction_kind': 'values', 'parameters': {}},
                                  'regression', [])
        self.assertEqual(metric, {'metric_id': 'rmse', 'direction': 'minimize', 'prediction_kind': 'values', 'parameters': {}})

    def test_disagreeing_direction_is_corrected_and_recorded(self):
        metric = self.resolve('mae')
        self.assertEqual(metric['direction'], 'minimize')
        self.assertTrue(any('direction' in note for note in metric['declared']['notes']))

    def test_unknown_metric_becomes_a_logged_proxy_of_the_same_prediction_kind(self):
        for kind, prediction, expected in (('regression', 'values', 'rmse'), ('multiclass_classification', 'labels', 'f1_macro'),
                                           ('binary_classification', 'probabilities', 'log_loss')):
            with self.subTest(kind=kind):
                metric = registry.resolve({'metric_id': 'pinball_loss_q90', 'direction': 'minimize', 'prediction_kind': prediction,
                                           'parameters': {'alpha': .9}}, kind, ['n', 'y'] if 'binary' in kind else ['a', 'b', 'c'])
                self.assertEqual(metric['metric_id'], expected)
                self.assertEqual(metric['declared']['status'], 'proxy')
                self.assertEqual(metric['declared']['metric_id'], 'pinball_loss_q90')
                self.assertTrue(any('proxy' in note for note in metric['declared']['notes']))
                self.assertTrue(any('alpha' in note for note in metric['declared']['notes']))

    def test_declared_original_wins_over_an_interpreter_proxy_when_implemented(self):
        metric = registry.resolve({'metric_id': 'rmse', 'direction': 'minimize', 'prediction_kind': 'values', 'parameters': {},
                                   'declared': {'metric_id': 'RMSLE', 'status': 'proxy', 'notes': ['interpreter thought unsupported']}},
                                  'regression', [], np.array([1., 2.]))
        self.assertEqual(metric['metric_id'], 'rmsle')
        missing = registry.resolve({'metric_id': 'mae', 'direction': 'minimize', 'prediction_kind': 'values', 'parameters': {},
                                    'declared': {'metric_id': 'tweedie_deviance', 'status': 'proxy', 'notes': []}}, 'regression', [])
        self.assertEqual((missing['metric_id'], missing['declared']['status'], missing['declared']['metric_id']),
                         ('mae', 'proxy', 'tweedie_deviance'))

    def test_binary_names_adapt_to_multiclass_explicitly(self):
        labels = ['a', 'b', 'c']
        for name, expected in (('roc_auc', 'roc_auc_ovr'), ('f1', 'f1_macro'), ('recall', 'recall_macro')):
            with self.subTest(name=name):
                metric = registry.resolve({'metric_id': name, 'direction': 'maximize', 'prediction_kind': 'labels', 'parameters': {}},
                                          'multiclass_classification', labels)
                self.assertEqual((metric['metric_id'], metric['declared']['status']), (expected, 'adapted'))

    def test_monotone_equivalents_are_reported_as_their_implemented_metric(self):
        metric = registry.resolve({'metric_id': 'gini', 'direction': 'maximize', 'prediction_kind': 'probabilities', 'parameters': {}},
                                  'binary_classification', ['0', '1'], np.array(['0', '1', '0']))
        self.assertEqual((metric['metric_id'], metric['declared']['status'], metric['parameters']['positive_label']), ('roc_auc', 'equivalent', '1'))

    def test_positive_label_defaults(self):
        self.assertEqual(registry.default_positive_label(['0', '1'])[0], '1')
        self.assertEqual(registry.default_positive_label(['No', 'Yes'])[0], 'Yes')
        self.assertEqual(registry.default_positive_label(['normal', 'attack'])[0], 'attack')
        self.assertEqual(registry.default_positive_label(['x', 'z'], ['x', 'x', 'z'])[0], 'z')
        self.assertEqual(registry.default_positive_label(['x', 'z'], ['x', 'z'])[0], 'z')
        supplied = registry.resolve({'metric_id': 'f1', 'direction': 'maximize', 'prediction_kind': 'labels',
                                     'parameters': {'positive_label': 'x'}}, 'binary_classification', ['x', 'z'])
        self.assertEqual(supplied['parameters']['positive_label'], 'x')
        self.assertNotIn('declared', supplied)

    def test_qwk_order_defaults_to_numeric_class_names(self):
        metric = registry.resolve({'metric_id': 'qwk', 'direction': 'maximize', 'prediction_kind': 'labels', 'parameters': {}},
                                  'multiclass_classification', ['1', '10', '2'])
        self.assertEqual(metric['parameters']['label_order'], ['1', '2', '10'])

    def test_log_metric_with_negative_training_targets_falls_back_to_rmse(self):
        metric = self.resolve('rmsle', truth=np.array([-1., 2.]))
        self.assertEqual((metric['metric_id'], metric['declared']['status']), ('rmse', 'proxy'))

    def test_invalid_task_kind_still_raises(self):
        with self.assertRaises(AgentError):
            registry.resolve({'metric_id': 'rmse'}, 'ranking', [])

    def test_resolved_metric_satisfies_the_contract_schema(self):
        from agent.contracts import Schemas
        schemas = Schemas()
        for metric in (self.resolve('rmse'), self.resolve('pinball'), self.resolve('mape'),
                       registry.resolve({'metric_id': 'auc'}, 'multiclass_classification', ['a', 'b', 'c'])):
            schemas.validate(metric, 'MetricSpec')


class LearnerProxyTests(unittest.TestCase):
    def test_objectives_and_target_space(self):
        self.assertEqual(registry.default_target_transform(spec('rmsle', 'regression')), 'shifted_log1p')
        self.assertEqual(registry.default_target_transform(spec('mape', 'regression')), 'shifted_log1p')
        self.assertEqual(registry.default_target_transform(spec('rmse', 'regression')), 'identity')
        self.assertEqual(registry.regression_objective(spec('mae', 'regression')), registry.ABSOLUTE)
        self.assertTrue(registry.tunes_decisions(spec('mcc', 'binary_classification', ['a', 'b'])))
        self.assertFalse(registry.tunes_decisions(spec('accuracy', 'binary_classification', ['a', 'b'])))
        self.assertTrue(registry.class_balanced(spec('balanced_accuracy', 'binary_classification', ['a', 'b'])))

    def test_output_domain_clips_only_where_the_metric_requires_it(self):
        np.testing.assert_array_equal(registry.apply_output_domain(spec('rmsle', 'regression'), np.array([[-1.], [2.]])), [[0.], [2.]])
        np.testing.assert_array_equal(registry.apply_output_domain(spec('rmse', 'regression'), np.array([[-1.]])), [[-1.]])

    def test_build_pipeline_applies_metric_defaults_but_keeps_explicit_choices(self):
        from agent.ml import ShiftedLogTargetModel, build_pipeline
        self.assertIsInstance(build_pipeline('lightgbm-regressor-v1', spec('rmsle', 'regression'), {}), ShiftedLogTargetModel)
        self.assertNotIsInstance(build_pipeline('lightgbm-regressor-v1', spec('rmsle', 'regression'), {'target_transform': 'identity'}),
                                 ShiftedLogTargetModel)
        absolute = build_pipeline('lightgbm-regressor-v1', spec('mae', 'regression'), {})
        self.assertEqual(absolute.parameters['objective'], 'l1')
        explicit = build_pipeline('lightgbm-regressor-v1', spec('mae', 'regression'), {'objective': 'huber'})
        self.assertEqual(explicit.parameters['objective'], 'huber')
        self.assertEqual(build_pipeline('hist-native-regressor-v1', spec('medae', 'regression'), {}).parameters['loss'], 'absolute_error')

    def test_catalog_defaults_follow_the_metric_target_space(self):
        from agent.catalog import Catalog
        from agent.config import Config
        log_catalog = Catalog('regression', Config(), spec('rmsle', 'regression'))
        self.assertEqual(log_catalog.normalize('lightgbm-regressor-v1', {})['target_transform'], 'shifted_log1p')
        self.assertEqual(log_catalog.normalize('lightgbm-regressor-v1', {'target_transform': 'identity'})['target_transform'], 'identity')
        plain = Catalog('regression', Config(), spec('rmse', 'regression'))
        self.assertEqual(plain.normalize('lightgbm-regressor-v1', {})['target_transform'], 'identity')

    def test_log_target_learner_scores_better_rmsle_than_identity(self):
        from agent.ml import build_pipeline
        rng = np.random.default_rng(3)
        X = pd.DataFrame({'x': rng.normal(size=1500), 'z': rng.normal(size=1500)})
        y = np.exp(1.5*X['x']+.5*X['z']+rng.normal(0, .2, 1500))
        task = spec('rmsle', 'regression')
        scores = {}
        for transform in ('identity', 'shifted_log1p'):
            model = build_pipeline('lightgbm-regressor-v1', task, {'target_transform': transform, 'n_estimators': 200})
            model.fit(X.iloc[:1000], y.iloc[:1000])
            prediction = registry.apply_output_domain(task, model.predict(X.iloc[1000:]))
            scores[transform] = registry.score(task, y.iloc[1000:], prediction)
        self.assertLess(scores['shifted_log1p'], scores['identity'])

    def test_strong_classifier_tunes_decisions_for_the_declared_metric(self):
        from agent.strong import StrongTabularModel
        rng = np.random.default_rng(5)
        X = pd.DataFrame({'x': rng.normal(size=1200), 'z': rng.normal(size=1200)})
        y = np.where(X['x']+.3*rng.normal(size=1200) > 1.4, 'rare', 'common')
        task = {**spec('mcc', 'binary_classification', ['common', 'rare']), 'task_kind': 'binary_classification'}
        model = StrongTabularModel(task_spec=task, parameters={'bags': 1, 'learning_rate': .1}, seed=1).fit(X, y)
        self.assertEqual(model.report_['decision']['metric_id'], 'mcc')
        self.assertGreater(model.report_['class_weight_power'], 0)
        accuracy = {**spec('accuracy', 'binary_classification', ['common', 'rare']), 'task_kind': 'binary_classification'}
        plain = StrongTabularModel(task_spec=accuracy, parameters={'bags': 1, 'learning_rate': .1}, seed=1).fit(X, y)
        self.assertNotIn('decision', plain.report_)
        self.assertEqual(plain.report_['class_weight_power'], 0)


class TaskResolutionTests(unittest.TestCase):
    """resolve_spec and the interpreter fallback on synthetic tasks."""
    def harness(self, temp, kind, metric):
        from agent.artifacts import Artifacts
        from agent.contracts import Schemas
        from agent.observability.logging import RunLogger
        from tools.data import profile_data
        task = make_demo(Path(temp)/'task', kind, rows=120, test_rows=10, metric=metric)
        log = RunLogger(Path(temp)/'run', 'unit-run', level='ERROR')
        artifacts = Artifacts(Path(temp)/'run', 'unit-run', log)
        profile = profile_data({'candidate_paths': {name.split('.')[0]: str(task.parent/name) for name in
                                                    ('train.csv', 'test.csv', 'sample_submission.csv')}, 'max_report_bytes': 24000}, artifacts)
        return task, artifacts, profile, Schemas()

    def test_unknown_metric_resolves_to_a_proxy_spec_and_logs_it(self):
        from agent.context import demo_metadata
        from agent.task import resolve_spec
        with tempfile.TemporaryDirectory() as temp:
            task, artifacts, profile, schemas = self.harness(temp, 'regression', 'pinball_loss_q90')
            draft = demo_metadata(task.read_text())['draft']
            resolved = resolve_spec(draft, profile, task.parent, artifacts, schemas)
            self.assertEqual(resolved['metric']['metric_id'], 'rmse')
            self.assertEqual(resolved['metric']['declared']['status'], 'proxy')
            events = [json.loads(line) for line in (Path(temp)/'run'/'events.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['event'] == 'task.metric_resolved' and e['level'] == 'WARNING' for e in events))

    def test_unsupported_interpretation_is_reduced_to_a_supervised_draft(self):
        from agent.task import resolve_spec
        from agent.task_interpreter import supervised_reduction
        with tempfile.TemporaryDirectory() as temp:
            for kind, expected in (('regression', 'regression'), ('binary', 'binary_classification'),
                                   ('classification', 'multiclass_classification')):
                with self.subTest(kind=kind):
                    task, artifacts, profile, schemas = self.harness(Path(temp)/kind, kind, None)
                    text = task.read_text()
                    paths = {'train': str(task.parent/'train.csv'), 'test': str(task.parent/'test.csv'),
                             'sample_submission': str(task.parent/'sample_submission.csv')}
                    draft = supervised_reduction({'task_text': text, 'candidate_paths': paths, 'task_root': str(task.parent)},
                                                 profile, artifacts, None, 'ranking protocol')
                    self.assertEqual((draft['task_kind'], draft['target_column']), (expected, 'label'))
                    self.assertIn('ranking protocol', draft['metric']['declared']['notes'][0])
                    resolved = resolve_spec(draft, profile, task.parent, artifacts, schemas)
                    self.assertEqual(resolved['output']['path'], str((task.parent/'submission.csv').resolve()))

    def test_output_format_follows_the_sample_not_the_metric_kind(self):
        from agent.context import demo_metadata
        from agent.task import resolve_spec
        from tools.data import profile_data
        with tempfile.TemporaryDirectory() as temp:
            task, artifacts, profile, schemas = self.harness(Path(temp)/'labels', 'binary', 'roc_auc')
            draft = demo_metadata(task.read_text())['draft']
            self.assertNotIn('prediction_kind', draft['output'])
            resolved = resolve_spec(draft, profile, task.parent, artifacts, schemas)
            self.assertEqual((resolved['metric']['prediction_kind'], resolved['output']['prediction_kind']), ('probabilities', 'labels'))
            task, artifacts, profile, schemas = self.harness(Path(temp)/'scores', 'binary', 'roc_auc')
            sample = pd.read_csv(task.parent/'sample_submission.csv', dtype=str)
            sample['label'] = '0.5'
            sample.to_csv(task.parent/'sample_submission.csv', index=False)
            profile = profile_data({'candidate_paths': {'train': str(task.parent/'train.csv'), 'test': str(task.parent/'test.csv'),
                                                        'sample_submission': str(task.parent/'sample_submission.csv')},
                                    'max_report_bytes': 24000}, artifacts)
            resolved = resolve_spec(demo_metadata(task.read_text())['draft'], profile, task.parent, artifacts, schemas)
            self.assertEqual((resolved['output']['prediction_kind'], resolved['output']['positive_label']), ('probabilities', 'yes'))

    def test_reduction_follows_stated_regression_for_small_integer_targets(self):
        from agent.task_interpreter import supervised_reduction
        with tempfile.TemporaryDirectory() as temp:
            task, artifacts, profile, _ = self.harness(temp, 'regression', None)
            train = pd.read_csv(task.parent/'train.csv', dtype=str, keep_default_na=False)
            train['label'] = [str(i % 5) for i in range(len(train))]
            train.to_csv(task.parent/'train.csv', index=False)
            text = '# Task\nRegression of label. Metric: RMSE.\n'
            from tools.data import profile_data
            profile = profile_data({'candidate_paths': {'train': str(task.parent/'train.csv'), 'test': str(task.parent/'test.csv'),
                                                        'sample_submission': str(task.parent/'sample_submission.csv')},
                                    'max_report_bytes': 24000}, artifacts)
            draft = supervised_reduction({'task_text': text, 'candidate_paths': {}, 'task_root': str(task.parent)},
                                         profile, artifacts, None, 'unsupported')
            self.assertEqual((draft['task_kind'], draft['metric']['metric_id']), ('regression', 'rmse'))

    def test_probability_metric_with_numeric_binary_labels_submits_scores(self):
        from agent.context import demo_metadata
        from agent.task import resolve_spec
        from tools.data import profile_data
        with tempfile.TemporaryDirectory() as temp:
            task, artifacts, _, schemas = self.harness(temp, 'binary', 'roc_auc')
            for name in ('train.csv',):
                frame = pd.read_csv(task.parent/name, dtype=str, keep_default_na=False)
                frame['label'] = frame['label'].map({'yes': '1', 'no': '0'})
                frame.to_csv(task.parent/name, index=False)
            sample = pd.read_csv(task.parent/'sample_submission.csv', dtype=str)
            sample['label'] = '0'
            sample.to_csv(task.parent/'sample_submission.csv', index=False)
            profile = profile_data({'candidate_paths': {'train': str(task.parent/'train.csv'), 'test': str(task.parent/'test.csv'),
                                                        'sample_submission': str(task.parent/'sample_submission.csv')},
                                    'max_report_bytes': 24000}, artifacts)
            resolved = resolve_spec(demo_metadata(task.read_text())['draft'], profile, task.parent, artifacts, schemas)
            self.assertEqual((resolved['output']['prediction_kind'], resolved['output']['positive_label']), ('probabilities', '1'))

    def test_mape_with_negative_targets_keeps_metric_losses_in_slices(self):
        from agent import error_slices
        rng = np.random.default_rng(0)
        y = rng.normal(size=900)*10
        data = pd.DataFrame({'x': rng.normal(size=900), 'row_id': np.arange(900)})
        prediction = np.where(data.x > 1, y*3, y*1.1)
        task = {'task_spec_id': 't', 'task_kind': 'regression', 'target_column': 'y', 'target_columns': ['y'],
                'excluded_features': ['row_id'], 'allowed_labels': [], 'metric': registry.resolve({'metric_id': 'mape'}, 'regression', [], y)}
        report = error_slices.analyze(data, y, prediction.reshape(-1, 1), np.arange(900) % 3, task)
        self.assertEqual(report['loss_scale'], 'identity')
        self.assertAlmostEqual(report['oof_metric'], registry.score(task, y, prediction.reshape(-1, 1)))
        self.assertTrue(any(c['column'] in {'x', '__prediction'} for s in report['slices'] for c in s['conditions']))

    def test_reduction_refuses_tables_without_one_requested_target(self):
        from agent.task_interpreter import supervised_reduction
        with tempfile.TemporaryDirectory() as temp:
            task, artifacts, profile, _ = self.harness(temp, 'regression', None)
            sample = pd.read_csv(task.parent/'sample_submission.csv', dtype=str)
            sample['label2'] = '0'
            sample.to_csv(task.parent/'sample_submission.csv', index=False)
            from tools.data import profile_data
            profile = profile_data({'candidate_paths': {'train': str(task.parent/'train.csv'), 'test': str(task.parent/'test.csv'),
                                                        'sample_submission': str(task.parent/'sample_submission.csv')},
                                    'max_report_bytes': 24000}, artifacts)
            self.assertIsNone(supervised_reduction({'task_text': task.read_text(), 'candidate_paths': {}, 'task_root': str(task.parent)},
                                                   profile, artifacts, None, 'two outputs'))


class MetricEndToEndTests(unittest.TestCase):
    """Full mock-mode runs: each declared metric trains, evaluates and publishes a valid CSV."""
    CASES = (('regression', 'rmsle'), ('regression', 'mape'), ('regression', 'mae'), ('binary', 'mcc'), ('binary', 'f1'),
             ('classification', 'qwk'), ('classification', 'roc_auc_ovr'), ('classification', 'balanced_accuracy'),
             ('regression', 'tweedie_deviance'), ('classification', 'top_3_accuracy'))

    def test_declared_metrics_run_end_to_end(self):
        for kind, metric in self.CASES:
            with self.subTest(kind=kind, metric=metric), tempfile.TemporaryDirectory() as temp:
                task = make_demo(Path(temp)/'task', kind, metric=metric)
                runs = Path(temp)/'runs'
                env = {**os.environ, 'LLM_MODE': 'mock', 'TIME_BUDGET_SECONDS': '45', 'MAX_RESEARCH_EXPERIMENTS': '2',
                       'LOG_LEVEL': 'ERROR', 'MAX_WORKERS': '2', 'EXPERIMENT_TIMEOUT_SECONDS': '30',
                       'RESEARCH_MODE': 'catalog', 'CATALOG_WARMUP_EXPERIMENTS': '0'}
                env.pop('PYTHONPATH', None)
                result = subprocess.run([sys.executable, str(ROOT/'solution.py'), str(task), '--llm-mode', 'mock', '--work-dir', str(runs)],
                                        env=env, cwd=temp, text=True, capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr[-4000:])
                events = [json.loads(line) for line in next(runs.glob('run-*/events.jsonl')).read_text().splitlines()]
                self.assertEqual(events[-1]['event'], 'run.finished')
                self.assertNotEqual(events[-1].get('reason'), 'verified_emergency_output', 'constant fallback is not a solution')
                evaluated = [e for e in events if e['event'] == 'experiment.evaluated']
                self.assertTrue(evaluated)
                resolved = json.loads(next(runs.glob('run-*/task/task-spec.json')).read_text())['metric']
                self.assertTrue(all(e['metric_id'] == resolved['metric_id'] for e in evaluated))
                learned = [e for e in evaluated if e['recipe_id'].split('-')[0] != 'dummy']
                self.assertTrue(learned, 'a learned model must be evaluated')
                output = pd.read_csv(task.parent/'submission.csv', dtype=str, keep_default_na=False)
                test = pd.read_csv(task.parent/'test.csv', dtype=str, keep_default_na=False)
                self.assertEqual(output['id'].tolist(), test['id'].tolist())
                if registry.default_target_transform({'task_kind': 'regression', 'metric': resolved}) == 'shifted_log1p':
                    # The default (first fit) is log1p; background search may still try identity as an axis.
                    transforms = [e['hyperparameters'].get('target_transform') for e in learned if 'target_transform' in e['hyperparameters']]
                    self.assertEqual(transforms[0], 'shifted_log1p', 'log metrics train catalog learners on log1p targets by default')
                    self.assertGreater(transforms.count('shifted_log1p'), len(transforms)/2)
                if kind == 'regression':
                    values = output['label'].astype(float)
                    self.assertTrue(np.isfinite(values).all())
                    if registry.get(resolved['metric_id']).output_floor is not None:
                        self.assertGreaterEqual(values.min(), 0)
                else:
                    labels = set(pd.read_csv(task.parent/'train.csv', dtype=str)['label'])
                    self.assertLessEqual(set(output['label']), labels)


if __name__ == '__main__':
    unittest.main()
