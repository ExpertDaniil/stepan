"""Top-k equal-weight ensemble: estimator semantics, worker execution and the controller precheck."""
from copy import deepcopy
from types import SimpleNamespace
import json
import time
import unittest

import numpy as np
import pandas as pd

from agent import metrics as registry
from agent.ensemble import ENSEMBLE_SOURCE, AveragingEnsemble, combine, make_ensemble_pipeline, vote
from tests.helpers import Harness


def spec(metric_id, task_kind, labels=()):
    return {'task_kind': task_kind, 'target_column': 'y', 'target_columns': ['y'], 'allowed_labels': list(labels),
            'excluded_features': [], 'metric': registry.resolve({'metric_id': metric_id}, task_kind, list(labels))}


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = pd.DataFrame({'a': rng.normal(size=400), 'b': rng.normal(size=400), 'k': rng.choice(['p', 'q'], 400)})

    def members(self, regression):
        kind = 'regressor' if regression else 'classifier'
        return [{'kind': 'owned', 'recipe_id': f'linear-{kind}-v1', 'hyperparameters': {}, 'experiment_id': 'exp-1'},
                {'kind': 'owned', 'recipe_id': f'hist-gradient-{kind}-v1', 'hyperparameters': {'max_iter': 30}, 'experiment_id': 'exp-2'}]

    def test_regression_is_the_mean_of_independently_built_members(self):
        from agent.ml import build_pipeline
        y = self.X.a*3+np.where(self.X.k == 'p', 1, -1)
        task = spec('rmse', 'regression')
        model = make_ensemble_pipeline(task, {'members': self.members(True), '_cpu_threads': 1, '_seed': 7}).fit(self.X, y)
        expected = np.mean([build_pipeline(m['recipe_id'], task, {**m['hyperparameters'], '_seed': 7}).fit(self.X, y).predict(self.X)
                            for m in self.members(True)], axis=0)
        np.testing.assert_allclose(model.predict(self.X), expected)

    def test_labels_vote_and_probabilities_average(self):
        y = np.where(self.X.a+self.X.b > 0, 'up', 'down')
        labels_task = spec('f1_macro', 'binary_classification', ['down', 'up'])
        model = make_ensemble_pipeline(labels_task, {'members': self.members(False)}).fit(self.X, y)
        self.assertEqual(set(model.predict(self.X)), {'down', 'up'})
        np.testing.assert_allclose(model.predict_proba(self.X).sum(axis=1), 1)
        probability_task = spec('log_loss', 'binary_classification', ['down', 'up'])
        model = make_ensemble_pipeline(probability_task, {'members': self.members(False)}).fit(self.X, y)
        np.testing.assert_array_equal(model.predict(self.X), model.classes_[model.predict_proba(self.X).argmax(axis=1)])

    def test_vote_breaks_ties_by_member_rank(self):
        np.testing.assert_array_equal(vote([np.array(['a', 'b', 'c']), np.array(['b', 'b', 'a'])]), ['a', 'b', 'c'])
        np.testing.assert_array_equal(vote([np.array(['a']), np.array(['b']), np.array(['b'])]), ['b'])
        np.testing.assert_allclose(combine(spec('rmse', 'regression'), [np.array([[1.], [3.]]), np.array([[3.], [5.]])]), [[2.], [4.]])

    def test_invalid_members_are_rejected(self):
        for members in ([], [self.members(True)[0]],
                        [*self.members(True)[:1], {'kind': 'owned', 'recipe_id': 'dummy-regressor-v1', 'hyperparameters': {}, 'experiment_id': 'x'}],
                        [*self.members(True)[:1], {'kind': 'owned', 'recipe_id': 'python-pipeline-v1', 'hyperparameters': {}, 'experiment_id': 'x'}]):
            with self.subTest(members=members), self.assertRaises(ValueError):
                make_ensemble_pipeline(spec('rmse', 'regression'), {'members': members})
        with self.assertRaises(ValueError):
            make_ensemble_pipeline(spec('rmse', 'regression'), {'members': self.members(True), 'weights': [1, 2]})


class WorkerExecutionTests(unittest.TestCase):
    def test_ensemble_source_passes_code_checks_and_runs_in_the_worker(self):
        from agent.worker import execute
        from tools.code import validate_code, write_code
        from agent.evaluation import Evaluator
        validate_code(ENSEMBLE_SOURCE)
        h = Harness('regression')
        try:
            split = Evaluator(h.spec, h.artifacts, h.store, h.log).create_protocol({'task_spec_id': h.spec['task_spec_id'],
                'dataset_id': h.spec['dataset_id'], 'proposal': {'strategy': 'random_holdout', 'parameters': {'valid_fraction': .2},
                'seed': 42, 'reason': 'Synthetic ensemble execution'}})
            code = write_code({'researcher_id': 'controller-ensemble', 'filename': 'ensemble.py', 'content': ENSEMBLE_SOURCE,
                               'declared_entrypoint': 'make_pipeline'}, h.artifacts)['artifact']
            members = [{'kind': 'owned', 'recipe_id': 'linear-regressor-v1', 'hyperparameters': {'alpha': 1.0}, 'experiment_id': 'exp-1'},
                       {'kind': 'owned', 'recipe_id': 'hist-gradient-regressor-v1', 'hyperparameters': {'max_iter': 30}, 'experiment_id': 'exp-2'}]
            job = {'job_id': 'unit-ensemble', 'task_spec': h.spec, 'phase': 'train', 'guard_generated': True,
                   'train_path': str(h.task.parent/'train.csv'), 'test_path': str(h.task.parent/'test.csv'),
                   'code_path': str(h.artifacts.resolve(code['artifact_id'])), 'hyperparameters': {'members': members},
                   'seed': 42, 'cpu_threads': 1, 'prediction_path': str(h.run/'validation.csv'), 'model_path': str(h.run/'ensemble.joblib'),
                   'model_snapshot_max_bytes': 0, 'train_indices_path': str(h.artifacts.resolve(split['train_rows_artifact_id'])),
                   'valid_indices_path': str(h.artifacts.resolve(split['valid_rows_artifact_id']))}
            execute(job)
            frame = pd.read_csv(job['prediction_path'])
            self.assertTrue(np.isfinite(frame['prediction']).all())
            job.update(phase='refit', prediction_path=str(h.run/'candidate.csv'))
            execute(job)
            self.assertEqual(len(pd.read_csv(job['prediction_path'])), len(pd.read_csv(h.task.parent/'test.csv')))
        finally:
            h.close()


class ControllerTests(unittest.TestCase):
    """maybe_add_ensemble spends training only when saved predictions already improve."""
    def controller(self, matrices, truth, metric_values):
        from agent.pipeline_search import VersionSearch
        task = spec('rmse', 'regression')
        events, added = [], []
        plans = {eid: {'recipe_id': 'linear-regressor-v1', 'code_artifact_id': 'code-'+eid, 'researcher_id': 'researcher-1',
                       'model': {'family': 'linear', 'hyperparameters': {'alpha': float(i+1)}}, 'plan_id': 'p'+eid, 'parent_ids': []}
                 for i, eid in enumerate(matrices)}
        evaluations = {eid: {'valid': True, 'violations': [], 'primary_metric': {'metric_id': 'rmse', 'direction': 'minimize',
                       'value': metric_values[eid], 'validation_id': 'v', 'dataset_id': 'd', 'fidelity': 'full'}} for eid in matrices}
        db = SimpleNamespace(execute=lambda *_: [(eid,) for eid in matrices])
        store = SimpleNamespace(best=lambda: min(metric_values, key=metric_values.get), db=db, run_id='run',
                                experiment_evaluation=evaluations.get, get=lambda eid: {'plan': plans[eid]})
        evaluator = SimpleNamespace(selection_predictions=lambda eid: ('v', [0, 1, 2, 3], matrices[eid]),
                                    selection_truth=lambda rows: truth)
        config = SimpleNamespace(final_ensemble_size=3, time_budget_seconds=100, ensemble_start_fraction=0.)
        budget = SimpleNamespace(snapshot=lambda: {'search_allowed': True}, started=time.monotonic()-1)
        catalog = SimpleNamespace(recipes={'linear-regressor-v1': {'family': 'linear'}, 'python-pipeline-v1': {'family': 'custom'}})
        graph = SimpleNamespace(branch=lambda _: {'state': 'active'})
        c = SimpleNamespace(store=store, evaluator=evaluator, config=config, budget=budget, catalog=catalog, graph=graph,
                            spec=task, failed_refits=set(), log=SimpleNamespace(event=lambda name, **f: events.append((name, f))),
                            artifacts=SimpleNamespace(resolve=lambda _: None),
                            call=lambda cid, payload, handler, key=None: {'artifact': {'artifact_id': 'code-ensemble'}})
        search = VersionSearch.__new__(VersionSearch)
        search.c, search.versions = c, {'v1': {'initial_experiment_id': 'exp-a', 'best_experiment_id': 'exp-a',
                                               'ticket': {'work_order': {'branch_id': 'b'}}, 'anchor': plans['exp-a']}}
        search.add = lambda plan, ticket, **_: added.append((plan, ticket))
        return search, events, added

    def test_precheck_gates_training_and_builds_a_member_plan(self):
        truth = np.array([0., 0., 0., 0.])
        independent = {'exp-a': np.array([[1.], [-1.], [1.], [-1.]]), 'exp-b': np.array([[-1.], [1.], [-1.], [1.]])}
        search, events, added = self.controller(independent, truth, {'exp-a': 1.0, 'exp-b': 1.0})
        search.maybe_add_ensemble()
        self.assertEqual(len(added), 1)
        plan = added[0][0]
        self.assertEqual(plan['recipe_id'], 'python-pipeline-v1')
        self.assertEqual([m['experiment_id'] for m in plan['model']['hyperparameters']['members']], ['exp-a', 'exp-b'])
        self.assertEqual(plan['feature_steps'][0]['members'], ['exp-a', 'exp-b'])
        self.assertTrue(any(name == 'search.ensemble_proposed' for name, _ in events))
        search.ensemble_retry_at = 0
        search.maybe_add_ensemble()
        self.assertEqual(len(added), 1, 'the same member set is proposed once')

    def test_identical_or_not_better_members_cost_no_training(self):
        truth = np.array([0., 0., 0., 0.])
        same = {'exp-a': np.array([[1.], [-1.], [1.], [-1.]]), 'exp-b': np.array([[1.], [-1.], [1.], [-1.]])}
        search, events, added = self.controller(same, truth, {'exp-a': 1.0, 'exp-b': 1.0})
        search.maybe_add_ensemble()
        self.assertEqual(added, [])
        self.assertEqual(events[-1][1]['reason'], 'fewer_than_two_distinct_members')
        # Distinct (corr 0.98) and within 3%, but the average (RMSE 1.005) loses to the best member (1.0).
        worse = {'exp-a': np.array([[1.], [-1.], [1.], [-1.]]), 'exp-b': np.array([[1.2], [-.8], [.8], [-1.2]])}
        search, events, added = self.controller(worse, truth, {'exp-a': 1.0, 'exp-b': 1.0198})
        search.maybe_add_ensemble()
        self.assertEqual(added, [])
        self.assertEqual(events[-1][1]['reason'], 'average_not_better_on_saved_selection_predictions')


if __name__ == '__main__':
    unittest.main()
