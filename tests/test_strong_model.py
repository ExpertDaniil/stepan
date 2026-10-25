"""Strong recipe: fit-only entity history, inner-checked modes and class decisions."""
import unittest

import numpy as np
import pandas as pd

from agent.strong import EntityHistory, StrongTabularModel, TableEncoder, discover_entity

TEMPORAL = {'strategy': 'temporal_holdout', 'seed': 0, 'column': 'period'}
REGRESSION = {'task_kind': 'regression', 'metric': {'metric_id': 'rmse'}}


def entity_panel(entities=600, periods=5, seed=0, persistent=True):
    """Rows are entity-periods; the target is mostly a stable entity level."""
    rng = np.random.RandomState(seed)
    size = rng.choice([1000, 2000, 5000, 9000, 12000], entities).astype(float)+rng.randint(0, 997, entities)
    kind = rng.choice(['a', 'b', 'c', 'd'], entities)
    level = rng.gamma(2.0, 30.0, entities)
    rows = []
    for period in range(1, periods+1):
        present = rng.rand(entities) < 0.8
        weather = rng.normal(size=1)[0]
        for e in np.flatnonzero(present):
            base = level[e] if persistent else rng.gamma(2.0, 30.0)
            rows.append({'size': size[e], 'kind': kind[e], 'period': period, 'weather': weather+rng.normal(scale=.1),
                         'noise_feature': rng.normal(), 'y': base+3*weather+rng.normal(scale=5)})
    return pd.DataFrame(rows)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))


class EntityHistoryTests(unittest.TestCase):
    def test_history_uses_only_strictly_earlier_periods(self):
        frame = entity_panel(entities=200, periods=4)
        X, y = frame.drop(columns='y'), frame['y'].copy()
        history = EntityHistory(['size', 'kind'], 'period').fit(X, y)
        before = history.transform(X)
        changed = y.copy()
        changed[frame['period'] >= 3] += 1000.0   # Perturb the current and later targets.
        after = EntityHistory(['size', 'kind'], 'period').fit(X, changed).transform(X)
        rows = frame['period'] <= 3
        pd.testing.assert_frame_equal(before[rows], after[rows])
        self.assertTrue((before.loc[frame['period'] == 1, '__history_count'] == 0).all())

    def test_unseen_key_has_no_history(self):
        frame = entity_panel(entities=50, periods=3)
        X, y = frame.drop(columns='y'), frame['y']
        history = EntityHistory(['size', 'kind'], 'period').fit(X, y)
        new = X.iloc[:3].copy()
        new['size'] = -1.0
        new['period'] = 9
        result = history.transform(new)
        self.assertTrue((result['__history_count'] == 0).all())
        self.assertTrue(result['__history_last'].isna().all())


class DiscoveryTests(unittest.TestCase):
    def test_recurring_key_is_found_without_names(self):
        frame = entity_panel()
        found = discover_entity(frame.drop(columns='y'), frame['y'], 'period')
        self.assertIsNotNone(found)
        self.assertEqual(set(found['columns']), {'size', 'kind'})
        self.assertLess(found['within_share'], 0.2)

    def test_no_key_when_entities_carry_no_stable_level(self):
        frame = entity_panel(persistent=False)
        frame['size'] = np.random.RandomState(3).rand(len(frame))  # Row-unique measurements, no identity.
        found = discover_entity(frame.drop(columns='y'), frame['y'], 'period')
        self.assertIsNone(found)


class StrongRegressionTests(unittest.TestCase):
    def fit_predict(self, frame, policy):
        fit, test = frame[frame['period'] < 5], frame[frame['period'] == 5]
        model = StrongTabularModel(task_spec=REGRESSION, parameters={'learning_rate': 0.1}, cpu_threads=2, seed=0,
                                   validation_policy=policy)
        model.fit(fit.drop(columns='y'), fit['y'])
        return model, rmse(test['y'], model.predict(test.drop(columns='y')))

    def test_history_is_used_when_entities_persist_and_beats_base(self):
        frame = entity_panel()
        model, error = self.fit_predict(frame, TEMPORAL)
        self.assertEqual(model.report_['history_mode'], 'joint')
        self.assertEqual(len(model.bag_models_), 3)
        _, base_error = self.fit_predict(frame, {'strategy': 'random_holdout', 'seed': 0, 'column': None})
        self.assertLess(error, 0.8*base_error)

    def test_non_temporal_protocol_never_uses_history(self):
        model, _ = self.fit_predict(entity_panel(), {'strategy': 'random_holdout', 'seed': 0, 'column': None})
        self.assertEqual(model.mode_, 'base')
        self.assertIsNone(model.report_['entity'])


class StrongClassificationTests(unittest.TestCase):
    def test_labels_and_aligned_probabilities(self):
        rng = np.random.RandomState(0)
        n = 3000
        X = pd.DataFrame({'a': rng.normal(size=n), 'b': rng.normal(size=n), 'c': rng.choice(['x', 'y', 'z'], n)})
        score = X['a']+0.5*(X['c'] == 'z')
        y = np.where(score > 1.6, 'rare', np.where(score > 0, 'mid', 'common')).astype(str)
        spec = {'task_kind': 'multiclass_classification', 'metric': {'metric_id': 'f1_macro'}}
        model = StrongTabularModel(task_spec=spec, parameters={}, cpu_threads=2, seed=0,
                                   validation_policy={'strategy': 'stratified_holdout', 'seed': 0, 'column': None})
        model.fit(X, y)
        probabilities = model.predict_proba(X.iloc[:50])
        self.assertEqual(probabilities.shape, (50, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(set(model.predict(X)) <= {'rare', 'mid', 'common'})
        decision = model.report_['decision']
        if not decision['accepted']:
            self.assertTrue(np.all(model.offsets_ == 0))

    def test_accuracy_metric_keeps_plain_argmax(self):
        rng = np.random.RandomState(1)
        X = pd.DataFrame({'a': rng.normal(size=800)})
        y = np.where(X['a'] > 0, 'p', 'n')
        spec = {'task_kind': 'binary_classification', 'metric': {'metric_id': 'accuracy'}}
        model = StrongTabularModel(task_spec=spec, parameters={}, cpu_threads=1, seed=0).fit(X, y)
        self.assertNotIn('decision', model.report_)
        self.assertEqual(model.report_['class_weight_power'], 0.0)


class EncoderTests(unittest.TestCase):
    def test_unknown_levels_frequencies_and_dates(self):
        fit = pd.DataFrame({'cat': ['a', 'b', 'a', None], 'num': [1, 2, None, 4],
                            'when': ['2020-01-01', '2020-02-03', '2021-05-06 10:00:00', '2020-01-01'],
                            'big': [str(i) for i in range(4)]})
        encoder = TableEncoder().fit(fit)
        encoder.frequencies_['big'] = {'0': .25}  # Mark as high cardinality for the check.
        encoder.categories_.pop('big', None)
        out = encoder.transform(pd.DataFrame({'cat': ['c'], 'num': ['x'], 'when': ['2022-03-04'], 'big': ['9']}))
        self.assertTrue(pd.isna(out['cat'].iloc[0]))
        self.assertTrue(np.isnan(out['num'].iloc[0]))
        self.assertEqual(out['when__year'].iloc[0], 2022.0)
        self.assertEqual(out['big'].iloc[0], 0.0)
        self.assertIn('when', encoder.dates_)


if __name__ == '__main__':
    unittest.main()
