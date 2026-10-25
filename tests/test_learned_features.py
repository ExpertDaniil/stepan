from copy import deepcopy
import io
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from agent.learned_features import TargetStatistics, MetricDecisionClassifier, target_statistics
from agent.ml import build_pipeline
from threadpoolctl import threadpool_limits


class LearnedFeatureTests(unittest.TestCase):
    def test_training_encoding_excludes_own_fold_and_refits_inside_cv(self):
        X = pd.DataFrame({'g': ['a']*40, 'x': np.arange(40.)})
        y = np.arange(40.)
        a = TargetStatistics(['g']).fit_transform(X, y)
        modified = y.copy(); modified[0] = 100000
        b = TargetStatistics(['g']).fit_transform(X, modified)
        self.assertEqual(a['target_stats__0'].iloc[0], b['target_stats__0'].iloc[0])
        # Calling fit on an outer subset must discard the full-data history.
        encoder = TargetStatistics(['g']).fit(X, y)
        encoder.fit(X.iloc[:10], y[:10])
        self.assertAlmostEqual(encoder.transform(X.iloc[-1:])['target_stats__0'].iloc[0], 4.5)

    def test_temporal_encodings_use_strictly_past_targets_and_priors(self):
        X = pd.DataFrame({'g': ['a']*6, 'period': [1,1,2,2,3,3]})
        y = np.array([1.,3.,10.,20.,100.,200.])
        a = TargetStatistics(['g'], time_column='period', smoothing=1).fit_transform(X, y)
        self.assertEqual(a['target_stats__0'].iloc[0], 0.)
        self.assertEqual(a['target_stats__0'].iloc[2], 2.)
        changed = y.copy(); changed[2:] += 99999
        b = TargetStatistics(['g'], time_column='period', smoothing=1).fit_transform(X, changed)
        pd.testing.assert_series_equal(a['target_stats__0'].iloc[:4], b['target_stats__0'].iloc[:4])

    def test_class_order_unknown_keys_serialization_and_batch_invariance(self):
        X = pd.DataFrame({'g': ['x','x','y','y'], 'h': ['a|b','a|b','c','c']})
        m = clone(TargetStatistics(['g','h'], classification=True, classes=['01','02','03'])).fit(X, ['01','02','02','02'])
        query = pd.DataFrame({'g':['x','new'], 'h':['a|b','none']})
        before = deepcopy(m.statistics_)
        result = m.transform(query)
        self.assertAlmostEqual(result.iloc[1]['target_stats__0'], .25)
        self.assertAlmostEqual(result.iloc[1]['target_stats__1'], .75)
        self.assertEqual(result.iloc[1]['target_stats__2'], 0.)
        self.assertEqual(result.iloc[1]['target_stats__support'], 0.)
        pd.testing.assert_frame_equal(result.iloc[:1], m.transform(query.iloc[:1]))
        buffer = io.BytesIO(); joblib.dump(m, buffer); buffer.seek(0)
        pd.testing.assert_frame_equal(result, joblib.load(buffer).transform(query))
        self.assertEqual(set(before), set(m.statistics_))

    def test_key_collisions_and_cardinality_limits(self):
        X = pd.DataFrame({'a': ['a','a|b',None,'M'], 'b': ['b|c','c','x','x']})
        encoder = TargetStatistics(['a','b'], smoothing=1).fit(X, [1.,2.,3.,4.])
        self.assertEqual(len(encoder.statistics_), 4)
        with self.assertRaises(ValueError):
            TargetStatistics(['a'], max_groups=1).fit(X, [1.,2.,3.,4.])
        with self.assertRaises(ValueError):
            target_statistics({'target_column':'y','excluded_features':[],'task_kind':'regression'}, ['a'])

    def test_native_histogram_categories_missing_unknown_and_roundtrip(self):
        X = pd.DataFrame({'cat': ['a','b',None,'c']*30, 'x': [1.,2.,np.nan,4.]*30})
        for kind in ('regression','multiclass_classification'):
            spec = {'task_kind':kind}
            params = {'max_iter':20,'min_samples_leaf':2}
            model = build_pipeline('hist-native-regressor-v1' if kind == 'regression' else 'hist-native-classifier-v1', spec, params)
            y = np.arange(120.) if kind == 'regression' else np.array(['A','B','A','B']*30)
            with threadpool_limits(1):
                model.fit(X, y)
                self.assertEqual(model.categorical_, [True,False])
                query = pd.DataFrame({'cat':['new',None], 'x':[np.nan,5.]})
                before = model.predict(query)
                buffer=io.BytesIO();joblib.dump(model,buffer);buffer.seek(0)
                np.testing.assert_array_equal(before,joblib.load(buffer).predict(query))

    def test_class_decision_is_inner_fit_only_and_keeps_probabilities(self):
        rng=np.random.RandomState(7)
        X=pd.DataFrame(rng.normal(size=(400,3)),columns=['a','b','c'])
        y=np.where(X.a+X.b > 1.4,'rare','common')
        model=MetricDecisionClassifier(LogisticRegression(),['common','rare']).fit(X,y)
        np.testing.assert_allclose(model.predict_proba(X),model.model_.predict_proba(X))
        self.assertEqual(len(model.weights_),2)
        self.assertFalse(model.decision_report_['probabilities_calibrated'])
        self.assertLess(model.decision_report_['tuning_rows'],len(X))
