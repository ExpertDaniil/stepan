from io import BytesIO
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from threadpoolctl import threadpool_limits

from agent.feature_engineering import FeatureProgram, make_feature_pipeline
from agent.feature_schema import validate_program, operation_facts
from agent.supervised import CrossFitHistory, MacroF1DecisionClassifier, bind_validation_policy


class CheckedClassifier(ClassifierMixin,BaseEstimator):
    def fit(self,X,y):
        self.seen_=set(X.index);self.classes_=np.unique(y)
        return self
    def predict_proba(self,X):
        if self.seen_ & set(X.index):raise AssertionError('In-sample predictions entered decision fitting')
        p=np.clip(.35+.2*X['signal'].to_numpy(),.05,.95)
        return np.column_stack([1-p,p])
    def predict(self,X):return self.classes_[self.predict_proba(X).argmax(axis=1)]


class SupervisedOperatorsTests(unittest.TestCase):
    def test_own_label_and_held_group_labels_cannot_enter_their_encodings(self):
        X=pd.DataFrame({'key':np.repeat(['A','B','C','D'],20),'value':np.arange(80.)})
        y=np.arange(80.)
        for policy in [{'strategy':'random_holdout','seed':42},
                       {'strategy':'group_holdout','seed':42,'column':'key'}]:
            with self.subTest(policy=policy):
                first=CrossFitHistory(['key'],validation_policy=policy).fit_transform(X,y)
                changed=y.copy();changed[0]+=100000
                second=CrossFitHistory(['key'],validation_policy=policy).fit_transform(X,changed)
                np.testing.assert_array_equal(first.iloc[0],second.iloc[0])
                if policy['strategy']=='group_holdout':
                    np.testing.assert_array_equal(first.iloc[:20],second.iloc[:20])
                    self.assertTrue((first['count']==0).all())

    def test_temporal_history_uses_only_earlier_periods(self):
        X=pd.DataFrame({'period':np.repeat([1,2,3],10),'key':['A']*30})
        y=np.repeat([1.,10.,100.],10)
        policy={'strategy':'temporal_holdout','column':'period','seed':42}
        first=CrossFitHistory(['key'],alpha=0,validation_policy=policy).fit_transform(X,y)
        self.assertTrue(first.loc[:9,'mean'].isna().all())
        np.testing.assert_array_equal(first.loc[10:19,'mean'],np.ones(10))
        changed=y.copy();changed[20:]=100000
        second=CrossFitHistory(['key'],alpha=0,validation_policy=policy).fit_transform(X,changed)
        np.testing.assert_array_equal(first,second)

    def test_unseen_keys_use_fit_prior_and_zero_support_without_mutation(self):
        X=pd.DataFrame({'key':['A','A','B','B']});y=np.array([1.,3.,5.,7.])
        model=CrossFitHistory(['key'],alpha=0,validation_policy={'strategy':'random_holdout','seed':1}).fit(X,y)
        query=pd.DataFrame({'key':['new','A',None]})
        result=model.transform(query)
        np.testing.assert_array_equal(result['mean'],[4.,2.,4.])
        np.testing.assert_array_equal(result['count'],[0.,2.,0.])
        stream=BytesIO();joblib.dump(model,stream);stream.seek(0)
        pd.testing.assert_frame_equal(result,joblib.load(stream).transform(query))
        with self.assertRaises(ValueError):CrossFitHistory(['key']).fit_transform(X,y)

    def test_named_target_features_crossfit_inside_a_real_owned_pipeline(self):
        X=pd.DataFrame({'key':np.repeat(['A','B','C','D'],20),'value':np.arange(80.)})
        y=np.tile(np.arange(20.),4)
        program=[{'op':'target_mean','name':'history','columns':['key'],'alpha':10},
                 {'op':'target_count','name':'support','columns':['key']}]
        validate_program(program,X.columns)
        self.assertEqual(operation_facts(program)['operations'][0]['fit_scope'],'fit_only_target_crossfit')
        spec={'task_kind':'regression','target_column':'y','excluded_features':[]}
        params={'base_recipe_id':'hist-native-regressor-v1','model_parameters':{'max_iter':20},'feature_program':program}
        model=bind_validation_policy(make_feature_pipeline(spec,params),{'strategy':'random_holdout','seed':42})
        with threadpool_limits(1):model.fit(X,y);expected=model.predict(X.iloc[:3])
        self.assertEqual(model.named_steps['features'].report_[0]['crossfit']['folds'],5)
        copy=clone(model)
        self.assertEqual(copy.named_steps['features'].validation_policy['strategy'],'random_holdout')
        stream=BytesIO();joblib.dump(model,stream);stream.seek(0)
        with threadpool_limits(1):np.testing.assert_array_equal(joblib.load(stream).predict(X.iloc[:3]),expected)

    def test_macro_f1_offsets_use_out_of_fit_predictions_and_preserve_probabilities(self):
        X=pd.DataFrame({'signal':np.linspace(-1,1,120)})
        y=np.array(['A']*80+['B']*40)
        model=MacroF1DecisionClassifier(CheckedClassifier(),{'strategy':'stratified_holdout','seed':42}).fit(X,y)
        self.assertEqual(model.report_['inner_oof_rows'],120)
        self.assertGreaterEqual(model.report_['inner_adjusted_macro_f1'],model.report_['inner_raw_macro_f1'])
        query=pd.DataFrame({'signal':[-.5,.5]},index=[500,501])
        np.testing.assert_array_equal(model.predict_proba(query),model.estimator_.predict_proba(query))
        stream=BytesIO();joblib.dump(model,stream);stream.seek(0)
        np.testing.assert_array_equal(model.predict(query),joblib.load(stream).predict(query))

    def test_rare_class_support_is_reported_without_fabricated_tuning(self):
        X=pd.DataFrame({'signal':np.arange(20.)});y=np.array(['A']*19+['B'])
        model=MacroF1DecisionClassifier(CheckedClassifier(),{'strategy':'stratified_holdout','seed':42}).fit(X,y)
        self.assertEqual(model.report_['status'],'insufficient_inner_class_support')
        np.testing.assert_array_equal(model.offsets_,[0.,0.])

    def test_history_keys_do_not_collide_and_group_limits_fail_explicitly(self):
        X=pd.DataFrame({'a':['a','ab','a','ab'],'b':['bc','c','bc','c']})
        model=CrossFitHistory(['a','b'],alpha=0).fit(X,np.array([1.,10.,1.,10.]))
        np.testing.assert_array_equal(model.transform(X)['mean'],[1.,10.,1.,10.])
        with self.assertRaises(ValueError):CrossFitHistory(['a','b'],max_groups=1).fit(X,np.arange(4.))
