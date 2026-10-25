from copy import deepcopy
from dataclasses import replace
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from agent.config import Config
from agent.contracts import canonical
from agent.ensemble import CrossFittedEnsemble,select_weights
from agent.specialists import role_for,normalize_proposal,FACTORY_SOURCE,SourceFeatures,make_specialist_pipeline
from agent.neural import TabularNetwork
from agent.errors import AgentError
from tests.graph_helpers import GraphHarness
from tests.helpers import Harness


class SpecialistTests(unittest.TestCase):
    def test_five_contexts_share_protocol_and_keep_two_training_workers(self):
        h=GraphHarness(research_mode='adaptive',ensemble_research_enabled=True,graph_peers=5,graph_hard_decisions=False)
        try:
            self.assertEqual(len(h.g.heads(h.root)),5)
            protocols=set()
            for i in range(5):
                peer='researcher-'+str(i+1)
                ticket=h.issue(peer)
                context=h.g.artifacts.read_json(ticket['work_order']['context_id'])
                self.assertEqual(context['metadata']['specialist']['researcher_id'],peer)
                user=json.loads(context['messages'][1]['content'])
                self.assertEqual(user['profile_summary']['specialist']['researcher_id'],peer)
                protocols.add(user['task_contract']['protocol_sha256'])
                self.assertLess(len(canonical(context).encode()),h.controller.config.graph_context_bytes)
            self.assertEqual(len(protocols),1)
            self.assertEqual(h.controller.config.max_workers,2)
        finally:h.close()

    def test_numeric_tuning_cannot_escape_executor_and_normalization_is_idempotent(self):
        h=Harness('regression')
        try:
            from agent.catalog import Catalog
            catalog=Catalog('regression',h.config)
            role=role_for('researcher-4',catalog)
            context={'metadata':{'specialist':role},'messages':[{}, {'content':json.dumps({'profile_summary':{}})}]}
            action={'action':'catalog','recipe_id':role['recipe_id'],'hyperparameters':{'alpha':999.},'code':None}
            with self.assertRaises(AgentError):normalize_proposal(action,context,catalog,h.artifacts)
            action['hyperparameters']={}
            normalize_proposal(action,context,catalog,h.artifacts)
            self.assertEqual(action['hyperparameters']['model_parameters']['alpha'],1.)
            before=deepcopy(action);normalize_proposal(action,context,catalog,h.artifacts)
            self.assertEqual(action,before)
            with self.assertRaises(AgentError):normalize_proposal({'action':'tune','tuning':{}},context,catalog,h.artifacts)
        finally:h.close()

    def test_source_features_keep_oof_fit_transform_and_are_serializable(self):
        spec={'task_kind':'regression','target_column':'y','excluded_features':[],
              '_fit_context':{'strategy':'random_holdout'},'allowed_labels':[]}
        source='from agent.learned_features import target_statistics\ndef make_features(task_spec):\n    return target_statistics(task_spec, columns=["g"])\n'
        X=pd.DataFrame({'g':['a']*40,'x':np.arange(40.)});y=np.arange(40.)
        encoder=SourceFeatures(source,spec)
        result=encoder.fit_transform(X,y)
        changed=y.copy();changed[0]=100000
        other=clone(encoder).fit_transform(X,changed)
        self.assertEqual(result['target_stats__0'].iloc[0],other['target_stats__0'].iloc[0])
        stream=io.BytesIO();joblib.dump(encoder,stream);stream.seek(0)
        pd.testing.assert_frame_equal(encoder.transform(X),joblib.load(stream).transform(X))


class EnsembleTests(unittest.TestCase):
    def spec(self,classification=False):
        return {'task_kind':'multiclass_classification' if classification else 'regression',
            'target_column':'y','excluded_features':[],'allowed_labels':['A','B','C'] if classification else [],
            'metric':{'metric_id':'f1_macro' if classification else 'rmse','direction':'maximize' if classification else 'minimize',
                      'prediction_kind':'labels' if classification else 'values','parameters':{}},
            'output':{'prediction_kind':'labels' if classification else 'values'},
            '_fit_context':{'strategy':'stratified_holdout' if classification else 'random_holdout','protocol_sha256':'frozen'}}

    def members(self,classification=False):
        suffix='classifier-v1' if classification else 'regressor-v1'
        return [{'base_recipe_id':'linear-'+suffix,'model_parameters':{'C':1.} if classification else {'alpha':1.}},
                {'base_recipe_id':'hist-native-'+suffix,'model_parameters':{'max_iter':20,'min_samples_leaf':5}}]

    def test_weights_recover_complementary_errors_without_requiring_architecture_names(self):
        truth=np.arange(50.)
        p=np.column_stack([truth+5,truth-5])
        w,report=select_weights(p,truth,self.spec())
        np.testing.assert_allclose(w,[.5,.5],atol=1e-5)
        self.assertLess(report['blend_score'],1e-5)

    def test_both_tasks_crossfit_full_pipelines_and_roundtrip(self):
        rng=np.random.RandomState(42)
        X=pd.DataFrame({'x':rng.normal(size=180),'z':rng.normal(size=180),'category':['a','b',None]*60})
        for classification in (False,True):
            y=np.where(X.x<-.5,'A',np.where(X.x>.5,'C','B')) if classification else (X.x*3+X.z).to_numpy()
            model=CrossFittedEnsemble(self.spec(classification),self.members(classification),folds=3)
            with threadpool_limits(1):
                clone(model).fit(X,y)
                model.fit(X,y);before=model.predict(X.iloc[:10])
                stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0)
                np.testing.assert_array_equal(before,joblib.load(stream).predict(X.iloc[:10]))
                if classification:np.testing.assert_allclose(model.predict_proba(X).sum(axis=1),1.)
            self.assertFalse(model.report_['confirmation_used'])
            self.assertEqual(model.report_['protocol_sha256'],'frozen')
            self.assertEqual(model.report_['oof_rows'],len(X))

    def test_temporal_oof_does_not_invent_first_period_predictions(self):
        X=pd.DataFrame({'x':np.arange(150.),'period':np.repeat(np.arange(5),30)})
        spec=self.spec();spec['_fit_context'].update(strategy='temporal_holdout',column='period')
        with threadpool_limits(1):
            model=CrossFittedEnsemble(spec,self.members(),folds=3).fit(X,np.arange(150.))
        self.assertLess(model.report_['oof_rows'],len(X))


class NeuralTests(unittest.TestCase):
    def test_cpu_transformer_and_mlp_fit_unknown_missing_and_serialize(self):
        X=pd.DataFrame({'x':np.arange(60.)/60,'category':['a','b',None]*20})
        for architecture in ('transformer','mlp'):
            model=TabularNetwork(regression=True,parameters={'architecture':architecture,'width':16,'blocks':1,'epochs':2,'batch_size':32})
            model.fit(X,np.sin(X.x).to_numpy())
            query=pd.DataFrame({'x':[np.nan,2.],'category':['unseen',None]})
            before=model.predict(query)
            self.assertTrue(np.isfinite(before).all())
            stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0)
            np.testing.assert_array_equal(before,joblib.load(stream).predict(query))
            self.assertEqual(model.report_['weights'],'from_scratch')
