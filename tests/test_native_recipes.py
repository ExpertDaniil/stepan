import io
import unittest
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from agent.catalog import Catalog
from agent.config import Config
from agent.ml import build_pipeline


class NativeRecipeTests(unittest.TestCase):
    def test_both_tasks_fit_unknown_categories_and_serialize(self):
        X=pd.DataFrame({'category':['a','b',None,'c']*30,'numeric':[1.,2.,np.nan,4.]*30})
        query=pd.DataFrame({'category':['new',None],'numeric':[np.nan,5.]})
        for kind,suffix in [('regression','regressor'),('multiclass_classification','classifier')]:
            catalog=Catalog(kind,Config())
            for family in ('hist-native','lightgbm'):
                rid=family+'-'+suffix+'-v1'
                self.assertIn(rid,catalog.recipes,'Install the pinned requirements.runtime.txt before this runtime check')
                with self.subTest(recipe=rid):
                    params=catalog.normalize(rid,{'max_iter' if family=='hist-native' else 'n_estimators':20})
                    model=build_pipeline(rid,{'task_kind':kind},params)
                    with threadpool_limits(1):
                        model.fit(X,np.arange(120.) if kind=='regression' else ['A','B','A','B']*30)
                        expected=model.predict(query)
                        stream=io.BytesIO();joblib.dump(model,stream);stream.seek(0)
                        np.testing.assert_array_equal(expected,joblib.load(stream).predict(query))
                    self.assertEqual(model.categorical_,[True,False])

    def test_high_cardinality_has_no_ordinal_semantics(self):
        X=pd.DataFrame({'category':[str(i) for i in range(300)],'numeric':np.arange(300.)})
        model=build_pipeline('hist-native-regressor-v1',{'task_kind':'regression'},{'max_iter':20})
        with threadpool_limits(1):model.fit(X,np.arange(300.))
        self.assertEqual(model.categorical_,[False,False])
        self.assertIn('category',model.frequencies_)
