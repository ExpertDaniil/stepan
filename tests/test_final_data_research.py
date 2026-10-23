"""Real feature fit checks and research orientation; no provider calls."""
import json
import unittest

import numpy as np
import pandas as pd

from agent.catalog import Catalog
from agent.config import Config
from agent.errors import AgentError
from agent.feature_engineering import make_feature_pipeline
from agent.ml import build_pipeline
from agent.research_actions import validate_action
from agent.research_landscape import build_landscape
from tests.graph_helpers import GraphHarness
from tests.test_adaptive_research import action
from tests import test_analysis_access as analysis_fixture
from tests.test_analysis_access import query


class ResearchOrientationTests(unittest.TestCase):
    def test_constant_reference_is_not_a_fundable_research_or_tuning_action(self):
        catalog = Catalog('regression', Config(research_mode='adaptive'))
        for kind in ('catalog', 'tune'):
            with self.subTest(kind=kind), self.assertRaises(AgentError) as caught:
                validate_action(action(kind, recipe_id='dummy-regressor-v1'), catalog)
            self.assertEqual(caught.exception.code, 'REFERENCE_ONLY_RECIPE')
        validate_action(action('stop', recipe_id='dummy-regressor-v1', hyperparameters={}), catalog)

    def test_context_excludes_dummy_but_keeps_other_peers_stronger_evidence(self):
        h = GraphHarness('regression', research_mode='adaptive')
        try:
            h.g.context.store = h.g.store
            h.result('researcher-1', {'alpha': 1}, 15)
            other = h.result('researcher-2', {'alpha': .1}, 3)
            context = h.g.artifacts.read_json(h.issue()['work_order']['context_id'])
            user = json.loads(context['messages'][1]['content'])
            recipes = user['profile_summary']['recipes']
            self.assertFalse(any(r['family'] == 'dummy' for r in recipes))
            landscape = user['profile_summary']['research_landscape']
            self.assertEqual(landscape['strongest_measured_learned']['primary_metric']['value'], 3)
            self.assertEqual(user['base_result']['metric']['value'], 15)
            self.assertEqual(landscape['measured_learned_configurations'], 2)
        finally:
            h.close()

    def test_landscape_with_no_learned_measurement_does_not_invent_one(self):
        h = GraphHarness('regression', research_mode='adaptive')
        try:
            report = build_landscape(h.g.store, h.g.artifacts, [])
            self.assertIsNone(report['strongest_measured_learned'])
            self.assertEqual(report['measured_learned_configurations'], 0)
        finally:
            h.close()


class RepresentationQualityTests(unittest.TestCase):
    def test_interaction_pipeline_learns_signal_that_additive_model_cannot(self):
        rng = np.random.default_rng(124)
        data = pd.DataFrame({'arbitrary_a': rng.normal(size=600), 'arbitrary_b': rng.normal(size=600)})
        target = 5*data.arbitrary_a*data.arbitrary_b + rng.normal(scale=.04, size=600)
        spec = {'task_kind': 'regression', 'target_column': 'unseen_answer', 'excluded_features': []}
        params = {'alpha': .01, 'target_transform': 'identity', 'numeric_interactions': False}
        baseline = build_pipeline('linear-regressor-v1', spec, params)
        program = [{'op': 'product', 'name': 'joint_signal', 'columns': ['arbitrary_a', 'arbitrary_b']}]
        candidate = make_feature_pipeline(spec, {'base_recipe_id': 'linear-regressor-v1',
            'model_parameters': params, 'feature_program': program})
        for model in (baseline, candidate):
            model.fit(data.iloc[:400], target.iloc[:400])
        base_error = np.sqrt(np.mean((baseline.predict(data.iloc[400:])-target.iloc[400:])**2))
        feature_error = np.sqrt(np.mean((candidate.predict(data.iloc[400:])-target.iloc[400:])**2))
        self.assertLess(feature_error, .1)
        self.assertLess(feature_error, base_error*.05)
        np.testing.assert_allclose(candidate.predict(data.iloc[400:]),
            np.concatenate([candidate.predict(data.iloc[400:450]), candidate.predict(data.iloc[450:])]))


class MultipleTargetAnalysisTests(unittest.TestCase):
    setUp = analysis_fixture.AnalysisAccessTests.setUp
    tearDown = analysis_fixture.AnalysisAccessTests.tearDown
    service = analysis_fixture.AnalysisAccessTests.service
    def test_every_target_is_excluded_from_inputs_and_filters(self):
        train = pd.DataFrame({'x': [1, 2, 3, 4], 'target': [3, 4, 5, 6], 'other_target': [7, 8, 9, 10]})
        service = self.service(train, train[['x']])
        service.spec['target_columns'] = ['target', 'other_target']
        for request in (query(columns=['other_target']), query(filter={'column': 'other_target', 'value': 7})):
            with self.subTest(query=request), self.assertRaises(AgentError):
                service.analyze(request)
        report = service.analyze(query(include_target=True))['report']
        self.assertEqual([c['column'] for c in report['columns']], ['x', 'target', 'other_target'])
        schema = service.analyze(query('schema', columns=[]))['report']
        self.assertEqual([c['role'] for c in schema['columns']], ['feature', 'target', 'target'])

    def test_controller_seed_frame_contains_only_fit_features(self):
        service = self.service()
        service.spec['excluded_features'] = ['group']
        frame = service.feature_frame()
        self.assertEqual(list(frame), ['x'])
        self.assertLessEqual(len(frame), 20000)
        self.assertLess(frame.x.max(), 29999)
