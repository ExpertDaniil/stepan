"""Synthetic regression checks. Execution is deliberately left to the user."""
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import clone

from agent.feature_engineering import FeatureProgram, make_feature_pipeline
from agent.feature_schema import validate_program
from agent.research_actions import ACTION_SCHEMA
from agent.contracts import Schemas


class FeatureProgramTests(unittest.TestCase):
    def test_custom_tuning_override_preserves_the_feature_program_and_input_parameters(self):
        params = {'base_recipe_id': 'linear-regressor-v1', 'model_parameters': {'alpha': 1.},
                  'feature_program': [{'op': 'missing', 'name': 'm', 'columns': ['a']}], 'model__alpha': 4.}
        original = deepcopy(params)
        pipeline = make_feature_pipeline({'task_kind': 'regression', 'target_column': 'y', 'excluded_features': []}, params)
        self.assertEqual(pipeline.named_steps['model'].named_steps['model'].alpha, 4.)
        self.assertEqual(pipeline.named_steps['features'].program, params['feature_program'])
        self.assertEqual(params, original)

    def test_numeric_group_keys_survive_csv_dtype_changes(self):
        model = FeatureProgram([{'op': 'group_mean', 'name': 'm', 'columns': ['x'], 'group_by': 'g'}])
        model.fit(pd.DataFrame({'g': [1., 1., 2.], 'x': [2., 6., 10.]}))
        result = model.transform(pd.DataFrame({'g': [1, 2], 'x': [100, 200]}))
        self.assertEqual(result['m'].tolist(), [4., 10.])

    def test_group_statistics_are_inductive_and_batch_invariant(self):
        fit = pd.DataFrame({'kind': ['a', 'a', 'b'], 'size': [2., 6., 10.]})
        program = [{'op': 'group_mean', 'name': 'mean_size', 'columns': ['size'], 'group_by': 'kind'},
                   {'op': 'ratio', 'name': 'relative', 'columns': ['size', 'mean_size']},
                   {'op': 'group_rank', 'name': 'rank', 'columns': ['size'], 'group_by': 'kind'}]
        model = FeatureProgram(program).fit(fit)
        before = deepcopy(model.statistics_)
        test = pd.DataFrame({'kind': ['a', 'a', 'new'], 'size': [4., 1000., 5.]})
        output = model.transform(test)
        self.assertEqual(output['mean_size'].iloc[0], 4.)
        self.assertEqual(output['relative'].iloc[0], 1.)
        self.assertEqual(output['rank'].iloc[0], .5)
        self.assertTrue(np.isnan(output['mean_size'].iloc[2]))
        pd.testing.assert_frame_equal(output.iloc[:1], model.transform(test.iloc[:1]))
        self.assertEqual(before['mean_size'], model.statistics_['mean_size'])

    def test_composed_categories_and_unseen_frequency(self):
        frame = pd.DataFrame({'a': ['x', 'x', 'y'], 'b': ['u', 'u', 'v']})
        model = FeatureProgram([{'op': 'cross', 'name': 'ab', 'columns': ['a', 'b']},
                                {'op': 'frequency', 'name': 'frequency', 'columns': ['ab']}])
        model = clone(model).fit(frame)
        result = model.transform(pd.DataFrame({'a': ['x', 'new'], 'b': ['u', 'v']}))
        self.assertAlmostEqual(result['frequency'].iloc[0], 2/3)
        self.assertEqual(result['frequency'].iloc[1], 0)

    def test_missing_zero_denominators_and_log_domain(self):
        frame = pd.DataFrame({'a': [1., -2., np.nan], 'b': [0., 2., 1.]})
        result = FeatureProgram([{'op': 'ratio', 'name': 'r', 'columns': ['a', 'b']},
                                 {'op': 'log1p', 'name': 'l', 'columns': ['a']},
                                 {'op': 'missing', 'name': 'm', 'columns': ['a']}]).fit_transform(frame)
        self.assertTrue(np.isnan(result['r'].iloc[0]))
        self.assertTrue(np.isnan(result['l'].iloc[1]))
        self.assertEqual(result['m'].tolist(), [0., 0., 1.])

    def test_target_overwrite_and_forward_dependencies_rejected(self):
        for program in [
            [{'op': 'sum', 'name': 'x', 'columns': ['a']}],
            [{'op': 'sum', 'name': 'z', 'columns': ['target']}],
            [{'op': 'sum', 'name': 'z', 'columns': ['later']}]]:
            with self.assertRaises(ValueError):
                validate_program(program, ['a', 'x'], ['target'])

    def test_schema_matches_runtime_declaration(self):
        schemas = Schemas()
        self.assertEqual(schemas.schema['$defs']['ResearchAction'], ACTION_SCHEMA)

    def test_null_optional_group_is_canonical_but_real_invalid_group_needs_repair(self):
        from agent.research_actions import normalize_action
        original = [{'op': 'target_mean', 'name': 'prior', 'columns': ['a'], 'group_by': None}]
        action = normalize_action({'action': 'features', 'hypothesis': 'Measure a prior', 'evidence_ids': [],
                                   'feature_program': original})
        self.assertNotIn('group_by', action['feature_program'][0])
        self.assertIn('group_by', original[0])
        Schemas().validate(action, 'ResearchAction')
        self.assertEqual(validate_program(original), action['feature_program'])
        with self.assertRaisesRegex(ValueError, r'feature_program\[0\].group_by: unsupported for target_mean.*keys in columns'):
            validate_program([{**original[0], 'group_by': 'a'}])
        with self.assertRaisesRegex(ValueError, r'feature_program\[0\].group_by: group_mean requires'):
            validate_program([{'op': 'group_mean', 'name': 'avg', 'columns': ['a'], 'group_by': None}])
