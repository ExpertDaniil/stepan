import unittest

import numpy as np
import pandas as pd

from agent.errors import AgentError
from agent.feature_schema import validate_program
from agent.feature_search import focus_for_round, seed_programs, validate_focus


class FeatureSearchTests(unittest.TestCase):
    def test_focus_rotates_without_permanent_roles(self):
        self.assertEqual([focus_for_round(0, i, 2) for i in range(2)], [True, False])
        self.assertEqual([focus_for_round(1, i, 2) for i in range(2)], [False, True])
        self.assertTrue(focus_for_round(0, 0, 1))
        self.assertFalse(focus_for_round(1, 0, 1))

    def test_feature_attention_permits_model_search(self):
        user = {'profile_summary': {'feature_focus': True}, 'base_code': 'old'}
        self.assertIsNone(validate_focus({'action': 'tune'}, user))
        validate_focus({'action': 'features'}, user)
        validate_focus({'action': 'stop'}, user)
        validate_focus({'action': 'pipeline', 'code': 'new', 'feature_steps': [{'step': 'group_statistics'}]}, user)

    def test_seeds_depend_on_fit_schema_not_public_names(self):
        frame = pd.DataFrame({'arbitrary_measure': [*range(39), 100000.],
                              'arbitrary_group': ['a', 'b']*20,
                              'another_group': ['x']*39+['y'],
                              'nullable': [np.nan, *range(39)]})
        programs = seed_programs(frame)
        self.assertGreater(len(programs), 3)
        self.assertLessEqual(len(programs), 8)
        for program in programs:
            validate_program(program, frame.columns)
        ops = {step['op'] for program in programs for step in program}
        self.assertTrue({'missing', 'frequency', 'cross', 'relative_to_group', 'group_rank'} <= ops)

    def test_empty_schema_has_no_fabricated_seed(self):
        self.assertEqual(seed_programs(pd.DataFrame()), [])
