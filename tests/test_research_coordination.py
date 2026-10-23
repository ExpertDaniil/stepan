from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.research_coordination import ResearchCoordination, domain_overlap, overlapping_spaces


class ResearchCoordinationTests(unittest.TestCase):
    def test_historical_catboost_boxes_overlap_but_distinct_axes_do_not(self):
        left = {'depth': {'type': 'int', 'low': 5, 'high': 10},
                'iterations': {'type': 'int', 'low': 600, 'high': 4000}}
        right = {'depth': {'type': 'int', 'low': 4, 'high': 8},
                 'iterations': {'type': 'int', 'low': 400, 'high': 2800}}
        self.assertTrue(overlapping_spaces(left, right))
        self.assertFalse(overlapping_spaces({**left, 'interaction': {'type': 'categorical', 'choices': [False, True]}}, right))
        self.assertEqual(domain_overlap({'type': 'categorical', 'choices': [True]},
                                        {'type': 'categorical', 'choices': [1]}), 0)

    def test_unseen_overlap_deferred_and_informed_deliberate_overlap_allowed(self):
        space = {'depth': {'type': 'int', 'low': 4, 'high': 8, 'step': 1, 'log': False}}
        session = {'series_id': 'series-other', 'scope': {'pipeline': {'source': 'same'}},
                   'space': space, 'trials': [{'finished': True}], 'status': 'running'}
        optimizer = SimpleNamespace(active={('branch-a', 'researcher-1'): session},
            has_series=lambda *args: True, _scope=lambda plan, *args: {'pipeline': {'source': plan['source']}})
        log = []
        c = SimpleNamespace(local_search=SimpleNamespace(optuna=optimizer), graph=None, catalog=None,
                            log=SimpleNamespace(event=lambda name, **data: log.append(name)))
        coordinator = ResearchCoordination(c)
        plan = {'researcher_id': 'researcher-2', 'source': 'same', 'plan_id': 'p'}
        intent = {'action': 'tune', 'tuning': {}}
        with patch('agent.tuning_space.normalize_space', return_value=space):
            self.assertEqual(coordinator.review(plan, intent, 'branch-a')['series_id'], 'series-other')
            intent['coordination'] = {'related_series_ids': ['series-other'], 'reason': 'Independent replication for a defined stability check.'}
            self.assertIsNotNone(coordinator.review(plan, intent, 'branch-a'))
            intent['delivered_feedback'] = [{'active_series_ids': ['series-other']}]
            self.assertIsNone(coordinator.review(plan, intent, 'branch-a'))
            self.assertIn('research.overlap_acknowledged', log)
            self.assertIsNone(coordinator.review({**plan, 'source': 'new-feature-source'}, intent, 'branch-a'))
            self.assertIsNone(coordinator.review(plan, intent, 'branch-b'))
