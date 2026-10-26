"""Paired bootstrap significance and the noise-aware publication gate."""
from types import SimpleNamespace
import unittest

import numpy as np

from agent import metrics as registry
from agent.significance import paired_bootstrap


def spec(metric_id, task_kind='regression', labels=()):
    return {'task_kind': task_kind, 'allowed_labels': list(labels),
            'metric': registry.resolve({'metric_id': metric_id}, task_kind, list(labels))}


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = rng.normal(size=4000)*10
        self.base = self.y+rng.normal(size=4000)*5

    def test_consistent_small_gain_is_significant(self):
        better = self.y+(self.base-self.y)*0.97  # every row a little closer
        verdict = paired_bootstrap(spec('rmse'), self.y, self.base, better)
        self.assertTrue(verdict['significant'], verdict)
        self.assertGreater(verdict['probability_better'], .99)
        self.assertEqual(verdict['method'], 'weighted_row_losses')

    def test_parameter_jitter_is_within_noise(self):
        rng = np.random.default_rng(1)
        jitter = self.base+rng.normal(size=4000)*1.0  # different rows win and lose
        jitter += (self.y-jitter)*0.001                # a tiny systematic tilt
        verdict = paired_bootstrap(spec('rmse'), self.y, self.base, jitter)
        self.assertFalse(verdict['significant'], verdict)

    def test_direction_and_nonadditive_metrics(self):
        labels = ['a', 'b', 'c']
        rng = np.random.default_rng(2)
        truth = rng.choice(labels, 3000)
        base = np.where(rng.random(3000) < .6, truth, rng.choice(labels, 3000))
        better = np.where(rng.random(3000) < .5, truth, base)
        verdict = paired_bootstrap(spec('f1_macro', 'multiclass_classification', labels), truth, base, better)
        self.assertEqual(verdict['method'], 'recomputed_metric')
        self.assertGreater(verdict['gain'], 0)
        self.assertTrue(verdict['significant'])
        worse = paired_bootstrap(spec('f1_macro', 'multiclass_classification', labels), truth, better, base)
        self.assertLess(worse['gain'], 0)
        self.assertFalse(worse['significant'])
        accuracy = paired_bootstrap(spec('accuracy', 'multiclass_classification', labels), truth, base, better)
        self.assertEqual(accuracy['method'], 'weighted_row_losses')
        self.assertTrue(accuracy['significant'])

    def test_maximize_regression_metric(self):
        better = self.y+(self.base-self.y)*0.9
        self.assertTrue(paired_bootstrap(spec('r2'), self.y, self.base, better)['significant'])
        self.assertFalse(paired_bootstrap(spec('r2'), self.y, better, self.base)['significant'])


class NoiseGateTests(unittest.TestCase):
    def controller(self, verdict, finalization=False):
        from agent.orchestrator import Orchestrator
        c = Orchestrator.__new__(Orchestrator)
        events = []
        c.config = SimpleNamespace(noise_aware_publication=True)
        c.store = SimpleNamespace(champion=lambda: {'experiment_id': 'exp-champion'}, best=lambda: 'exp-best')
        c.failed_refits = set()
        c.evaluator = SimpleNamespace(paired_significance=lambda a, b: {**verdict, 'gain': .1, 'interval': [0, .2], 'method': 'unit'})
        c.log = SimpleNamespace(event=lambda name, **fields: events.append((name, fields)))
        if finalization:
            c.finalization_started = True
        return c, events

    def test_insignificant_best_keeps_the_champion_during_search(self):
        c, events = self.controller({'significant': False, 'probability_better': .7})
        self.assertEqual(c.final_candidate(), 'exp-champion')
        self.assertEqual(events[0][0], 'submission.noise_gate')
        self.assertFalse(events[0][1]['significant'])
        c.final_candidate()
        self.assertEqual(len(events), 1, 'one log line per (champion, best) pair')

    def test_significant_best_is_published(self):
        c, _ = self.controller({'significant': True, 'probability_better': .99})
        self.assertEqual(c.final_candidate(), 'exp-best')

    def test_gate_never_overrides_finalization_or_disabled_config(self):
        c, _ = self.controller({'significant': False, 'probability_better': .5}, finalization=True)
        self.assertEqual(c.noise_gate('exp-best'), 'exp-best')
        c, _ = self.controller({'significant': False, 'probability_better': .5})
        c.config.noise_aware_publication = False
        self.assertEqual(c.final_candidate(), 'exp-best')


if __name__ == '__main__':
    unittest.main()
