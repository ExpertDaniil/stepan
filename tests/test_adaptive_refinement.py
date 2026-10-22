"""A shared final base is evidence, not an implicit adaptive search restriction."""
import unittest

from agent.errors import AgentError
from tests.graph_helpers import GraphHarness


class AdaptiveRefinementTests(unittest.TestCase):
    def start(self, h):
        return h.g.invoke('G15', {'source_experiment_id': h.g.store.best(),
            'expected_graph_revision': h.g.revision(), 'reason': 'Synthetic state-machine transition'})

    def test_adaptive_final_proposal_can_change_family_and_empirical_features(self):
        h = GraphHarness(research_mode='adaptive', graph_refinement_enabled=True)
        try:
            h.result('researcher-1', {'alpha': 1}, 10)
            source = h.g.store.best()
            self.start(h)
            ticket = h.issue()
            reply, payload = h.submit(ticket, {'target_transform': 'shifted_log1p'}, recipe='hist-gradient-regressor-v1')
            self.assertEqual(reply['lead_submission']['disposition'], 'created')
            plan = payload['proposal']['plan']
            self.assertEqual(plan['parent_ids'], [source])
            self.assertEqual(plan['model']['hyperparameters']['target_transform'], 'shifted_log1p')
            self.assertEqual(h.g.branch(h.root)['constraints'], {'required_capability_ids': [], 'forbidden_capability_ids': []})
        finally:
            h.close()

    def test_legacy_final_proposals_still_keep_family_and_features(self):
        for recipe, params, expected in [
                ('hist-gradient-regressor-v1', {}, 'REFINEMENT_FAMILY_CHANGED'),
                ('linear-regressor-v1', {'target_transform': 'shifted_log1p'}, 'REFINEMENT_COMPONENT_CHANGED')]:
            with self.subTest(recipe=recipe):
                h = GraphHarness(research_mode='catalog', graph_refinement_enabled=True)
                try:
                    h.result('researcher-1', {'alpha': 1}, 10)
                    self.start(h)
                    with self.assertRaises(AgentError) as caught:
                        h.submit(h.issue(), params, recipe=recipe)
                    self.assertEqual(caught.exception.code, expected)
                finally:
                    h.close()

    def test_adaptive_refinement_preserves_explicit_controlled_branch_constraints(self):
        h = GraphHarness(research_mode='adaptive', graph_refinement_enabled=True)
        try:
            fork, _, _ = h.make_fork()
            children = h.g.invoke('G02', fork)
            branch = children['accept_branch']['branch_id']
            h.result('researcher-1', {'alpha': .1, 'target_transform': 'shifted_log1p'}, 7, branch=branch)
            before = h.g.branch(branch)['constraints']
            result = self.start(h)
            self.assertEqual(result['branch']['branch_id'], branch)
            self.assertEqual(result['branch']['constraints'], before)
            with self.assertRaises(AgentError) as caught:
                h.submit(h.issue(branch=branch), {'alpha': .2, 'target_transform': 'identity'})
            self.assertEqual(caught.exception.code, 'REQUIRED_CAPABILITY_MISSING')
        finally:
            h.close()


if __name__ == '__main__':
    unittest.main()
