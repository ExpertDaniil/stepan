"""Revisable graph decisions and branch patience, with injected synthetic scores."""
import json
import time
import unittest
from unittest.mock import patch

from agent.config import Config
from agent.errors import AgentError
from agent.scientific_research import episode_allowance
from tests.graph_helpers import GraphHarness


class SoftBranchTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness(graph_hard_decisions=False, graph_branch_min_seconds=600)
        self.g = self.h.g

    def tearDown(self):
        self.h.close()

    def fork(self):
        request, _, _ = self.h.make_fork()
        return self.g.invoke('G02', request)

    def decide(self, branch, assessment=None):
        current = self.g.branch(branch)
        checkpoint = self.g.invoke('G01', {'branch_id': branch,
            'expected_branch_revision': current['branch_revision'], 'reason': 'manual'})['checkpoint']
        return self.g.invoke('G14', {'checkpoint_id': checkpoint['checkpoint_id'],
            'budget': self.g.budget.snapshot(), 'plateau_assessment_id': assessment})['decision']

    def test_production_defaults_allow_longer_revisable_research(self):
        config = Config()
        self.assertFalse(config.graph_hard_decisions)
        self.assertEqual(config.graph_branch_min_seconds, 600)
        self.assertEqual(config.graph_plateau_seconds, 1800)
        self.assertEqual(config.graph_plateau_window, 6)
        self.assertEqual(episode_allowance(config), 1620)

    def test_soft_fork_seeds_different_references_without_banning_a_component(self):
        result = self.fork()
        accept, alternative = result['accept_branch'], result['reject_branch']
        self.assertEqual(accept['constraints'], alternative['constraints'])
        self.assertEqual(alternative['constraints']['forbidden_capability_ids'], [])
        a, b = self.g.heads(accept['branch_id'])[0], self.g.heads(alternative['branch_id'])[0]
        self.assertNotEqual(a['experiment_id'], b['experiment_id'])
        self.g.invoke('G08', {'branch_id': alternative['branch_id'],
            'expected_branch_revision': alternative['branch_revision'], 'action': 'resume', 'reason': 'Test another combination'})
        ticket = self.h.issue(branch=alternative['branch_id'])
        context = self.g.artifacts.read_json(ticket['work_order']['context_id'])
        focus = json.loads(context['messages'][1]['content'])['profile_summary']['branch_focus']
        self.assertFalse(focus['hard_constraint'])
        self.assertEqual(focus['preference'], 'investigate_alternative')
        # A poor use of the same component cannot ban later uses in this line.
        reply, _ = self.h.submit(ticket, {'alpha': 3, 'target_transform': 'shifted_log1p'})
        self.h.commit(reply, 50)
        next_ticket = self.h.issue(branch=alternative['branch_id'])
        next_reply, _ = self.h.submit(next_ticket, {'alpha': .3, 'target_transform': 'shifted_log1p'})
        self.h.commit(next_reply, 7)
        self.assertEqual(self.g.branch(alternative['branch_id'])['constraints']['forbidden_capability_ids'], [])

    def test_explicit_inherited_restrictions_remain_enforced(self):
        with self.g.store.transaction():
            self.g.db.execute('INSERT INTO g_constraints VALUES(?,?,?,?,?)',
                (self.g.run_id, self.h.root, 'feature.numeric_interactions', 'forbidden', self.h.root))
        result = self.fork()
        for branch in (result['accept_branch'], result['reject_branch']):
            self.assertEqual(branch['constraints']['forbidden_capability_ids'], ['feature.numeric_interactions'])
        ticket = self.h.issue(branch=result['accept_branch']['branch_id'])
        with self.assertRaises(AgentError):
            self.h.submit(ticket, {'alpha': 3, 'numeric_interactions': True})

    def test_elective_fork_waits_until_branch_has_time_to_develop(self):
        self.g.config = self.h.controller.config = self.g.context.catalog.config = Config(research_mode='catalog',
            graph_refinement_enabled=False)
        request, _, _ = self.h.make_fork()
        self.assertEqual(self.decide(self.h.root)['action'], 'CONTINUE')
        later = self.g.elapsed()+610000
        with patch.object(self.g, 'elapsed', return_value=later):
            decision = self.decide(self.h.root)
        self.assertEqual(decision['action'], 'FORK')
        self.assertEqual(decision['parameters']['idea_id'], request['idea_id'])

    def test_young_branch_keeps_searching_despite_short_plateau_override(self):
        result = self.fork()
        branch = result['accept_branch']['branch_id']
        for peer, alpha in [('researcher-1', 3), ('researcher-2', 4)]:
            self.h.result(peer, {'alpha': alpha, 'target_transform': 'shifted_log1p'}, 30, branch=branch)
        self.g.policy['plateau'].update(window_full_evaluations=1, min_distinct_solutions=1, min_elapsed_seconds=0)
        b = self.g.branch(branch)
        assessment = self.g.invoke('G07', {'branch_id': branch, 'expected_branch_revision': b['branch_revision']})['assessment']
        self.assertEqual(assessment['status'], 'plateau')
        with patch.object(self.g.budget, 'research_capacity', return_value=100):
            self.assertEqual(self.decide(branch, assessment['assessment_id'])['action'], 'CONTINUE')
        # Patience never forces idle waiting when the branch has no executable work.
        self.h.controller.no_fit_branches = {branch}
        self.h.controller.no_fit_reasons = {branch: 'no_affordable_candidates'}
        self.assertEqual(self.decide(branch)['action'], 'BACKTRACK')

    def test_global_deadline_still_overrides_branch_patience(self):
        self.g.budget.deadline = time.monotonic()
        self.assertEqual(self.decide(self.h.root)['action'], 'FINALIZE')

    def test_one_bad_trial_does_not_archive_an_unresolved_alternative(self):
        result = self.fork()
        alternative = result['reject_branch']['branch_id']
        self.g.invoke('G08', {'branch_id': alternative, 'expected_branch_revision': 0,
            'action': 'resume', 'reason': 'Synthetic alternative trial'})
        self.h.result('researcher-1', {'alpha': 4}, 30, branch=alternative)
        branch = self.g.branch(alternative)
        self.g.invoke('G08', {'branch_id': alternative, 'expected_branch_revision': branch['branch_revision'],
            'action': 'pause', 'reason': 'Save the unresolved alternative'})
        self.g.policy['max_live_branches'] = 2
        self.assertIsNone(self.g.fork_victims(result['accept_branch']['branch_id']))
        self.h.controller.no_fit_branches = {alternative}
        self.assertEqual(self.g.fork_victims(result['accept_branch']['branch_id']), [alternative])

    def test_new_measurement_invalidates_a_prior_plateau_for_switching(self):
        result = self.fork()
        branch = result['accept_branch']['branch_id']
        for peer, alpha in [('researcher-1', 3), ('researcher-2', 4)]:
            self.h.result(peer, {'alpha': alpha, 'target_transform': 'shifted_log1p'}, 30, branch=branch)
        self.g.policy['plateau'].update(window_full_evaluations=1, min_distinct_solutions=1, min_elapsed_seconds=0)
        assessment = self.g.invoke('G07', {'branch_id': branch, 'expected_branch_revision': 0})['assessment']
        self.assertTrue(self.g.branch_has_current_plateau(branch))
        self.h.result('researcher-1', {'alpha': 5, 'target_transform': 'shifted_log1p'}, 5, branch=branch)
        self.assertFalse(self.g.branch_has_current_plateau(branch))
        with patch.object(self.g, 'elapsed', return_value=self.g.elapsed()+610000), \
                patch.object(self.g.budget, 'research_capacity', return_value=100):
            self.assertEqual(self.decide(branch, assessment['assessment_id'])['action'], 'CONTINUE')


if __name__ == '__main__':
    unittest.main()
