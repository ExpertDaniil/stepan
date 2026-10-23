"""Injected state-machine observations, not estimates of ML quality."""
from copy import deepcopy
import json
import unittest

from agent.context import knowledge_focus
from agent.errors import AgentError
from agent.research_actions import ACTION_SCHEMA, validate_research_notes
from agent.research_memory import build_research_memory, idea_id
from agent.researcher import Researcher
from tests.graph_helpers import GraphHarness
from tests.test_adaptive_research import action


def notes():
    return {'idea_id': None, 'question': 'Can this joint pipeline improve the current base?',
        'mechanism': 'Change feature handling and regularization together.',
        'decision_reason': 'This is affordable and can address the observed error pattern.',
        'previous_result_interpretation': 'Individual feature effects are still unknown.',
        'alternatives': [{'idea_id': None, 'question': 'Can frequency features help?',
            'mechanism': 'Rare category prevalence can describe a useful interaction.',
            'next_test': 'Fit category frequencies on fit rows inside the pipeline.',
            'why_deferred': 'Try the current cheaper candidate first.', 'evidence_ids': [], 'knowledge_ids': []}]}


class ResearchMemoryTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness('regression', research_mode='adaptive')
        self.g = self.h.g
        self.researcher = Researcher(self.g.bus, None, self.g.artifacts, self.g.spec, self.g.catalog, self.g.config, self.g.log)

    def tearDown(self):
        self.h.close()

    def save(self, ticket, plan, research_notes, *, stop=False, legacy=False):
        context = deepcopy(self.g.artifacts.read_json(ticket['work_order']['context_id']))
        if legacy:
            context['metadata'].pop('graph_binding', None)
        value = action('stop' if stop else 'catalog', research_notes=research_notes)
        output = self.g.artifacts.write_json('research/unit-output-'+ticket['work_order']['work_order_id']+'.json', value, 'llm_output')
        self.researcher.save_decision(value, plan, ticket['researcher_request']['payload'], context,
                                      {'output_artifact_id': output['artifact_id']})

    def measured_decision(self, research_notes, *, alpha=100):
        ticket = self.h.issue()
        payload = self.h.proposal(ticket, {'alpha': alpha, 'target_transform': 'shifted_log1p'})
        self.save(ticket, payload['proposal']['plan'], research_notes)
        reply = self.g.invoke('G04', payload)
        self.h.commit(reply, 55)
        return ticket, payload, reply

    def test_bad_joint_trial_does_not_close_individual_alternatives(self):
        _, payload, reply = self.measured_decision(notes())
        memory = self.g.research_memory(self.h.root, [])
        self.assertEqual(len(memory['open_ideas']), 2)
        self.assertTrue(all(item['status'] == 'open' for item in memory['open_ideas']))
        decision = memory['decisions'][0]
        self.assertEqual(decision['observations'][0]['experiment_id'], reply['lead_submission']['experiment_id'])
        self.assertEqual(decision['observations'][0]['primary_metric']['value'], 55)
        self.assertEqual(decision['configuration']['fingerprint_core']['validation_id'], payload['proposal']['plan']['validation_id'])
        self.assertIn('individual component effects remain unidentified', decision['conclusion_scope'])
        alternative = next(i for i in memory['open_ideas'] if i['last_decision_role'] == 'deferred')
        self.assertEqual(alternative['tested_experiment_ids'], [])
        self.assertIn('frequency', knowledge_focus(memory))

    def test_old_alternative_survives_new_decisions_that_do_not_repeat_it(self):
        self.measured_decision(notes())
        followup = notes();followup['alternatives'] = []
        followup['question'] = 'Another candidate'
        self.measured_decision(followup, alpha=1)
        memory = self.g.research_memory(self.h.root, [])
        self.assertTrue(any(i['question'] == 'Can frequency features help?' for i in memory['open_ideas']))
        ticket = self.h.issue()
        body = self.g.artifacts.read_json(ticket['work_order']['context_id'])
        carried = json.loads(body['messages'][1]['content'])['profile_summary']['research_memory']
        self.assertEqual(carried, memory)

    def test_stop_keeps_alternatives_without_fabricating_an_experiment(self):
        ticket = self.h.issue()
        self.save(ticket, None, notes(), stop=True)
        proposal = {'proposal_id': 'stop-unit', 'researcher_id': 'researcher-1', 'decision': 'stop', 'plan': None,
                    'stop_reason': 'No useful affordable action now'}
        self.g.invoke('G04', {'work_order_id': ticket['work_order']['work_order_id'], 'proposal': proposal,
                             'manifest': None, 'lineage': None})
        memory = self.g.research_memory(self.h.root, [])
        self.assertEqual(memory['decisions'][0]['observations'], [])
        self.assertIsNone(memory['decisions'][0]['configuration'])
        self.assertEqual(len(memory['open_ideas']), 1)

    def test_unsubmitted_decision_does_not_cross_graph_visibility(self):
        ticket = self.h.issue()
        payload = self.h.proposal(ticket)
        self.save(ticket, payload['proposal']['plan'], notes())
        self.assertEqual(self.g.research_memory(self.h.root, [])['decisions'], [])
        self.assertEqual(build_research_memory(self.g.store, self.g.artifacts, [])['decisions'], [])

    def test_sibling_decisions_do_not_enter_another_branch(self):
        fork, _, _ = self.h.make_fork()
        children = self.g.invoke('G02', fork)
        reject = children['reject_branch']['branch_id']
        accept = children['accept_branch']['branch_id']
        self.g.invoke('G08', {'branch_id': reject, 'expected_branch_revision': 0,
                             'action': 'resume', 'reason': 'Inspect sibling decision visibility'})
        ticket = self.h.issue(branch=reject)
        payload = self.h.proposal(ticket, {'alpha': .123})
        self.save(ticket, payload['proposal']['plan'], notes())
        reply = self.g.invoke('G04', payload)
        self.h.commit(reply, 55)
        self.assertTrue(self.g.research_memory(reject, [])['decisions'])
        self.assertEqual(self.g.research_memory(accept, [])['decisions'], [])

    def test_a_different_seed_cannot_supply_the_decisions_observation(self):
        ticket = self.h.issue()
        original = self.h.proposal(ticket, {'alpha': 1}, seed=42)
        self.save(ticket, original['proposal']['plan'], notes(), legacy=True)
        other = self.h.proposal(ticket, {'alpha': 1}, seed=43)
        reply = self.g.invoke('G04', other)
        self.h.commit(reply, 55)
        eid = reply['lead_submission']['experiment_id']
        memory = build_research_memory(self.g.store, self.g.artifacts, [eid])
        self.assertEqual(memory['decisions'][0]['observations'], [])

    def test_legacy_projection_retains_unmeasured_decision_without_a_score(self):
        ticket = self.h.issue()
        payload = self.h.proposal(ticket)
        self.save(ticket, payload['proposal']['plan'], notes(), legacy=True)
        memory = build_research_memory(self.g.store, self.g.artifacts, [])
        self.assertEqual(len(memory['open_ideas']), 2)
        self.assertEqual(memory['decisions'][0]['observations'], [])
        self.assertIn('No verified measurement', memory['decisions'][0]['conclusion_scope'])

    def test_notes_cannot_forge_idea_or_evidence_references(self):
        self.g.schemas._check(action('catalog', research_notes=notes()), ACTION_SCHEMA, '$')
        self.assertEqual(ACTION_SCHEMA, self.g.schemas.schema['$defs']['ResearchAction'])
        for field, value in [('idea_id', 'research-idea-forged'), ('evidence_ids', ['made-up']),
                             ('knowledge_ids', ['kg-forged'])]:
            changed = notes()
            changed['alternatives'][0][field] = value
            with self.subTest(field=field), self.assertRaises(AgentError):
                validate_research_notes(changed, {}, set(), set())
        self.assertEqual(idea_id('Question   A', 'Mechanism'), idea_id('question a', 'mechanism'))
