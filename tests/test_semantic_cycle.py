from copy import deepcopy
import unittest
from agent.research_memory import semantic_projection
from agent.research_feedback import compact_memory
from agent.semantic_planning import validation_scenario
from agent.errors import AgentError
from tests.helpers import Harness


class SemanticCycleTests(unittest.TestCase):
    def test_conflicting_hypotheses_keep_both_sources_after_compaction(self):
        scope={'run_id':'r','task_spec_id':'t','dataset_id':'d','validation_id':'v'}
        def update(statement):
            return {'topic':'row linkage','kind':'hypothesis','columns':['group'],
                    'statement':statement,'evidence_ids':[]}
        records=semantic_projection([('decision-a',{'semantic_scope':scope,'semantic_updates':[update('Repeated rows may be the same object.')]}),
                                    ('decision-b',{'semantic_scope':scope,'semantic_updates':[update('Repeated rows may share a rounded descriptor.')]})])
        view=compact_memory({'semantic_records':records},max_bytes=6000)
        alternatives=view['conflicts'][0]['alternatives']
        self.assertEqual({x['source_decision_artifact_id'] for x in alternatives},{'decision-a','decision-b'})
        self.assertTrue(all(x['kind']=='hypothesis' and x['verification']=='model_statement_unverified' for x in alternatives))

    def test_future_scenario_requires_matching_quoted_evidence_and_protocol(self):
        h=Harness('regression')
        try:
            quote='Predict future observations using historical periods.'
            h.artifacts.write_text('task/task.md',quote,'task_description')
            proposal={'strategy':'temporal_holdout','parameters':{'column':'period',
                'scenario':{'deployment':'future_observations','quote':quote,'time_column':'period'}}}
            result=validation_scenario(proposal,h.spec,h.artifacts)
            self.assertEqual(result['deployment'],'future_observations')
            proposal['strategy']='random_holdout'
            with self.assertRaises(AgentError):validation_scenario(proposal,h.spec,h.artifacts)
            proposal['strategy']='temporal_holdout'
            proposal['parameters']['scenario']['quote']='invented quotation'
            with self.assertRaises(AgentError):validation_scenario(proposal,h.spec,h.artifacts)
        finally:h.close()

    def test_absent_scenario_remains_unknown(self):
        h=Harness('regression')
        try:
            result=validation_scenario({'parameters':{}},h.spec,h.artifacts)
            self.assertEqual(result['deployment'],'unknown')
        finally:h.close()
