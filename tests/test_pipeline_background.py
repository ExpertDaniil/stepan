from copy import deepcopy
from types import SimpleNamespace
import threading
import unittest

from agent.pipeline_candidates import declared_settings, normalize_candidate, automatic_space, version_id, execution_ticket
from agent.pipeline_search import VersionSearch
from agent.graph.controller import lineage_for
from agent.errors import AgentError
from tests.graph_helpers import GraphHarness


class PipelineBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness(graph_hard_decisions=False, research_mode='adaptive')
        self.c = self.h.controller
        self.c.graph = self.h.g
        self.c.update_final_reserve = lambda: None
        self.c.record_research_feedback = lambda *args, **kwargs: None

    def tearDown(self):
        self.h.close()

    def test_file_defaults_and_space_are_literal_and_mechanical(self):
        source = "DEFAULT_PARAMS={'alpha':1.0}\nSEARCH_SPACE={'alpha':{'type':'float','low':0.01,'high':10.0,'log':True}}\ndef make_pipeline(task_spec,params):\n    return None\n"
        value = normalize_candidate({'action':'pipeline','code':source})
        self.assertEqual(value['hyperparameters'], {'alpha':1.0})
        self.assertEqual(value['recipe_id'], 'python-pipeline-v1')
        self.assertTrue(value['expected_effect'])
        self.assertIn('alpha', declared_settings(source)['SEARCH_SPACE'])
        with self.assertRaises(AgentError):
            declared_settings('DEFAULT_PARAMS=dict(alpha=1)')
        with self.assertRaises(AgentError):
            declared_settings('def make_pipeline(:')

    def test_background_order_does_not_occupy_researcher(self):
        ticket = self.h.issue()
        payload = self.h.proposal(ticket)
        bg = execution_ticket(self.c, ticket)
        payload['work_order_id'] = bg['work_order']['work_order_id']
        payload['lineage'] = lineage_for(bg['work_order'])
        reply = self.h.g.invoke('G04', payload)
        self.assertEqual(self.h.g.order(ticket['work_order']['work_order_id'])['state'], 'issued')
        self.assertEqual(self.h.g.order(bg['work_order']['work_order_id'])['state'], 'background')
        self.assertEqual(self.h.g.db.execute("SELECT count(*) FROM g_work_orders WHERE state IN ('preparing','issued','submitted')").fetchone()[0], 1)
        outcome = self.h.commit(reply, 3.0)
        self.assertEqual(outcome['head_results'][0]['disposition'], 'kept_background')
        self.assertEqual(self.c.store.best(), reply['lead_submission']['experiment_id'])
        self.assertIn(reply['lead_submission']['experiment_id'], self.h.g.visible_history(self.h.root))

    def test_version_identity_freezes_representation_not_tuning_axes(self):
        ticket = self.h.issue()
        plan = self.h.proposal(ticket)['proposal']['plan']
        space = automatic_space(self.c.catalog, plan, '')
        one = version_id(plan, 'source-hash', space)
        other = deepcopy(plan)
        other['model']['hyperparameters']['alpha'] = 2.0
        self.assertEqual(one, version_id(other, 'source-hash', space))
        self.assertNotEqual(one, version_id(other, 'changed-source', space))

    def test_initial_winner_pauses_old_study_and_preserves_it(self):
        ticket, reply, payload, _ = self.h.result('researcher-1', {'alpha':1}, 10)
        search = VersionSearch(self.c)
        first = search.add(payload['proposal']['plan'], ticket, evaluated=reply['lead_submission']['experiment_id'])
        second_plan = deepcopy(payload['proposal']['plan'])
        second_plan['model']['hyperparameters']['target_transform'] = 'shifted_log1p'
        second = search.add(second_plan, ticket)
        metric = {**first['best_metric'], 'value':9.0}
        search.measured(second, first['best_experiment_id'], metric, None, True)
        self.assertTrue(first['paused'].startswith('superseded_by_initial:'))
        self.assertEqual(first['best_metric']['value'], 10)
        self.assertEqual(second['best_metric']['value'], 9)

    def test_weaker_initial_keeps_leader_and_challenger(self):
        ticket, reply, payload, _ = self.h.result('researcher-1', {'alpha':1}, 10)
        search = VersionSearch(self.c)
        first = search.add(payload['proposal']['plan'], ticket, evaluated=reply['lead_submission']['experiment_id'])
        second_plan = deepcopy(payload['proposal']['plan'])
        second_plan['model']['hyperparameters']['target_transform'] = 'shifted_log1p'
        second = search.add(second_plan, ticket)
        search.measured(second, first['best_experiment_id'], {**first['best_metric'], 'value':11.0}, None, True)
        self.assertIsNone(first['paused'])
        self.assertIsNone(second['paused'])
        self.assertEqual(len(search.eligible()), 2)

    def test_failed_dialogue_keeps_analysis_owner_visible(self):
        ticket = self.h.issue()
        work = ticket['work_order']['work_order_id']
        self.h.g.abandon(work, 'synthetic timeout after completed analysis')
        memory = self.h.g.research_memory(self.h.root, [])
        self.assertIn(work, memory['visible_work_order_ids'])
        self.assertIn(work, memory['visible_analysis_cutoffs'])

    def test_initial_observation_seeds_optuna_without_fit(self):
        ticket, reply, payload, _ = self.h.result('researcher-1', {'alpha':1}, 10)
        search = VersionSearch(self.c)
        version = search.add(payload['proposal']['plan'], ticket, evaluated=reply['lead_submission']['experiment_id'])
        study = search.study(version)
        self.assertEqual(len(study.trials), 1)
        self.assertEqual(study.best_value, 10)
        trial, plan = search.ask(version)
        self.assertEqual(plan['parent_ids'], [version['initial_experiment_id']])
        self.assertEqual(plan['code_artifact_id'], payload['proposal']['plan']['code_artifact_id'])
        study.tell(trial, state=search.library.trial.TrialState.FAIL)

    def test_analysis_cutoff_preserves_fork_visibility(self):
        from tools.analysis import AnalysisService
        service = AnalysisService.__new__(AnalysisService)
        service.lock = threading.RLock()
        service.records = [
            {'artifact_id':'before','work_order_id':'order','completed_elapsed_ms':10,'query':{'code':'old'}},
            {'artifact_id':'after','work_order_id':'order','completed_elapsed_ms':30,'query':{'code':'new'}},
        ]
        visible = service.recent(['order'], cutoffs={'order':20})
        self.assertEqual([r['artifact_id'] for r in visible], ['before'])
        self.assertNotIn('code', visible[0]['query'])
        self.assertEqual(service.records[0]['query']['code'], 'old')

    def test_resource_requests_leave_analysis_headroom(self):
        budget = self.c.budget
        sized = budget.sized_resources({'cpu_threads':3,'memory_limit_mb':10240,'timeout_seconds':30})
        self.assertLessEqual(sized['memory_limit_mb']+budget.analysis_memory, budget.memory_capacity)


if __name__ == '__main__':
    unittest.main()
