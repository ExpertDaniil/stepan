"""Task recovery, research scheduling and delivery under explicit constraints."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch
import threading
import time
import unittest

from agent.budget import Budget
from agent.confirmation import select_finalists
from agent.errors import AgentError
from agent.fit_diagnostics import FitDiagnostics
from agent.orchestrator import Orchestrator
from agent.pipeline_search import VersionSearch
from agent.submission import Submission
from agent.task_interpreter import TaskInterpreter, recover_contract
from agent.task_rules import bind_requirement, validation_requirement
from tests.helpers import Harness
from tests.graph_helpers import GraphHarness


class TaskRecoveryTests(unittest.TestCase):
    def test_early_diagnostics_leave_budget_for_a_later_leader(self):
        h=Harness('regression')
        try:
            cfg=replace(h.config,time_budget_seconds=1000,research_oof_budget_fraction=.04)
            c=SimpleNamespace(config=cfg,budget=Budget(cfg,100,h.log))
            service=FitDiagnostics(c)
            with patch('agent.fit_diagnostics.time.monotonic',return_value=100):
                self.assertEqual(service.remaining(),20)
                service.spent=20
                self.assertEqual(service.remaining(),0)
            with patch('agent.fit_diagnostics.time.monotonic',return_value=600):
                self.assertEqual(service.remaining(),10)
        finally:h.close()

    def test_metric_formatting_does_not_change_recovered_training_contract(self):
        h = Harness('regression')
        try:
            task = {'task_root': str(h.task.parent), 'task_text_artifact_id': h.code['artifact_id'],
                'candidate_paths': {k: str(h.task.parent/v) for k,v in {
                    'train':'train.csv','test':'test.csv','sample_submission':'sample_submission.csv'}.items()}}
            for text in ['**Metric:** RMSE', '*Metric*: RMSE', '- **Метрика:** RMSE',
                         '| Metric | RMSE |', '## Metric\nRMSE']:
                with self.subTest(text=text):
                    task['task_text'] = h.task.read_text().replace('Metric: RMSE, lower is better.', text)
                    result = recover_contract(task,h.profile,h.artifacts)
                    self.assertEqual(result['draft']['metric']['metric_id'],'rmse')
            task['task_text'] += '\nMetric: MAE'
            self.assertIsNone(recover_contract(task,h.profile,h.artifacts)['draft'])
        finally:h.close()

    def test_explicit_future_protocol_survives_provider_failure_without_random_substitution(self):
        h = Harness('regression',with_periods=True)
        try:
            interpreter = TaskInterpreter(h.config,h.artifacts,h.bus,SimpleNamespace(log=h.log),
                {'task_text':'Predict future observations.\n**Time column:** period'},h.profile)
            interpreter.ask = Mock(side_effect=AgentError('LLM_UNAVAILABLE','test'))
            proposal = interpreter.choose_validation(h.spec)
            self.assertEqual(proposal['strategy'],'temporal_holdout')
            self.assertEqual(proposal['parameters']['column'],'period')
            interpreter.ask.assert_not_called()
            interpreter.task['task_text'] = 'Predict future observations.'
            # Unresolved prose is logged; a measured holdout beats a constant submission.
            self.assertEqual(interpreter.choose_validation(h.spec)['strategy'],'random_holdout')
            req = validation_requirement('Predict future observations.\nTime column: period',['period'])
            with self.assertRaises(AgentError):bind_requirement({'strategy':'random_holdout','parameters':{}},req)
            self.assertIsNone(validation_requirement('A timestamp exists, deployment unspecified.',['timestamp']))
            self.assertIsNone(validation_requirement('We do not predict future observations.',['period']))
        finally:h.close()


class VersionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.h=GraphHarness(graph_hard_decisions=False,research_mode='adaptive',data_research_min_seconds=0)
        self.c=self.h.controller;self.c.graph=self.h.g
        self.c.update_final_reserve=lambda:None
        self.c.record_research_feedback=lambda *a,**k:None

    def tearDown(self):self.h.close()

    def versions(self):
        ticket,reply,payload,_=self.h.result('researcher-1',{'alpha':1},10)
        service=VersionSearch(self.c)
        first=service.add(payload['proposal']['plan'],ticket,evaluated=reply['lead_submission']['experiment_id'])
        plan=deepcopy(payload['proposal']['plan']);plan['model']['hyperparameters']['target_transform']='shifted_log1p'
        second=service.add(plan,ticket)
        return service,first,second

    def test_a_new_initial_winner_retains_the_older_study_and_fair_probe(self):
        service,first,second=self.versions()
        study=service.study(first)
        service.measured(second,first['best_experiment_id'],{**first['best_metric'],'value':9},None,True)
        self.assertIn(first,service.eligible())
        self.assertIs(service.study(first),study)
        first['trials']=second['trials']=3
        first['last_scheduled']=0;second['last_scheduled']=10;service.attempted=4
        self.assertIs(service.choose(),first)

    def test_a_cooling_initial_does_not_block_a_ready_initial(self):
        service,first,second=self.versions()
        first['retry_at']=time.monotonic()+60;service.queue.insert(0,first)
        started=[]
        with patch.object(service,'start',side_effect=lambda v,**k:started.append(v) or True):service.fill()
        self.assertEqual(started,[second])
        self.assertIn(first,service.queue)

    def test_transient_initial_failure_reenters_the_queue_after_recheck(self):
        service,_,second=self.versions()
        service.queue.clear();second.update(paused='initial_unaffordable',retryable_initial=True,retry_at=0)
        with patch.object(service,'start',return_value=False),patch.object(self.c.budget,'can_train',return_value={'allowed':True}):service.fill()
        self.assertIsNone(second['paused'])
        self.assertIn(second,service.queue)

    def test_new_candidate_does_not_cancel_an_existing_fit(self):
        service,first,second=self.versions()
        self.c.config=replace(self.c.config,max_workers=1)
        self.c.cancel_job=Mock()
        service.jobs={'job':(SimpleNamespace(event=None),first,None,False)}
        service.fill()
        self.c.cancel_job.assert_not_called()

    def test_background_optuna_cannot_start_during_the_data_window(self):
        service,first,_=self.versions()
        self.c.budget.config=replace(self.c.config,data_research_min_seconds=1800)
        self.c.budget.start_data_research()
        self.assertEqual(service.eligible(),[])
        self.assertFalse(service.start(first))
        self.assertIsNone(service.library)


class ReadyOutputTests(unittest.TestCase):
    def test_previous_verified_output_can_win_confirmation_and_publish_without_refit(self):
        h=Harness()
        try:
            submission=Submission(h.spec,h.artifacts,h.store,h.log)
            def publish(score,label):
                ev=h.evaluation(score);art=h.candidate(label)
                report=submission.validate_output({'task_spec_id':h.spec['task_spec_id'],
                    'reference_test_artifact_id':h.spec['test_artifact_id'],'candidate_artifact_id':art['artifact_id']})
                current=h.store.champion()
                submission.publish({'experiment_id':ev['experiment_id'],'evaluation_id':ev['evaluation_id'],
                    'candidate_artifact_id':art['artifact_id'],'validation_report_id':report['validation_report_id'],
                    'candidate_sha256':report['candidate_sha256'],'expected_champion_version':current['champion_version'] if current else 0})
                return ev,art
            old,old_art=publish(.7,'A');new,_=publish(.8,'B');h.evaluation(.1)
            budget=Budget(h.config,time.monotonic(),h.log)
            budget.deadline=time.monotonic()+1
            c=Orchestrator.__new__(Orchestrator)
            c.submission,c.store,c.spec,c.artifacts,c.budget,c.log=submission,h.store,h.spec,h.artifacts,budget,h.log
            c.failed_refits=set();c.graph=None;c.science=None;c.update_final_reserve=lambda:None
            c.call=lambda cid,p,handler,**kw:h.bus.call(cid,p,handler,**kw)['data']
            self.assertEqual(select_finalists(c,new['primary_metric']),[new['experiment_id'],old['experiment_id']])
            current=h.store.champion()
            record={'chosen_experiment_id':old['experiment_id'],'expected_champion_version':current['champion_version'],
                'prior_champion_experiment_id':current['experiment_id']}
            art=h.artifacts.write_json('validation/final-ready.json',record,'final_confirmation')
            submission.final_confirmation={**record,'artifact_id':art['artifact_id']}
            c.final_candidate=lambda:old['experiment_id']
            c._finalize_best()  # No Runner exists: attempting another fit would fail this test.
            self.assertEqual(h.store.champion()['experiment_id'],old['experiment_id'])
            self.assertEqual((h.task.parent/'submission.csv').read_bytes(),h.artifacts.resolve(old_art['artifact_id']).read_bytes())
            h.artifacts.resolve(old_art['artifact_id']).write_text('changed')
            self.assertIsNone(submission.ready_candidate(old['experiment_id']))
        finally:h.close()
