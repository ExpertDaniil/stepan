"""Error research, renewable work and deadline boundaries on unseen synthetic tasks."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor

from agent.budget import Budget
from agent.config import Config
from agent.context import output_schemas
from agent.duration import DurationModel
from agent.error_research import ErrorResearch, paired_result
from agent.errors import AgentError
from agent.fit_diagnostics import FitDiagnostics, error_profile, inner_folds, execute_oof
from agent.llm.gateway import Gateway
from agent.orchestrator import Orchestrator
from agent.pipeline_drafts import edit_solution, resolve_candidate
from agent.pipeline_search import VersionSearch
from agent.research_execution import PeerRecovery
from agent.runner import Runner
from tests.graph_helpers import GraphHarness
from tools.analysis import AnalysisService
from tools.python_analysis import run_analysis


class ErrorProfileTests(unittest.TestCase):
    def test_rare_sse_cost_is_not_vetoed_by_two_percent_or_ranked_by_mae(self):
        frame = pd.DataFrame({'opaque_sensor': ['ordinary']*990+['rare']*10, 'answer':0})
        task = {'task_kind':'regression','target_column':'answer','excluded_features':[], 'metric':{'metric_id':'rmse'}}
        pred = np.r_[np.ones(990)*2,np.ones(10)*30]
        report = error_profile(frame,np.zeros(1000),pred,np.arange(1000)%3,task)
        rare = report['hard_groups'][0]
        self.assertEqual(rare['group'],'rare')
        self.assertLess(rare['support_fraction'],.02)
        self.assertAlmostEqual(rare['sse'],9000)
        self.assertTrue(report['groups_overlap'])
        self.assertFalse(report['error_concentration'][0]['test_applicable'])

    def test_macro_f1_uses_class_metrics_and_confusion_without_sse_claims(self):
        task = {'task_kind':'multiclass_classification','target_column':'answer','excluded_features':[],
                'allowed_labels':['cobalt','ochre','teal'],
                'metric':{'metric_id':'f1_macro','prediction_kind':'labels','direction':'maximize'}}
        y = np.asarray(['cobalt']*80+['ochre']*18+['teal']*2)
        pred = np.asarray(['cobalt']*100)
        report = error_profile(pd.DataFrame({'q':np.arange(100)%2}),y,pred,np.arange(100)%3,task)
        self.assertNotIn('sse',report)
        self.assertIn('confusion',report['class_summary'])
        rare = next(g for g in report['hard_groups'] if g['definition']=={'kind':'target_class','label':'teal'})
        self.assertTrue(rare['support_uncertain'])
        self.assertFalse(rare['test_applicable'])
        self.assertLess(report['primary_metric_value'],.5)

    def test_preprocessing_and_estimator_are_new_inside_each_oof_fold(self):
        fits = []
        class SeenTransformer(TransformerMixin,BaseEstimator):
            def fit(self,X,y=None):
                self.seen = set(X.index);fits.append(self.seen);return self
            def transform(self,X):return X
        task = {'task_kind':'regression','target_column':'z','metric':{'prediction_kind':'values'},'allowed_labels':[]}
        X = pd.DataFrame({'unfamiliar':np.arange(90)}, index=np.arange(90)*3)
        y = pd.Series(np.arange(90), index=X.index)
        folds = inner_folds(pd.DataFrame({'z':y}),task,{'strategy':'random_holdout','parameters':{}},42)
        with tempfile.TemporaryDirectory() as tmp:
            execute_oof(SimpleNamespace(make_pipeline=lambda *_:Pipeline([('learned',SeenTransformer()),('model',DummyRegressor())])),
                task,{},X,y,{'cpu_threads':1,'diagnostic':{'folds':folds},'prediction_path':str(Path(tmp)/'oof.csv')},None,ExitStack())
        self.assertEqual(len(fits),3)
        for seen,fold in zip(fits,folds):self.assertFalse(seen & set(X.index[fold['predict']]))

    def test_paired_rmse_decomposition_and_nonadditive_f1(self):
        task = {'task_kind':'regression','metric':{'metric_id':'rmse','direction':'minimize'}}
        result = paired_result(task,[0,0,0,0],[[4],[4],[1],[1]],[[1],[1],[5],[5]],[1,1,0,0])
        self.assertGreater(result['inside_group']['improvement'],0)
        self.assertLess(result['overall']['improvement'],0)
        self.assertAlmostEqual(result['overall']['sse_reduction'],result['inside_group']['sse_reduction']+result['outside_group']['sse_reduction'])
        task = {'task_kind':'multiclass_classification','allowed_labels':['a','b'],
            'metric':{'metric_id':'f1_macro','direction':'maximize','prediction_kind':'labels'}}
        result = paired_result(task,['a','a','b','b'],[['a'],['a'],['a'],['a']],[['a'],['b'],['b'],['b']],[1,1,0,0])
        self.assertNotIn('sse_reduction',result['overall'])
        self.assertFalse(result['group_scores_additive'])


class DiagnosticIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness(research_mode='adaptive',graph_hard_decisions=False,time_budget_seconds=900,
            research_oof_budget_fraction=.1, research_oof_rows=100, research_oof_timeout_seconds=30)
        self.c = self.h.controller
        c = self.c
        c.graph = self.h.g
        c.science = None
        c.error_research = ErrorResearch(c)
        c.publish_research_updates = lambda:None
        for name in ('call','lease','release','resources','cancel_job','wait_jobs','_heartbeat_and_limits'):
            setattr(c,name,MethodType(getattr(Orchestrator,name),c))
        c.runner = Runner(c.artifacts,c.config,c.budget,c.log)
        c.context.analysis = AnalysisService(c.spec,c.validation,c.artifacts,c.budget,c.log)
        _,reply,_,_ = self.h.result('researcher-1',{'alpha':1},10)
        self.eid = reply['lead_submission']['experiment_id']
        self.champion = patch.object(c.store,'champion',return_value={'experiment_id':self.eid,'champion_version':1});self.champion.start()
        self.diag = FitDiagnostics(c)

    def tearDown(self):
        self.diag.shutdown()
        for job in self.c.runner.active():self.c.cancel_job(job,'test cleanup');self.c.wait_jobs([job])
        self.champion.stop();self.h.close()

    def complete(self,eid=None):
        eid = eid or self.eid
        self.diag.ensure([eid])
        started = time.monotonic()
        while self.diag.running and time.monotonic()-started < 20:
            self.diag.poll();time.sleep(.02)
        self.assertFalse(self.diag.running)
        self.assertTrue(self.diag.records)
        return self.diag.records[-1]

    def test_budget_wait_keeps_the_pending_diagnostic(self):
        with patch.object(self.diag, 'remaining', return_value=3):
            self.diag.ensure([self.eid])
        self.assertEqual(self.diag.pending, [self.eid])
        self.assertEqual(self.diag.attempted, set())
        self.assertIsNone(self.diag.running)
        record = self.complete()
        self.assertEqual(record['report']['analysis_execution'], 'leased_oof_worker')

    def test_denied_resources_do_not_consume_attempts_then_start_amid_training(self):
        c = self.c
        for _ in range(3):
            with patch.object(c,'lease',return_value={'granted':False}):self.diag.ensure([self.eid])
        self.assertEqual(self.diag.attempted,set())
        self.assertEqual(self.diag.spent,0)
        # A real existing train lease remains active while OOF acquires the other.
        holder = c.lease('other-worker','train',{'cpu_threads':1,'memory_limit_mb':256,'timeout_seconds':30})
        self.assertTrue(holder['granted'])
        try:
            self.diag.ensure([self.eid])
            self.assertTrue(self.diag.running)
            self.assertEqual(len(c.budget.leases),2)
            while self.diag.running:self.diag.poll();time.sleep(.02)
            self.assertEqual(len(c.budget.leases),1)
        finally:
            lease=holder['lease'];c.call('C13',{'operation':'release','owner_id':'other-worker','phase':'train',
                'requested':{'cpu_threads':1,'memory_limit_mb':256,'timeout_seconds':30},
                'lease_id':lease['lease_id'],'fencing_token':lease['fencing_token']},c.budget.lease)

    def test_oof_analysis_aligns_rows_excludes_holdout_and_never_reads_sql(self):
        record = self.complete()
        ticket = self.h.issue()
        context = self.c.artifacts.read_json(ticket['work_order']['context_id'])
        service = self.c.context.analysis
        column = next(n for n in service.train_columns if n not in [self.c.spec['target_column'],*self.c.spec['excluded_features']])
        query = {'experiment_id':'best','question':'Check aligned residuals','columns':[column],'include_target':True,
            'code':'def analyze(X,y,T,context):\n    o=context["oof"]\n    return {"rows":len(X),"aligned":X.index.equals(y.index) and X.index.equals(o["errors"].index) and X.index.equals(o["fold_ids"].index),"error":float(o["squared_errors"].sum().sum())}\n'}
        with patch.object(self.c.store,'get',side_effect=AssertionError('SQL from model thread')):
            result = run_analysis(service,query,context)
        self.assertTrue(result['report']['success'],result)
        self.assertTrue(result['report']['result']['aligned'])
        self.assertEqual(result['report']['result']['rows'],record['report']['predicted_rows'])
        self.assertLessEqual(set(service.oof_sources[self.eid]['identity']['predicted_row_ids']), service.fit_indices)
        with self.assertRaises(AgentError):run_analysis(service,{**query,'experiment_id':'foreign-branch'},context)
        service.oof_sources[self.eid]['identity']['predicted_row_ids'].append(999999)
        query['code'] += '\n# new check'
        with self.assertRaises(AgentError):run_analysis(service,query,context)

    def test_warmup_waits_for_oof_memory_instead_of_failing_start(self):
        from agent.graph.controller import acquire_training_lease
        c = self.c
        c.budget.cpu_capacity = 7
        c.budget.memory_capacity = 12675
        c.budget.analysis_memory = 1024
        c.science = SimpleNamespace(diagnostics=self.diag)
        self.diag.ensure([self.eid])
        self.assertIsNotNone(self.diag.running)
        request = {'cpu_threads':3, 'memory_limit_mb':10240, 'timeout_seconds':30}
        self.assertFalse(c.lease('warmup-next', 'train', request)['granted'])
        allocation = acquire_training_lease(c, 'warmup-next', request)
        self.assertTrue(allocation['granted'])
        self.assertTrue(self.diag.records)
        lease = allocation['lease']
        c.call('C13', {'operation':'release', 'owner_id':'warmup-next', 'phase':'train',
            'requested':request, 'lease_id':lease['lease_id'], 'fencing_token':lease['fencing_token']}, c.budget.lease)

    def test_empty_assignment_gets_group_after_oof_without_mutating_ticket(self):
        c = self.c
        ids = self.h.g.visible_history(self.h.root)
        c.error_research.advice(ids, 'researcher-1')
        key = 'legacy/researcher-1'
        frozen = deepcopy(c.error_research.assignments[key])
        self.assertIsNone(frozen['problem_key'])
        self.complete()
        advice = c.error_research.advice(ids, 'researcher-1')
        self.assertIsNotNone(advice['observation'])
        self.assertIsNotNone(c.error_research.assignments[key]['problem_key'])
        self.assertIsNone(frozen['definition'])

    def test_new_leader_refresh_and_more_than_two_diagnostic_jobs(self):
        for index in range(3):
            if index:
                _,reply,_,_ = self.h.result('researcher-1',{'alpha':index+2},10-index)
                self.eid=reply['lead_submission']['experiment_id']
            self.complete()
        self.assertEqual(len(self.diag.records),3)
        self.assertEqual(len(self.diag.attempted),3)
        advice1 = self.c.error_research.advice(self.h.g.visible_history(self.h.root),'researcher-1')
        advice2 = self.c.error_research.advice(self.h.g.visible_history(self.h.root),'researcher-2')
        self.assertNotEqual(advice1['recommendation']['question'],advice2['recommendation']['question'])
        self.assertEqual(advice1['leader']['experiment_id'],self.eid)
        self.assertIn('recommendation',advice1)


class ContinuationTests(unittest.TestCase):
    def test_executable_dispatcher_rotates_fast_peer_while_other_peer_waits(self):
        from agent.llm.gateway import MockTransport
        from agent.runtime import main
        from scripts.make_demo import make_demo
        original=MockTransport.complete
        release=threading.Event();calls={}
        def complete(transport,request,timeout,context):
            if not context['metadata'].get('pipeline_execution'):
                return original(transport,request,timeout,context)
            peer=context['metadata']['researcher_id'];calls[peer]=calls.get(peer,0)+1
            if peer=='researcher-1' and calls[peer]==1:
                value={'action':'research','next_step':'continue','hypothesis':'Continue the same unresolved synthetic question.'}
            else:
                if peer=='researcher-1':release.set()
                elif not release.wait(5):raise AssertionError('Fast peer waited for the other dialogue')
                value={'action':'stop','hypothesis':'Finite scripted work completed.'}
            return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps(value)}}],
                    'usage':{'completion_tokens':30}}
        with tempfile.TemporaryDirectory() as tmp:
            task=make_demo(Path(tmp)/'task','regression',rows=90,test_rows=12)
            settings={'LLM_MODE':'mock','TIME_BUDGET_SECONDS':'30','RESEARCH_MODE':'adaptive',
                'BACKGROUND_TUNING_ENABLED':'1','CATALOG_WARMUP_EXPERIMENTS':'0','LLM_MAX_CALLS':'0',
                'LLM_MAX_TOTAL_OUTPUT_TOKENS':'0','GRAPH_MIN_FREE_DISK_BYTES':'0',
                'GRAPH_REFINEMENT_ENABLED':'0','LOG_LEVEL':'ERROR','SCIENTIFIC_RESEARCH_ENABLED':'0'}
            with patch.dict(os.environ,settings),patch.object(MockTransport,'complete',complete):
                code=main([str(task),'--work-dir',str(Path(tmp)/'runs'),'--llm-mode','mock'])
            self.assertEqual(code,0);self.assertTrue((task.parent/'submission.csv').is_file())
            self.assertGreaterEqual(calls.get('researcher-1',0),2)
            self.assertTrue(release.is_set())

    def test_owned_tool_schemas_and_examples_match_contracts(self):
        from agent.config import ROOT
        from agent.contracts import Schemas
        from agent.pipeline_drafts import EDIT_SCHEMA
        from tools.python_analysis import PYTHON_ANALYSIS_SCHEMA
        schemas=Schemas()
        self.assertEqual(schemas.schema['$defs']['PythonAnalysisQuery'],PYTHON_ANALYSIS_SCHEMA)
        self.assertEqual(schemas.schema['$defs']['PipelineSourceEdit'],EDIT_SCHEMA)
        for name,definition in [('python-analysis-oof.json','PythonAnalysisQuery'),('pipeline-source-edit.json','PipelineSourceEdit')]:
            schemas.validate(json.loads((ROOT/'contracts/examples'/name).read_text()),definition)

    def test_native_source_edit_tool_produces_a_reference_only_pipeline(self):
        from agent.researcher import Researcher
        h=GraphHarness(research_mode='adaptive',graph_hard_decisions=False)
        try:
            c=h.controller
            source="from agent.ml import build_pipeline\nDEFAULT_PARAMS={'alpha':2.0}\ndef make_pipeline(task_spec,params):\n    return build_pipeline('linear-regressor-v1',task_spec,params)\n"
            class Transport:
                def complete(self,request,timeout,context):
                    replies=[m for m in request['messages'] if m['role']=='tool']
                    if not replies:
                        message={'tool_calls':[{'id':'save','type':'function','function':{'name':'edit_solution','arguments':json.dumps({'code':source})}}]}
                    else:
                        ident=json.loads(replies[-1]['content'])['draft_id']
                        message={'content':json.dumps({'action':'pipeline','code_ref':ident})}
                    return {'choices':[{'finish_reason':'stop','message':message}], 'usage':{'completion_tokens':50}}
            gateway=Gateway(c.config,c.budget,c.artifacts,c.bus,c.log,output_schemas(c.bus),Transport())
            gateway.analysis=c.context.analysis=AnalysisService(c.spec,c.validation,c.artifacts,c.budget,c.log)
            researcher=Researcher(c.bus,gateway,c.artifacts,c.spec,c.catalog,c.config,c.log)
            researcher.validation_id=c.validation['validation_id']
            ticket=h.issue()
            proposal=c.bus.call('C09',ticket['researcher_request']['payload'],researcher.propose)['data']
            self.assertEqual(proposal['decision'],'propose')
            self.assertEqual(proposal['plan']['model']['hyperparameters']['alpha'],2.)
            self.assertEqual(c.artifacts.resolve(proposal['plan']['code_artifact_id']).read_text(),source)
        finally:h.close()

    def test_recovery_uses_time_and_quarantines_the_action_not_peer(self):
        now=[0.];r=PeerRecovery(clock=lambda:now[0])
        for _ in range(8):
            state=r.failure('one',AgentError('INVALID_LLM_OUTPUT','bad input'),0,'source-a')
            self.assertFalse(r.ready('one',0));now[0]=state['retry_at']+.1
            self.assertTrue(r.ready('one',0))
        self.assertTrue(r.ready('other',0))
        key,a=r.quarantine('code-a',{'x':1},'fit error')
        key2,b=r.quarantine('code-b',{'x':1},'fit error')
        self.assertNotEqual(key,key2)
        r.defer('one','explicit_stop',0)
        now[0]+=1000
        self.assertFalse(r.ready('one',0));self.assertTrue(r.ready('one',1))

    def test_study_survives_more_than_128_trials_and_batch_review(self):
        h=GraphHarness(research_mode='adaptive',graph_hard_decisions=False,time_budget_seconds=900)
        try:
            c=h.controller;c.graph=h.g
            c.update_final_reserve=lambda:None
            ticket,reply,payload,_=h.result('researcher-1',{'alpha':1},10)
            s=VersionSearch(c);v=s.add(payload['proposal']['plan'],ticket,evaluated=reply['lead_submission']['experiment_id'])
            study=s.study(v)
            for _ in range(130):
                self.assertIn(v,s.eligible())
                trial,plan=s.ask(v);study.tell(trial,10.)
            self.assertIs(study,s.study(v));self.assertEqual(len(study.trials),131)
            self.assertGreater(v['batch_limit'],128)
        finally:h.close()

    def test_one_unaffordable_trial_does_not_permanently_pause_the_version(self):
        h=GraphHarness(research_mode='adaptive',graph_hard_decisions=False,time_budget_seconds=900)
        try:
            c=h.controller;c.graph=h.g
            ticket,reply,payload,_=h.result('researcher-1',{'alpha':1},10)
            search=VersionSearch(c)
            version=search.add(payload['proposal']['plan'],ticket,evaluated=reply['lead_submission']['experiment_id'])
            with patch.object(c.budget,'can_train',return_value={'allowed':False}):
                self.assertFalse(search.start(version))
            self.assertIsNone(version['paused'])
            self.assertEqual(search.study(version).trials[-1].state.name,'PRUNED')
            version['retry_at']=0
            self.assertIn(version,search.eligible())
        finally:h.close()

    def test_unlimited_output_is_documented_internal_quota_only(self):
        with patch.dict(os.environ,{'LLM_MAX_CALLS':'0','LLM_MAX_TOTAL_OUTPUT_TOKENS':'0'}):
            cfg=Config.from_env()
        budget=Budget(cfg,time.monotonic(),None)
        self.assertIsNone(budget.token_snapshot()['output_available'])
        self.assertIsNone(budget.snapshot()['llm_calls_remaining'])

    def test_hung_research_executor_does_not_hold_python_process_exit(self):
        code='from agent.research_execution import ResearchExecutor\nimport threading\np=ResearchExecutor(1)\np.submit(threading.Event().wait)\np.shutdown(wait=False)\nprint("closed")\n'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=3)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('closed',result.stdout)

    def test_drafts_support_exact_edits_and_reference_submission_without_forms(self):
        h=GraphHarness(research_mode='adaptive',graph_hard_decisions=False)
        try:
            c=h.controller;ticket=h.issue();ctx=c.artifacts.read_json(ticket['work_order']['context_id'])
            source="from agent.ml import build_pipeline\nDEFAULT_PARAMS={'alpha':1.0}\ndef make_pipeline(task_spec,params):\n    return build_pipeline('linear-regressor-v1',task_spec,params)\n"
            result=edit_solution({'code':source},ctx,c.artifacts,c.log)
            changed=edit_solution({'draft_id':result['draft_id'],'edits':[{'old':"'alpha':1.0",'new':"'alpha':2.0"}]},ctx,c.artifacts,c.log)
            action=resolve_candidate({'action':'pipeline','code_ref':changed['draft_id']},ctx,c.artifacts)
            self.assertEqual(action['hyperparameters']['alpha'],2.)
            self.assertNotIn('investigation',action)
            with self.assertRaises(AgentError):resolve_candidate({'action':'pipeline','code_ref':'foreign'},ctx,c.artifacts)
            with self.assertRaises(AgentError):edit_solution({'draft_id':result['draft_id'],'edits':[{'old':'absent','new':'x'}]},ctx,c.artifacts,c.log)
            self.assertEqual(resolve_candidate({'action':'pipeline','code_ref':result['draft_id']},ctx,c.artifacts)['hyperparameters']['alpha'],1.)
        finally:h.close()

    def test_malformed_and_truncated_dialogues_rotate_repeatedly_with_archived_results(self):
        h=GraphHarness(research_mode='adaptive',graph_hard_decisions=False,time_budget_seconds=900)
        try:
            c=h.controller
            class Transport:
                def complete(self,request,timeout,context):
                    return {'choices':[{'finish_reason':'length','message':{'content':'{"action":"pipeline","code":"unfinished'}}]}
            gateway=Gateway(c.config,c.budget,c.artifacts,c.bus,c.log,output_schemas(c.bus),Transport())
            for _ in range(4):
                ticket=h.issue()
                answer=gateway.complete({'purpose':'research_proposal','context_artifact_id':ticket['work_order']['context_id'],
                    'response_schema_id':'ResearchAction-v1','tool_schema_ids':[], 'max_output_tokens':256,
                    'deadline_seconds':30,'attempt_limit':2})
                self.assertEqual(answer['parsed_payload']['action'],'research')
                self.assertEqual(answer['parsed_payload']['next_step'],'continue')
                self.assertTrue(any(call['name']=='transcript_archive' for call in answer['tool_calls']))
                h.g.abandon(ticket['work_order']['work_order_id'],'Continue bounded dialogue')
            ctx=c.artifacts.read_json(h.issue()['work_order']['context_id'])
            sections=c.artifacts.read_json(ctx['metadata']['detail_artifact_id'])['sections']
            self.assertTrue(sections['transcripts_index'])
            self.assertFalse(c.budget.search_stopped)
            self.assertFalse(c.budget.llm_dialogues)
        finally:h.close()

    def test_budget_exhaustion_does_not_clear_on_clock_until_probe_succeeds(self):
        h=GraphHarness(research_mode='adaptive',graph_hard_decisions=False)
        try:
            c=h.controller
            gateway=Gateway(c.config,c.budget,c.artifacts,c.bus,c.log,output_schemas(c.bus),SimpleNamespace())
            gateway.daily_budget_exhausted.set()
            with self.assertRaises(AgentError):gateway.check_availability()
            gateway.daily_retry_at=time.monotonic()-1
            gateway.check_availability()
            self.assertTrue(gateway.daily_budget_exhausted.is_set())
            gateway.daily_probe_active=True
            with self.assertRaises(AgentError):gateway.check_availability()
        finally:h.close()


class FinalBudgetTests(unittest.TestCase):
    def test_measured_csv_validation_and_publication_can_increase_io_reserve(self):
        budget=Budget(Config(time_budget_seconds=10800),time.monotonic(),None)
        self.assertEqual(budget.output_reserve(),60)
        budget.observe_publication(90)
        self.assertEqual(budget.output_reserve(),135)

    def test_late_unaffordable_winner_keeps_deliverable_and_freezes_selection(self):
        h=GraphHarness(research_mode='adaptive',graph_hard_decisions=False,time_budget_seconds=900)
        try:
            c=h.controller;c.graph=h.g;c.failed_refits=set();c.science=None
            c.submission=SimpleNamespace(ready_candidate=lambda eid:None)
            c.researcher=SimpleNamespace(gateway=SimpleNamespace(cancelled=threading.Event()))
            for name in ('begin_finalization','affordable_final_candidate','final_candidate','update_final_reserve'):
                setattr(c,name,MethodType(getattr(Orchestrator,name),c))
            _,reply,_,_=h.result('researcher-1',{'alpha':1},10)
            ready=reply['lead_submission']['experiment_id']
            with patch.object(c.store,'champion',return_value={'experiment_id':ready,'champion_version':1}):
                _,reply,_,_=h.result('researcher-2',{'alpha':2},9)
                late=reply['lead_submission']['experiment_id']
                c.budget.deadline=time.monotonic()+300
                c.budget.duration.estimate=lambda p,phase='train':1000 if p['model']['hyperparameters']['alpha']==2 else 20
                c.begin_finalization()
                self.assertEqual(c.final_candidate(),ready)
                self.assertNotEqual(c.final_candidate(),late)
                self.assertTrue(c.budget.search_stopped)
                self.assertTrue(c.researcher.gateway.cancelled.is_set())
                c.budget.duration.estimate=lambda *_:1
                self.assertEqual(c.final_candidate(),ready)
        finally:h.close()

    def test_failed_frozen_finalist_selects_only_affordable_fallback(self):
        h=GraphHarness(research_mode='adaptive',graph_hard_decisions=False,time_budget_seconds=900)
        try:
            c=h.controller;c.graph=h.g;c.failed_refits=set();c.finalization_started=True
            c.final_candidate=MethodType(Orchestrator.final_candidate,c)
            c.affordable_final_candidate=MethodType(Orchestrator.affordable_final_candidate,c)
            _,reply,_,_=h.result('researcher-1',{'alpha':1},10);ready=reply['lead_submission']['experiment_id']
            with patch.object(c.store,'champion',return_value={'experiment_id':ready,'champion_version':1}):
                _,reply,_,_=h.result('researcher-2',{'alpha':2},9);winner=reply['lead_submission']['experiment_id']
                c.finalization_candidate=winner;c.failed_refits.add(winner)
                self.assertEqual(c.final_candidate(),ready)
        finally:h.close()

    def test_five_minutes_admits_fitting_trial_and_rejects_longer_one(self):
        cfg=Config(time_budget_seconds=10800)
        with patch('agent.budget.time.monotonic',return_value=0):
            budget=Budget(cfg,0,None);budget.deadline=300
            plan={'recipe_id':'x','code_artifact_id':'code','model':{'family':'custom','hyperparameters':{}},'resources':{'cpu_threads':1}}
            budget.set_final_plan(plan)
            budget.duration.estimate=lambda p,phase='train':150 if phase=='refit' else p.get('cost',100)
            budget.output_reserve=lambda:30
            self.assertEqual(budget.reserve_seconds(),181)  # refit lease + handoff + output
            self.assertTrue(budget.can_train({**plan,'cost':100})['allowed'])
            self.assertFalse(budget.can_train({**plan,'cost':125})['allowed'])

    def test_custom_timings_are_specific_to_source_and_effective_parameters(self):
        cfg=Config(time_budget_seconds=300)
        m=DurationModel(cfg)
        plan={'recipe_id':'python-pipeline-v1','code_artifact_id':'first','model':{'family':'custom','hyperparameters':{'depth':4}},'resources':{'cpu_threads':1}}
        m.observe(plan,'train',40,fit_seconds=39)
        different=deepcopy(plan);different['model']['hyperparameters']['depth']=8
        self.assertNotEqual(m.sample_key(plan),m.sample_key(different))
        self.assertFalse(m.samples[m.sample_key(different)])
        self.assertGreater(m.estimate(plan),40)


if __name__=='__main__':unittest.main()


class EffectAndUtilityTests(unittest.TestCase):
    """A candidate is compared with its research base (effect) and with the current leader (utility)."""

    def setUp(self):
        self.h = GraphHarness('regression', graph_hard_decisions=False, research_mode='adaptive')
        self.c = self.h.controller
        self.c.graph = self.h.g
        self.c.error_research = ErrorResearch(self.c)

    def tearDown(self):
        self.h.close()

    def measured(self, alpha, offset):
        import pandas as pd
        from agent.predictions import targets, validation_columns
        c, h = self.c, self.h
        ticket = h.issue()
        reply, _ = h.submit(ticket, {'alpha': alpha})
        eid = reply['lead_submission']['experiment_id']
        c.store.apply_event(h.train_event(reply), 'train')
        rows = c.evaluator.protocols[c.validation['validation_id']]['valid_indices']
        truth = pd.read_csv(c.artifacts.resolve(c.spec['train_artifact_id'])).loc[rows, targets(c.spec)[0]].to_numpy(float)
        path = c.artifacts.safe_path('predictions/'+eid+'.csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'__row_index': rows, validation_columns(c.spec)[0]: truth+offset}).to_csv(path, index=False)
        art = c.artifacts.register(path, 'validation_predictions', 'text/csv')
        ev = c.evaluator.evaluate({'experiment_id': eid, 'task_spec_id': c.spec['task_spec_id'], 'dataset_id': c.spec['dataset_id'],
            'validation_id': c.validation['validation_id'], 'predictions_artifact_id': art['artifact_id'], 'fidelity': 'full'})['evaluation']
        h.g.invoke('G05', {'experiment_id': eid, 'evaluation_id': ev['evaluation_id'],
                           'selection_policy_id': 'same-protocol-primary-v1'}, key='eval-'+eid)
        return eid

    def test_effect_against_base_and_utility_against_a_newer_leader(self):
        base = self.measured(1.0, 3.0)
        leader = self.measured(2.0, 1.0)  # background tuning found a better solution meanwhile
        candidate = self.measured(3.0, 2.0)
        self.assertEqual(self.c.store.best(), leader)
        self.c.error_research.compare(candidate, {'base_experiment_id': base, 'researcher_id': 'researcher-1'})
        summary = self.c.error_research.results[-1]
        self.assertEqual(summary['proposal_base_experiment_id'], base)
        self.assertEqual(summary['status'], 'overall_improved')
        self.assertAlmostEqual(summary['overall']['improvement'], 1.0, places=6)
        self.assertEqual(summary['versus_current_leader']['experiment_id'], leader)
        self.assertAlmostEqual(summary['versus_current_leader']['improvement'], -1.0, places=6)
