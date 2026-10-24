"""Execution/provenance regressions for open research, not model-quality claims."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import time
import unittest
from unittest.mock import patch

from agent.errors import AgentError
from agent.context import output_schemas
from agent.llm.gateway import Gateway
from agent.researcher import Researcher
from agent.research_actions import normalize_action, validate_action
from tests.graph_helpers import GraphHarness
from tests import test_analysis_access as fixture
from tools.python_analysis import run_analysis


class PythonAnalysisTests(unittest.TestCase):
    setUp = fixture.AnalysisAccessTests.setUp
    tearDown = fixture.AnalysisAccessTests.tearDown
    service = fixture.AnalysisAccessTests.service
    # Reuse data fixture only, not inherited test methods.
    def test_python_sees_fit_only_and_returns_tables(self):
        service = self.service()
        service.budget.reserve_seconds = lambda: 0
        query = {'question':'Check available target and input support', 'columns':['x','group'],
            'include_target':True, 'compare_test':True,
            'code':"def analyze(X_fit, y_fit, X_test, context):\n    return {'max_target':float(y_fit.max()), 'fit':len(X_fit), 'test':len(X_test), 'groups':X_fit.groupby('group').size()}\n"}
        report = run_analysis(service, query)['report']
        self.assertTrue(report['success'], report)
        self.assertEqual(report['result']['max_target'], 7)
        self.assertNotIn('holdout-only', report['result']['groups'])
        self.assertEqual(report['result']['fit'], 20000)
        self.assertTrue(report['sampled'])
        repeated = run_analysis(service, {**query,'question':'Same computation, different prose'})
        self.assertTrue(repeated['reused'])

    def test_python_traceback_can_be_repaired_and_no_environment_or_files(self):
        service = self.service()
        service.budget.reserve_seconds = lambda: 0
        q = {'question':'Test execution errors', 'columns':['x'],
             'code':"def analyze(X_fit,y_fit,X_test,context):\n    return X_fit['absent'].mean()\n"}
        first = run_analysis(service,q)
        self.assertFalse(first['report']['success'])
        self.assertIn('KeyError',first['report']['error'])
        second = run_analysis(service,{**q,'code':q['code'].replace('absent','x')})
        self.assertTrue(second['report']['success'],second)
        for code in ["import os\ndef analyze(a,b,c,d):\n return os.environ\n",
                     "def analyze(a,b,c,d):\n return open('/etc/passwd').read()\n"]:
            with self.assertRaises(AgentError):run_analysis(service,{**q,'code':code})

    def test_python_target_cannot_be_passed_as_input(self):
        service = self.service()
        with self.assertRaises(AgentError):
            run_analysis(service,{'question':'Bad columns','columns':['target'],
                'code':'def analyze(a,b,c,d):\n return 1\n'})


class ResearchCheckpointTests(unittest.TestCase):
    def test_cold_pilots_are_bounded_portable_and_do_not_start_with_catboost(self):
        from agent.catalog import Catalog
        from agent.config import Config
        for kind in ['regression','classification']:
            catalog=Catalog(kind,Config())
            choices=catalog.warmup_choices()
            self.assertNotEqual(catalog.recipes[choices[0][0]]['family'],'catboost')
            for recipe,params in choices:
                self.assertEqual(catalog.normalize(recipe,params),params)
                for key in ['n_estimators','max_iter','iterations']:
                    if key in params:self.assertLessEqual(params[key],96)


    def test_research_action_passes_c09_and_saves_no_fit_checkpoint(self):
        h = GraphHarness('regression',research_mode='adaptive')
        try:
            g=h.g
            class Transport:
                def complete(self,request,timeout,context):
                    return {'choices':[{'message':{'content':json.dumps({'action':'research',
                        'hypothesis':'Need a second fit-only analysis before choosing a model.',
                        'evidence_ids':[],'next_step':'continue'})},'finish_reason':'stop'}]}
            gateway=Gateway(g.config,g.budget,g.artifacts,g.bus,g.log,output_schemas(g.bus),Transport())
            gateway.analysis = SimpleNamespace()
            researcher=Researcher(g.bus,gateway,g.artifacts,g.spec,g.catalog,g.config,g.log,
                                  feature_columns=list(h.h.profile['summary']['train_columns']))
            ticket=h.issue()
            proposal=g.bus.call('C09',ticket['researcher_request']['payload'],researcher.propose)['data']
            self.assertEqual(proposal['decision'],'research')
            self.assertIsNone(proposal['plan'])
            records=[v for v in g.artifacts.entries.values() if v['meta']['kind']=='research_decision']
            self.assertEqual(len(records),1)
        finally:h.close()

    def test_semantic_action_rejection_is_repaired_inside_gateway(self):
        h=GraphHarness('regression',research_mode='adaptive')
        try:
            g=h.g
            class Transport:
                requests=[]
                def complete(self,request,timeout,context):
                    self.requests.append(deepcopy(request))
                    answer={'action':'catalog' if len(self.requests)==1 else 'research',
                        'recipe_id':'not-a-recipe', 'hypothesis':'Inspect before choosing a valid recipe.',
                        'evidence_ids':[]}
                    return {'choices':[{'message':{'content':json.dumps(answer)},'finish_reason':'stop'}]}
            transport=Transport()
            gateway=Gateway(g.config,g.budget,g.artifacts,g.bus,g.log,output_schemas(g.bus),transport)
            gateway.analysis = SimpleNamespace()
            researcher=Researcher(g.bus,gateway,g.artifacts,g.spec,g.catalog,g.config,g.log)
            ticket=h.issue()
            result=researcher.propose(ticket['researcher_request']['payload'])
            self.assertEqual(result['decision'],'research')
            self.assertEqual(len(transport.requests),2)
            self.assertTrue(any('Error:' in (m.get('content') or '') for m in transport.requests[1]['messages']))
        finally:h.close()

    def test_invalid_tuning_space_is_repaired_before_leaving_the_dialogue(self):
        h=GraphHarness('regression',research_mode='adaptive')
        try:
            g=h.g
            class Transport:
                requests=[]
                def complete(self,request,timeout,context):
                    self.requests.append(deepcopy(request))
                    value={'action':'research','hypothesis':'Use a cheaper analysis first.','evidence_ids':[]}
                    if len(self.requests)==1:
                        value.update(action='tune',recipe_id='linear-regressor-v1',hyperparameters={'alpha':1},
                            tuning={'search_space':{'alpha':{'type':'float','low':3,'high':1}},
                                    'max_trials':2,'budget_seconds':30})
                    return {'choices':[{'message':{'content':json.dumps(value)},'finish_reason':'stop'}]}
            transport=Transport()
            gateway=Gateway(g.config,g.budget,g.artifacts,g.bus,g.log,output_schemas(g.bus),transport)
            gateway.analysis=SimpleNamespace()
            researcher=Researcher(g.bus,gateway,g.artifacts,g.spec,g.catalog,g.config,g.log)
            result=researcher.propose(h.issue()['researcher_request']['payload'])
            self.assertEqual(result['decision'],'research')
            self.assertEqual(len(transport.requests),2)
            self.assertTrue(any('INVALID_TUNING_SPACE' in (m.get('content') or '') for m in transport.requests[1]['messages']))
        finally:h.close()

    def test_tool_context_overflow_creates_an_honest_controller_checkpoint(self):
        h=GraphHarness('regression',research_mode='adaptive')
        try:
            g=h.g
            class Transport:
                def complete(self,request,timeout,context):
                    return {'choices':[{'message':{'tool_calls':[{'id':'inspect','type':'function',
                        'function':{'name':'analyze_data','arguments':json.dumps({'kind':'summary',
                            'columns':['x_numeric'],'include_target':False,'group_by':None})}}]},'finish_reason':'tool_calls'}]}
            class Analysis:
                def analyze(self,args,context=None):
                    art=g.artifacts.write_json('analysis/checkpoint-fixture.json',{'scope':'fit_only'},'research_analysis')
                    return {'artifact_id':art['artifact_id'],'report':{'scope':'fit_only'}}
            gateway=Gateway(g.config,g.budget,g.artifacts,g.bus,g.log,output_schemas(g.bus),Transport())
            gateway.analysis=Analysis()
            researcher=Researcher(g.bus,gateway,g.artifacts,g.spec,g.catalog,g.config,g.log)
            original=gateway._request
            calls=[]
            def request(*args,**kwargs):
                calls.append(1)
                if len(calls)>1:raise AgentError('CONTEXT_TOO_LARGE','Synthetic tool transcript pressure')
                return original(*args,**kwargs)
            with patch.object(gateway,'_request',request):
                proposal=researcher.propose(h.issue()['researcher_request']['payload'])
            self.assertEqual(proposal['decision'],'research')
            records=[g.artifacts.read_json(k) for k,v in g.artifacts.entries.items() if v['meta']['kind']=='research_decision']
            self.assertEqual(records[0]['decision_origin'],'controller')
            self.assertIsNone(records[0]['configuration'])
            self.assertTrue(records[0]['evidence_ids'])
            self.assertFalse(g.budget.llm_dialogues)
        finally:h.close()

    def test_large_base_code_is_pageable_and_exact_source_is_preserved(self):
        h=GraphHarness('regression',research_mode='adaptive')
        try:
            g=h.g
            ticket=h.issue();submitted,payload=h.submit(ticket)
            h.evaluate(submitted,1.0)
            # A valid large source remains in the measured plan. Reissue must fit.
            code=g.artifacts.meta(payload['proposal']['plan']['code_artifact_id'])
            # Build using a fresh frozen source through ContextBuilder's graph path.
            original=g.context.build_graph
            def capture(payload,graph):
                graph=deepcopy(graph);graph['base_code']='x'*150000
                graph['base_result']=g.memory_record(submitted['lead_submission']['experiment_id'])
                graph['base_result']['hyperparameters']['large_metadata']='y'*150000
                return original(payload,graph)
            g.context.build_graph=capture
            ticket=h.issue(peer='researcher-2');body=g.artifacts.read_json(ticket['work_order']['context_id'])
            user=json.loads(body['messages'][1]['content'])
            self.assertIsNone(user['base_code'])
            frozen=g.artifacts.read_json(body['metadata']['detail_artifact_id'])
            self.assertEqual(frozen['sections']['base_code'],'x'*150000)
        finally:h.close()
