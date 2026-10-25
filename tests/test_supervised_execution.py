"""Real worker/evaluator/refit execution; provider responses are explicitly mocked."""
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch
import io
import json
import os
import tempfile
import time
import unittest

import pandas as pd

from agent.llm.gateway import MockTransport
from agent.runtime import main
from agent.pipeline_search import VersionSearch
from scripts.make_demo import make_demo


class SupervisedExecutionTests(unittest.TestCase):
    def test_compact_background_waits_for_data_work_and_keeps_output_at_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=make_demo(root/'task','regression',rows=120,test_rows=16)
            original=MockTransport.complete;start=VersionSearch.start;trials=[];calls=[]
            def scripted(transport,request,timeout,context):
                result=original(transport,request,timeout,context)
                if context['metadata']['schema_id']=='ResearchAction-v1':
                    user=json.loads(context['messages'][1]['content']);window=user.get('profile_summary',user)['data_research']
                    calls.append(window['active'])
                    value={'action':'features','recipe_id':'linear-regressor-v1','hyperparameters':{},
                        'feature_program':[{'op':'missing','name':'data_check_'+str(len(calls)),'columns':['x_numeric']}]} if window['active'] else {'action':'stop'}
                    result['choices'][0]['message']['content']=json.dumps(value)
                return result
            def observed(service,version,**kwargs):
                result=start(service,version,**kwargs)
                if result and not kwargs.get('initial'):
                    trials.append(service.c.budget.data_research_snapshot())
                    service.c.budget.deadline=min(service.c.budget.deadline,time.monotonic()+service.c.budget.reserve_seconds()+2)
                return result
            settings={'LLM_MODE':'mock','RESEARCH_MODE':'adaptive','RESEARCH_PROTOCOL':'compact','BACKGROUND_TUNING_ENABLED':'1',
                'GRAPH_ENABLED':'1','TIME_BUDGET_SECONDS':'60','DATA_RESEARCH_MIN_SECONDS':'3','MAX_WORKERS':'1',
                'GRAPH_PEERS':'1','BACKGROUND_CPU_THREADS':'1','TRAIN_CPU_THREADS':'1','CATALOG_WARMUP_EXPERIMENTS':'0',
                'LLM_MAX_CALLS':'12','SCIENTIFIC_RESEARCH_ENABLED':'0','RESEARCH_OOF_ENABLED':'0',
                'FEATURE_RESEARCH_ENABLED':'0','GRAPH_MIN_FREE_DISK_BYTES':'0','MODEL_SNAPSHOT_MAX_BYTES':'0','LOG_LEVEL':'ERROR'}
            with patch.dict(os.environ,settings),patch.object(MockTransport,'complete',scripted),patch.object(VersionSearch,'start',observed),redirect_stderr(io.StringIO()) as stderr:
                code=main([str(task),'--work-dir',str(root/'runs')])
            self.assertEqual(code,0,stderr.getvalue())
            self.assertTrue(any(calls))
            self.assertTrue(trials,stderr.getvalue())
            self.assertTrue(all(not w['active'] and w['elapsed_seconds']>=3 for w in trials))
            self.assertEqual(len(pd.read_csv(task.parent/'submission.csv')),16)

    def test_history_and_macro_decisions_execute_through_workers_and_publish(self):
        for kind,temporal in [('regression',False),('regression',True),('classification',False)]:
            with self.subTest(kind=kind,temporal=temporal):self.execute(kind,temporal)

    def execute(self,kind,temporal):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=make_demo(root/'task',kind,rows=160,test_rows=20)
            if temporal:
                for name in ['train.csv','test.csv']:
                    path=task.parent/name;frame=pd.read_csv(path,dtype={'id':str})
                    frame['period']=[i//20 for i in range(len(frame))];frame.to_csv(path,index=False)
                task.write_text(task.read_text()+'\nPredict future observations.\nTime column: period\n')
            parameters={'base_recipe_id':'hist-native-'+('regressor' if kind=='regression' else 'classifier')+'-v1',
                'model_parameters':{'max_iter':20},'feature_program':
                [{'op':'target_mean','name':'history','columns':['category'],'alpha':10},
                 {'op':'target_count','name':'support','columns':['category']}] if kind=='regression' else []}
            if kind=='classification':parameters['decision_rule']='macro_f1'
            source='from agent.feature_engineering import make_feature_pipeline\nDEFAULT_PARAMS='+repr(parameters)+'\nSEARCH_SPACE={}\ndef make_pipeline(task_spec,params):\n    return make_feature_pipeline(task_spec,params)\n'
            original=MockTransport.complete;calls=0
            def scripted(transport,request,timeout,context):
                nonlocal calls
                result=original(transport,request,timeout,context)
                if context['metadata']['schema_id']=='ResearchAction-v1':
                    calls+=1
                    value={'action':'pipeline','code':source} if calls==1 else {'action':'stop','hypothesis':'Synthetic operator execution complete.'}
                    result['choices'][0]['message']={'role':'assistant','content':json.dumps(value)}
                return result
            settings={'LLM_MODE':'mock','RESEARCH_MODE':'adaptive','RESEARCH_PROTOCOL':'compact','BACKGROUND_TUNING_ENABLED':'1',
                'GRAPH_ENABLED':'1','GRAPH_REFINEMENT_ENABLED':'0','TIME_BUDGET_SECONDS':'60','DATA_RESEARCH_MIN_SECONDS':'0',
                'MAX_WORKERS':'1','GRAPH_PEERS':'1','TRAIN_CPU_THREADS':'1','BACKGROUND_CPU_THREADS':'1',
                'CATALOG_WARMUP_EXPERIMENTS':'0','LLM_MAX_CALLS':'8','SCIENTIFIC_RESEARCH_ENABLED':'0','RESEARCH_OOF_ENABLED':'0',
                'FEATURE_RESEARCH_ENABLED':'0','GRAPH_MIN_FREE_DISK_BYTES':'0','MODEL_SNAPSHOT_MAX_BYTES':'0','LOG_LEVEL':'ERROR'}
            with patch.dict(os.environ,settings),patch.object(MockTransport,'complete',scripted),redirect_stderr(io.StringIO()) as stderr:
                code=main([str(task),'--work-dir',str(root/'runs')])
            self.assertEqual(code,0,stderr.getvalue())
            run=next((root/'runs').glob('run-*'))
            events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['state'],'completed',stderr.getvalue())
            self.assertTrue(any(e['event']=='pipeline.execution_started' for e in events),stderr.getvalue())
            self.assertEqual(len(pd.read_csv(task.parent/'submission.csv')),20)
            reports=[json.loads(p.read_text()) for p in run.rglob('features.json')]
            executed=[item for report in reports for item in report['owned_transformers']]
            self.assertTrue(executed,stderr.getvalue())
            if kind=='regression':
                self.assertTrue(any(row.get('statistics_scope')=='fit_only_target_crossfit'
                    for item in executed for row in item['report'] if isinstance(item['report'],list)))
            else:
                self.assertTrue(any(isinstance(item['report'],dict) and item['report'].get('inner_oof_rows',0)>0 for item in executed))
            jobs=[json.loads(p.read_text()) for p in run.rglob('job.json')]
            self.assertTrue(all(j['training_policy']['strategy']==('temporal_holdout' if temporal else 'random_holdout' if kind=='regression' else 'stratified_holdout') for j in jobs))
