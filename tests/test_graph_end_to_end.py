from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from agent.catalog import Catalog
from agent.llm.gateway import MockTransport
from agent.runtime import main
from agent.runner import Runner
from agent.errors import AgentError
from scripts.make_demo import make_demo
from scripts.reconstruct_solution import reconstruct


class GraphEndToEndTests(unittest.TestCase):
    def test_no_research_api_quota_finishes_cleanly_with_the_verified_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=make_demo(root/'task')
            settings={'LLM_MODE':'mock','GRAPH_ENABLED':'1','TIME_BUDGET_SECONDS':'90','MAX_RESEARCH_EXPERIMENTS':'3',
                'CATALOG_WARMUP_EXPERIMENTS':'0','LLM_MAX_CALLS':'2','LOG_LEVEL':'ERROR','RESEARCH_MODE':'catalog','GRAPH_MIN_FREE_DISK_BYTES':'0'}
            with patch.dict(os.environ,settings),redirect_stderr(io.StringIO()) as stderr:
                code=main([str(task),'--llm-mode','mock','--work-dir',str(root/'runs')])
            self.assertEqual(code,0,stderr.getvalue()[-4000:]);self.assertTrue((task.parent/'submission.csv').is_file())
            run=next((root/'runs').glob('run-*'))
            self.assertIn('State: completed',(run/'summary.md').read_text())
            events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
            finish=next(e for e in events if e['event']=='search.finished')
            self.assertEqual(finish['stop_reason'],'llm_call_limit')
            self.assertFalse(any(e['event']=='research.proposal_failed' for e in events))

    def test_spawn_failure_after_baseline_closes_assignments_and_retains_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=make_demo(root/'task')
            original=Runner.start_train
            def fail_research(runner,payload,spec,experiment):
                if payload['plan']['model']['family']!='dummy':
                    raise AgentError('SPAWN_FAILED','Synthetic graph spawn failure')
                return original(runner,payload,spec,experiment)
            settings={'LLM_MODE':'mock','GRAPH_ENABLED':'1','TIME_BUDGET_SECONDS':'90','MAX_RESEARCH_EXPERIMENTS':'2',
                'CATALOG_WARMUP_EXPERIMENTS':'0','LOG_LEVEL':'ERROR','RESEARCH_MODE':'catalog','GRAPH_MIN_FREE_DISK_BYTES':'0'}
            with patch.dict(os.environ,settings),patch.object(Runner,'start_train',fail_research),redirect_stderr(io.StringIO()):
                code=main([str(task),'--llm-mode','mock','--work-dir',str(root/'runs')])
            self.assertEqual(code,0);self.assertTrue((task.parent/'submission.csv').is_file())
            run=next((root/'runs').glob('run-*'));self.assertIn('State: degraded',(run/'summary.md').read_text())
            graph=json.loads(next(run.glob('exports/final-*/graph.json')).read_text())
            self.assertEqual(sum(e['state']=='failed'for e in graph['experiments']),2)
            self.assertTrue(all(w['state']not in{'issued','submitted'}for w in graph['work_orders']))
            self.assertTrue(all(b['state']!='pending'for b in graph['bindings']))

    def test_real_training_confirms_transfer_forks_and_backtracks(self):
        import numpy as np
        import pandas as pd
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=make_demo(root/'task','regression',rows=180,test_rows=30)
            for name in ['train.csv','test.csv']:
                frame=pd.read_csv(task.parent/name,dtype={'id':str},keep_default_na=False)
                x=np.linspace(0,3,len(frame))
                frame['x_numeric']=x;frame['z_numeric']=0.;frame['category']='same'
                if name=='train.csv':frame['label']=np.exp(2*x)
                frame.to_csv(task.parent/name,index=False)
            original=MockTransport.complete
            def response(transport,request,timeout,context):
                if context['metadata']['schema_id']!='ResearchChoice-v1':
                    return original(transport,request,timeout,context)
                user=json.loads(context['messages'][1]['content']);peer=context['metadata']['researcher_id']
                restrictions=user['restrictions'];required=restrictions['required_capability_ids'];forbidden=restrictions['forbidden_capability_ids']
                if required:
                    alpha=10000-int(peer[-1]);transform='shifted_log1p'
                elif forbidden:
                    alpha=.01+int(peer[-1])*.01;transform='identity'
                else:
                    alpha=1. if peer=='researcher-1' else 3.
                    transform='shifted_log1p' if peer=='researcher-1' else 'identity'
                result={'recipe_id':'linear-regressor-v1','hyperparameters':{'alpha':alpha,'target_transform':transform},
                    'hypothesis':'Synthetic controlled test of an approved target transform','expected_effect':'Verify transfer through independently evaluated real predictions'}
                return {'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps(result)}}]}
            def warmups(catalog):
                return [('linear-regressor-v1',catalog.normalize('linear-regressor-v1',{'alpha':a}))for a in [1.,3.]]
            settings={'LLM_MODE':'mock','GRAPH_ENABLED':'1','TIME_BUDGET_SECONDS':'180','MAX_RESEARCH_EXPERIMENTS':'8',
                'CATALOG_WARMUP_EXPERIMENTS':'2','MAX_WORKERS':'2','GRAPH_PEERS':'2','GRAPH_TRAIN_WORKERS':'1',
                'GRAPH_PLATEAU_WINDOW':'1','GRAPH_PLATEAU_SECONDS':'0','GRAPH_MIN_FREE_DISK_BYTES':'0',
                'GRAPH_MIN_EFFECT':'0.001','LOG_LEVEL':'ERROR','EXPERIMENT_TIMEOUT_SECONDS':'30','RESEARCH_MODE':'catalog'}
            with patch.dict(os.environ,settings),patch.object(MockTransport,'complete',response),patch.object(Catalog,'warmup_choices',warmups),patch.object(Catalog,'assignment_recipes',return_value=['linear-regressor-v1']),redirect_stderr(io.StringIO())as stderr:
                code=main([str(task),'--llm-mode','mock','--work-dir',str(root/'runs')])
            self.assertEqual(code,0,stderr.getvalue()[-8000:])
            run=next((root/'runs').glob('run-*'))
            self.assertIn('State: completed',(run/'summary.md').read_text(),stderr.getvalue()[-8000:])
            events=[json.loads(line)for line in (run/'events.jsonl').read_text().splitlines()]
            output=pd.read_csv(task.parent/'submission.csv',dtype={'id':str})
            self.assertEqual(len(output),30)
            self.assertTrue(np.isfinite(output['label']).all())
            exports=list(run.glob('exports/final-*/graph.json'));self.assertEqual(len(exports),1)
            graph=json.loads(exports[0].read_text())
            self.assertEqual(len(graph['branches']),3)
            self.assertTrue(any(i['status']=='confirmed' for i in graph['ideas']))
            self.assertTrue(any(b['decision']=='reject' and b['state']=='active' for b in graph['branches']))
            self.assertTrue(any(t['transfer_source_trial_id']for t in graph['component_trials']))
            self.assertTrue(any(e['event']=='graph.committed' and e.get('operation')=='G08' for e in events))
            self.assertTrue(all(w['state']in{'completed','stopped','revoked','expired'}for w in graph['work_orders']))
            self.assertTrue(all(b['state']!='pending' for b in graph['bindings']))
            ranking=json.loads(next(run.glob('exports/final-*/ranking.json')).read_text())
            self.assertEqual(ranking['rows'][0]['experiment_id'],ranking['best_evaluated_experiment_id'])
            self.assertLess(ranking['rows'][0]['primary_metric']['value'],20.)
            destination=root/'reconstructed'
            result=reconstruct(run,ranking['best_evaluated_experiment_id'],destination,task.parent)
            self.assertFalse(result['executes_training'])
            process=subprocess.run([sys.executable,str(destination/'run_refit.py')],cwd=destination,text=True,capture_output=True,timeout=35)
            self.assertEqual(process.returncode,0,process.stderr[-4000:])
            reproduced=pd.read_csv(destination/'candidate.csv',dtype={'id':str})
            self.assertEqual(reproduced['id'].tolist(),output['id'].tolist())
            np.testing.assert_allclose(reproduced['label'],output['label'],rtol=1e-9,atol=1e-9)
