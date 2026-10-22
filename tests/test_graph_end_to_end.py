from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from agent.catalog import Catalog
from agent.llm.gateway import MockTransport
from agent.runtime import main
from agent.runner import Runner
from agent.orchestrator import Orchestrator
from agent.graph.service import GraphService
from agent.errors import AgentError
from scripts.make_demo import make_demo
from scripts.reconstruct_solution import reconstruct


class GraphEndToEndTests(unittest.TestCase):
    def test_warmup_transfer_forks_from_measured_bases_without_a_research_api_call(self):
        import numpy as np
        import pandas as pd
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=make_demo(root/'task','regression',rows=120,test_rows=20)
            for name in ['train.csv','test.csv']:
                frame=pd.read_csv(task.parent/name,dtype={'id':str},keep_default_na=False)
                x=np.linspace(0,3,len(frame));frame['x_numeric']=x;frame['z_numeric']=0.;frame['category']='same'
                if name=='train.csv':frame['label']=np.exp(2*x)
                frame.to_csv(task.parent/name,index=False)
            def warmups(catalog):
                return [('linear-regressor-v1', catalog.normalize('linear-regressor-v1', {'alpha': a, 'target_transform': t}))
                        for t in ['identity','shifted_log1p'] for a in [1.,3.]]
            original_update=Orchestrator.update_final_reserve
            def finish_after_measured_fork(controller):
                original_update(controller)
                if controller.graph.db.execute('SELECT count(*) FROM g_branches').fetchone()[0] == 3:
                    controller.budget.deadline=time.monotonic()+controller.budget.reserve_seconds()+.5
            settings={'LLM_MODE':'mock','GRAPH_ENABLED':'1','TIME_BUDGET_SECONDS':'120','MAX_RESEARCH_EXPERIMENTS':'6',
                'CATALOG_WARMUP_EXPERIMENTS':'4','LLM_MAX_CALLS':'2','LOG_LEVEL':'ERROR','RESEARCH_MODE':'catalog',
                'GRAPH_MIN_FREE_DISK_BYTES':'0','MODEL_SNAPSHOT_MAX_BYTES':'0'}
            with patch.dict(os.environ,settings),patch.object(Catalog,'warmup_choices',warmups),patch.object(Orchestrator,'update_final_reserve',finish_after_measured_fork),redirect_stderr(io.StringIO()) as stderr:
                code=main([str(task),'--llm-mode','mock','--work-dir',str(root/'runs')])
            self.assertEqual(code,0,stderr.getvalue()[-4000:])
            run=next((root/'runs').glob('run-*'))
            graph=json.loads(next(run.glob('exports/final-*/graph.json')).read_text())
            self.assertEqual(len(graph['branches']),3)
            self.assertEqual(len(graph['branch_seeds']),4)
            plans={e['experiment_id']:e['plan'] for e in graph['experiments']}
            self.assertTrue(all(plans[s['head']['experiment_id']]['model']['family']=='linear' for s in graph['branch_seeds']))
            accept=next(b['branch_id'] for b in graph['branches'] if b['decision']=='accept')
            self.assertTrue(all(plans[s['head']['experiment_id']]['model']['hyperparameters']['target_transform']=='shifted_log1p'
                                for s in graph['branch_seeds'] if s['branch_id']==accept))
            events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(e['event']=='llm.call_started' for e in events),2)
            self.assertTrue(any(e['event']=='research.component_transfer_prepared' and e['purpose']=='pilot' for e in events))
            self.assertTrue((task.parent/'submission.csv').is_file())

    def test_no_research_api_quota_finishes_cleanly_with_the_verified_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=make_demo(root/'task')
            settings={'LLM_MODE':'mock','GRAPH_ENABLED':'1','TIME_BUDGET_SECONDS':'60','MAX_RESEARCH_EXPERIMENTS':'3',
                'CATALOG_WARMUP_EXPERIMENTS':'0','LLM_MAX_CALLS':'2','LOG_LEVEL':'ERROR','RESEARCH_MODE':'catalog','GRAPH_MIN_FREE_DISK_BYTES':'0',
                'MODEL_SNAPSHOT_MAX_BYTES':'0'}
            with patch.dict(os.environ,settings),redirect_stderr(io.StringIO()) as stderr:
                code=main([str(task),'--llm-mode','mock','--work-dir',str(root/'runs')])
            self.assertEqual(code,0,stderr.getvalue()[-4000:]);self.assertTrue((task.parent/'submission.csv').is_file())
            run=next((root/'runs').glob('run-*'))
            self.assertIn('State: completed',(run/'summary.md').read_text())
            events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
            finish=next(e for e in events if e['event']=='search.finished')
            self.assertIn(finish['stop_reason'], {'time_reserve','no_affordable_candidates'})
            self.assertTrue(any(e['event']=='search.local_trial' for e in events))
            self.assertFalse(any(e['event']=='research.proposal_failed' for e in events))
            self.assertFalse(list(run.rglob('*.joblib')))
            self.assertTrue(any(e['event']=='job.model_snapshot' and e['status']=='disabled' for e in events))

    def test_spawn_failure_after_baseline_closes_assignments_and_retains_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);task=make_demo(root/'task')
            original=Runner.start_train
            failures=[]
            def fail_research(runner,payload,spec,experiment):
                if payload['plan']['model']['family']!='dummy':
                    failures.append(experiment['experiment_id'])
                    if len(failures)>=2:
                        runner.budget.deadline=time.monotonic()+runner.budget.reserve_seconds()+.5
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
        self.run_branch_search(refinement=False)

    def test_real_training_refines_best_ancestor_and_retains_csv_after_worse_final_trials(self):
        self.run_branch_search(refinement=True)

    def run_branch_search(self, refinement):
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
            original_catalog=Catalog.__init__
            def linear_catalog(catalog,*args):
                original_catalog(catalog,*args)
                catalog.recipes={k:v for k,v in catalog.recipes.items() if v['family'] in {'dummy','linear'}}
            proposal_counts={}
            original_update=Orchestrator.update_final_reserve
            def finish_after_backtrack(controller):
                original_update(controller)
                graph=controller.graph
                active=graph.db.execute("SELECT branch_id FROM g_branches WHERE state='active' AND decision='reject'").fetchone()
                phase=graph.refinement()
                final_evals=graph.db.execute('SELECT count(DISTINCT experiment_id) FROM g_branch_evaluations WHERE run_id=? AND event_seq>?',(graph.run_id,phase['started_event_seq'])).fetchone()[0] if phase else 0
                if (not refinement and active and graph.branch_trial_count(active[0])>0) or (refinement and final_evals>=2):
                    controller.budget.deadline=time.monotonic()+controller.budget.reserve_seconds()+.5
            original_due=GraphService.refinement_due
            def measured_phase_boundary(graph,pending=0):
                # The exact 85% arithmetic is unit-tested; advance the phase clock
                # after real branch/backtrack evidence instead of idling in CI.
                active=graph.db.execute("SELECT branch_id FROM g_branches WHERE state='active' AND decision='reject'").fetchone()
                if refinement and not graph.refinement() and active and graph.branch_trial_count(active[0])>0:
                    return 'search_time_share_reached'
                return original_due(graph,pending)
            def response(transport,request,timeout,context):
                if context['metadata']['schema_id']!='ResearchChoice-v1':
                    return original(transport,request,timeout,context)
                user=json.loads(context['messages'][1]['content']);peer=context['metadata']['researcher_id']
                proposal_counts[peer]=proposal_counts.get(peer,0)+1
                variant=proposal_counts[peer] if refinement else 0
                restrictions=user['restrictions'];required=restrictions['required_capability_ids'];forbidden=restrictions['forbidden_capability_ids']
                if user.get('research_phase') == 'refine':
                    alpha=9000.+int(peer[-1]);transform='shifted_log1p'
                elif required:
                    alpha=(8000+variant if refinement else 10000)-int(peer[-1]);transform='shifted_log1p'
                elif forbidden:
                    alpha=.01+int(peer[-1])*.01+variant*.001;transform='identity'
                else:
                    alpha=1. if peer=='researcher-1' else 3.
                    transform='shifted_log1p' if peer=='researcher-1' else 'identity'
                result={'recipe_id':'linear-regressor-v1','hyperparameters':{'alpha':alpha,'target_transform':transform},
                    'hypothesis':'Synthetic controlled test of an approved target transform','expected_effect':'Verify transfer through independently evaluated real predictions'}
                return {'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':json.dumps(result)}}]}
            def warmups(catalog):
                return [('linear-regressor-v1',catalog.normalize('linear-regressor-v1',{'alpha':a}))for a in [1.,3.]]
            settings={'LLM_MODE':'mock','GRAPH_ENABLED':'1','TIME_BUDGET_SECONDS':'180','MAX_RESEARCH_EXPERIMENTS':'12' if refinement else '8',
                'CATALOG_WARMUP_EXPERIMENTS':'2','MAX_WORKERS':'2','GRAPH_PEERS':'2','GRAPH_TRAIN_WORKERS':'1',
                'GRAPH_PLATEAU_WINDOW':'1','GRAPH_PLATEAU_SECONDS':'1' if refinement else '0','GRAPH_MIN_FREE_DISK_BYTES':'0',
                'GRAPH_REFINEMENT_ENABLED':'1' if refinement else '0',
                'RESEARCH_SERIES_SIZE':'1',
                'GRAPH_MIN_EFFECT':'0.001','LOG_LEVEL':'ERROR','EXPERIMENT_TIMEOUT_SECONDS':'30','RESEARCH_MODE':'catalog'}
            with patch.dict(os.environ,settings),patch.object(MockTransport,'complete',response),patch.object(Catalog,'__init__',linear_catalog),patch.object(Catalog,'warmup_choices',warmups),patch.object(Catalog,'assignment_recipes',return_value=['linear-regressor-v1']),patch.object(GraphService,'refinement_due',measured_phase_boundary),patch.object(Orchestrator,'update_final_reserve',finish_after_backtrack),redirect_stderr(io.StringIO())as stderr:
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
            if refinement:
                phase=graph['refinement'];self.assertIsNotNone(phase,json.dumps([e for e in events if e['event'] in {'search.finished','research.proposal_failed'}])[-4000:])
                selected=next(b for b in graph['branches'] if b['branch_id']==phase['branch_id'])
                self.assertEqual(selected['decision'],'root')
                self.assertEqual(selected['state'],'active')
                self.assertTrue(all(b['state']=='archived' for b in graph['branches'] if b['branch_id']!=selected['branch_id']))
                graph_events=[json.loads(line) for line in next(run.glob('exports/final-*/graph-events.jsonl')).read_text().splitlines()]
                self.assertTrue(any(e['record_schema']=='GBranch' and e['record']['branch_id']==selected['branch_id'] and e['record']['state']=='sealed' for e in graph_events))
                self.assertEqual(sum(e['event']=='search.phase_changed' for e in events),1)
                final_ids={e['record']['work_order_id'] for e in graph_events if e['record_schema']=='GWorkOrder' and e['event_seq']>phase['started_event_seq']}
                final_orders=[w for w in graph['work_orders'] if w['work_order_id'] in final_ids]
                self.assertEqual({w['researcher_id'] for w in final_orders},{'researcher-1','researcher-2'})
                self.assertTrue(all(w['branch_id']==selected['branch_id'] for w in final_orders))
                self.assertTrue(all(w['base_experiment_id']==phase['source_experiment_id'] for w in final_orders))
            else:
                self.assertTrue(any(b['decision']=='reject' and b['state']=='active' for b in graph['branches']))
            self.assertTrue(any(t['transfer_source_trial_id']for t in graph['component_trials']))
            self.assertTrue(any(e['event']=='graph.committed' and e.get('operation')=='G08' for e in events))
            self.assertTrue(all(w['state']in{'completed','stopped','revoked','expired'}for w in graph['work_orders']))
            self.assertTrue(all(b['state']!='pending' for b in graph['bindings']))
            ranking=json.loads(next(run.glob('exports/final-*/ranking.json')).read_text())
            self.assertEqual(ranking['rows'][0]['experiment_id'],ranking['best_evaluated_experiment_id'])
            self.assertLess(ranking['rows'][0]['primary_metric']['value'],20.)
            if refinement:
                self.assertEqual(ranking['best_evaluated_experiment_id'],phase['source_experiment_id'])
                self.assertEqual(ranking['published_champion']['experiment_id'],phase['source_experiment_id'])
            destination=root/'reconstructed'
            result=reconstruct(run,ranking['best_evaluated_experiment_id'],destination,task.parent)
            self.assertFalse(result['executes_training'])
            process=subprocess.run([sys.executable,str(destination/'run_refit.py')],cwd=destination,text=True,capture_output=True,timeout=35)
            self.assertEqual(process.returncode,0,process.stderr[-4000:])
            reproduced=pd.read_csv(destination/'candidate.csv',dtype={'id':str})
            self.assertEqual(reproduced['id'].tolist(),output['id'].tolist())
            np.testing.assert_allclose(reproduced['label'],output['label'],rtol=1e-9,atol=1e-9)
