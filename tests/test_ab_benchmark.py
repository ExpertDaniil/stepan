"""Validate the benchmark boundary, checkpoint causality and actual child adapters."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from scripts.ab_benchmark import prepare, child_env, score_csv, summarize, activity_seconds
from scripts.make_demo import make_demo


def arguments(directory, kind='regression'):
    return argparse.Namespace(out=directory,task=None,target=None,id_column=None,
        demo=True,kind=kind,metric=None,audit_fraction=.2,seed=42,mock=True,
        seconds=30,checkpoints=[5,15,30],order='A,B',env_file=None)


class BenchmarkTests(unittest.TestCase):
    def test_overlapping_api_time_is_not_double_counted_as_idle(self):
        events=[]
        for second,event,fields in [(0,'llm.call_started',{'call_no':1}),
            (1,'llm.call_started',{'call_no':2}),(2,'job.started',{'job_id':'fit'}),
            (4,'llm.call_completed',{'call_no':1}),(5,'job.finished',{'job_id':'fit'}),
            (6,'llm.call_failed',{'call_no':2})]:
            events.append({'timestamp':f'2026-01-01T00:00:0{second}+00:00','event':event,**fields})
        result=activity_seconds(events)
        self.assertEqual(result['api_pending_wall_seconds'],6)
        self.assertEqual(result['api_pending_without_worker_seconds'],3)
        self.assertEqual(result['worker_active_wall_seconds'],3)

    def test_disjoint_equal_inputs_and_private_labels_not_in_child_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, protocol, controlled = prepare(arguments(Path(tmp)/'ab','classification'))
            train = pd.read_csv(root/'A/train.csv',dtype={'id':str})
            audit = pd.read_csv(root/'private/audit-labels.csv',dtype={'id':str})
            self.assertFalse(set(train.id) & set(audit.id))
            self.assertEqual(len(train)+len(audit),360)
            for name in protocol['input_sha256']:
                self.assertEqual((root/'A'/name).read_bytes(),(root/'B'/name).read_bytes())
            self.assertNotIn('label',pd.read_csv(root/'B/test.csv').columns)
            self.assertNotIn('audit-labels', (root/'B/task.md').read_text())
            self.assertNotIn('private', (root/'B/contract.json').read_text())
            self.assertFalse((root/'A/private').exists())
            self.assertGreater(len(controlled),30)

    def test_explicit_identity_and_rare_class_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = make_demo(Path(tmp)/'source','classification')
            frame = pd.read_csv(task.parent/'train.csv')
            frame.loc[0,'label'] = 'singleton'
            frame.to_csv(task.parent/'train.csv',index=False)
            args = arguments(Path(tmp)/'ab','classification')
            args.demo=False;args.task=task;args.target='label';args.id_column='id'
            with self.assertRaisesRegex(ValueError,'observations per class'):prepare(args)

    def test_child_environment_drops_local_quota_and_keeps_credentials(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            args = arguments(Path(tmp)/'ab');_,_,controlled=prepare(args)
            with patch.dict(os.environ,{'LLM_MAX_CALLS':'1','TIME_BUDGET_SECONDS':'3',
                                       'OPENAI_API_KEY':'synthetic-preserved'}):
                env = child_env(args,controlled)
            self.assertEqual(env['LLM_MAX_CALLS'],'0')
            self.assertEqual(env['TIME_BUDGET_SECONDS'],'30')
            self.assertEqual(env['OPENAI_API_KEY'],'synthetic-preserved')

    def test_audit_scoring_checks_order_scale_and_unknown_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'pred.csv'
            truth=pd.DataFrame({'id':['001','002'],'value':[2.,4.]})
            protocol={'id_column':'id','target':'value','kind':'regression'}
            truth.to_csv(p,index=False)
            self.assertEqual(score_csv(p,truth,protocol)['rmse'],0)
            truth.iloc[::-1].to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'order'):score_csv(p,truth,protocol)
            pd.DataFrame({'id':['001','002'],'value':[2.,float('inf')]}).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'Non-finite'):score_csv(p,truth,protocol)
            truth['value']=['one','two'];protocol['kind']='classification'
            pd.DataFrame({'id':['001','002'],'value':['one','other']}).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'Unknown'):score_csv(p,truth,protocol)

    def test_scores_stay_closed_until_both_exit_and_never_use_future_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,protocol,_=prepare(arguments(Path(tmp)/'ab'))
            with self.assertRaisesRegex(ValueError,'Both runs'):summarize(root)
            truth=pd.read_csv(root/'private/audit-labels.csv',dtype={'id':str})
            for mode in ['A','B']:
                case=root/mode;(case/'publications').mkdir()
                truth.to_csv(case/'publications/late.csv',index=False)
                (case/'publications/index.jsonl').write_text(json.dumps({'ready_seconds':10,'csv':'late.csv'})+'\n')
                (case/'exit.json').write_text('{"exit_code":0}')
            result=summarize(root)
            self.assertIsNone(result['comparison'][0]['A'])
            self.assertEqual(result['comparison'][1]['A'],0)
            self.assertIsNone(result['modes']['A']['final_audit_metrics'])
            (root/'A/test.csv').write_text('changed\n')
            with self.assertRaisesRegex(ValueError,'Frozen input'):summarize(root)


class BenchmarkProcessTests(unittest.TestCase):
    def test_timeout_probe_reproduces_missing_initial_csv_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'probe'
            result=subprocess.run([sys.executable,'scripts/ab_benchmark.py','probe','--out',str(out)],
                cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=45)
            self.assertEqual(result.returncode,2,result.stderr)
            report=json.loads((out/'result/probe.json').read_text())
            self.assertEqual(report['network_calls'],0)
            self.assertTrue(report['bootstrap_failed'])
            self.assertFalse(report['protected_against_startup_timeout'])


if __name__ == '__main__':unittest.main()
