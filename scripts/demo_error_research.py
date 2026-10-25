"""Short scripted real fit/evaluate/refit checks; no paid API or ML-quality claim."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.llm.gateway import MockTransport
from agent.runtime import main
from scripts.make_demo import make_demo


def run(kind, destination, seconds=60):
    original = MockTransport.complete
    submitted, analyzed = set(), set()
    def complete(transport, request, timeout, context):
        if not context['metadata'].get('pipeline_execution'):
            return original(transport,request,timeout,context)
        peer = context['metadata']['researcher_id']
        user = json.loads(context['messages'][1]['content'])
        spec = user['task_contract']['task_spec']
        if peer not in analyzed and request.get('tools'):
            analyzed.add(peer)
            column = next(c for c in user['profile_summary']['data']['train_columns'] if c not in [spec['target_column'],*spec['excluded_features']])
            query = {'question':'Inspect errors or the available fit inputs on the synthetic leader', 'columns':[column],
                'include_target':True,
                'code':'def analyze(X,y,T,context):\n    o=context.get("oof")\n    return {"rows":len(X),"aligned":X.index.equals(o["fold_ids"].index) if o else X.index.equals(y.index),"oof_available":o is not None}\n'}
            if user['profile_summary'].get('scientific_protocol',{}).get('fit_diagnostics'):
                query['experiment_id']='best'
            message = {'role':'assistant','content':None,'tool_calls':[{'id':'synthetic-analysis','type':'function',
                'function':{'name':'run_analysis','arguments':json.dumps(query)}}]}
        else:
            if peer in submitted:
                payload = {'action':'stop','hypothesis':'The finite scripted model has no additional source proposals.'}
            else:
                submitted.add(peer)
                recipe = 'linear-regressor-v1' if kind=='regression' else 'linear-classifier-v1'
                param = 'alpha' if kind=='regression' else 'C'
                source = ('from agent.ml import build_pipeline\nDEFAULT_PARAMS = '+repr({param:1.,'numeric_interactions':peer.endswith('2')})+
                    '\nSEARCH_SPACE = '+repr({param:{'type':'float','low':.1,'high':10.,'log':True}})+
                    '\ndef make_pipeline(task_spec,params):\n    return build_pipeline('+repr(recipe)+',task_spec,params)\n')
                payload = {'action':'pipeline','code':source}
            message = {'role':'assistant','content':json.dumps(payload)}
        return {'id':'scripted-error-research','choices':[{'finish_reason':'stop','message':message}]}
    task = make_demo(destination/'task',kind,rows=360,test_rows=30)
    settings = {'LLM_MODE':'mock','TIME_BUDGET_SECONDS':str(seconds),'RESEARCH_MODE':'adaptive',
        'BACKGROUND_TUNING_ENABLED':'1','BACKGROUND_MAX_TRIALS':'2','BACKGROUND_TRIAL_SECONDS':'15',
        'CATALOG_WARMUP_EXPERIMENTS':'1','DATA_RESEARCH_MIN_SECONDS':'0','GRAPH_REFINEMENT_ENABLED':'0',
        'GRAPH_HARD_DECISIONS':'0','GRAPH_MIN_FREE_DISK_BYTES':'0','LOG_LEVEL':'ERROR',
        'FEATURE_RESEARCH_ENABLED':'0','SCIENTIFIC_RESEARCH_ENABLED':'1', 'RESEARCH_OOF_BUDGET_FRACTION':'.1',
        'LLM_MAX_TOTAL_OUTPUT_TOKENS':'0','LLM_MAX_CALLS':'20','EXPERIMENT_TIMEOUT_SECONDS':'20',
        'MODEL_SNAPSHOT_MAX_BYTES':'0'}
    with patch.dict(os.environ,settings),patch.object(MockTransport,'complete',complete):
        status = main([str(task),'--work-dir',str(destination/'runs'),'--llm-mode','mock'])
    run_path = max((destination/'runs').glob('run-*'))
    events = [json.loads(line) for line in (run_path/'events.jsonl').read_text().splitlines()]
    names = ['pipeline.version_registered','pipeline.execution_started','pipeline.batch_extended',
        'diagnostics.oof_started','diagnostics.oof_completed','research.python_analysis_completed',
        'research.paired_comparison','research.proposal_failed','submission.published','validation.finalists_confirmed']
    result = {'task_kind':kind,'status':status,'run':str(run_path),'csv_exists':(task.parent/'submission.csv').is_file(),
        'events':{name:sum(e['event']==name for e in events) for name in names},
        'errors':[{'event':e['event'],'message':e.get('message',e.get('reason'))} for e in events
                  if e['level']=='ERROR' and not (e['event']=='job.finished' and e.get('state')=='cancelled')],
        'limitations':'Scripted replies, real fits and independent scores; not a long autonomous Qwen quality evaluation.'}
    print(json.dumps(result,ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['regression','classification'])
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seconds',default=60,type=int)
    args=parser.parse_args()
    result=run(args.kind,args.output,args.seconds)
    raise SystemExit(0 if result['status']==0 and result['csv_exists'] else 1)
