"""Replay a flat ZIP with scripted LLM, real Python EDA, checkpoint and training.

The child sees only root files copied by the platform. No repository PYTHONPATH.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
CHILD=r'''
import builtins,json,os,runpy,sys
original_import=builtins.__import__
installed=False
contexts={}
def importing(name,*args,**kwargs):
    global installed
    module=original_import(name,*args,**kwargs)
    if name=='agent.runtime' and not installed:
        installed=True
        cls=sys.modules['agent.llm.gateway'].MockTransport
        original=cls.complete
        def complete(self,request,timeout,context):
            if context['metadata']['schema_id']=='RepairEngineer-v1':
                packet=json.loads(request['messages'][1]['content'])
                candidate=dict(packet['original'])
                candidate.update(packet['allowed_edits'])
                value={'status':'repaired','candidate':candidate,'reason':'Restore the declared empty service fields.'}
                return {'choices':[{'message':{'content':json.dumps(value)},'finish_reason':'stop'}],
                        'usage':{'prompt_tokens':100,'completion_tokens':100}}
            if context['metadata']['schema_id']!='ResearchAction-v1':return original(self,request,timeout,context)
            user=json.loads(context['messages'][1]['content'])
            ident=user['work_order_id']
            if ident not in contexts:contexts[ident]=len(contexts)
            stage=contexts[ident]
            replies=[json.loads(m['content']) for m in request['messages'] if m['role']=='tool']
            evidence=[r['artifact_id'] for r in replies if r.get('artifact_id')]
            columns=['x_numeric','z_numeric']
            inv={'question':'Does the named interaction warrant testing a representation?',
                 'basis_ids':[],'columns':columns,'interpretation':'exploratory',
                 'mechanism':'Check fit support before measuring the joint representation.',
                 'assumptions':['Synthetic execution demonstration.'],
                 'expected_observation':'A readable fit-only report and independent model evaluation.',
                 'alternative_explanation':'A valid analysis may provide no predictive improvement.',
                 'decision_rule':'Retain only an independently evaluated configuration.'}
            if stage<2 and not replies and request.get('tools'):
                code=('def analyze(X_fit,y_fit,X_test,context):\n    return {"rows":len(X_fit),"missing":X_fit.isna().sum(),"target_present":y_fit is not None}\n'
                      if stage==0 else
                      'def analyze(X_fit,y_fit,X_test,context):\n    return {"product_quantiles":(X_fit.iloc[:,0]*X_fit.iloc[:,1]).quantile([0,.5,1])}\n')
                args={'question':inv['question'],'code':code,'columns':columns,'include_target':True}
                message={'tool_calls':[{'id':'python-check-'+str(stage),'type':'function','function':{'name':'run_analysis','arguments':json.dumps(args)}}]}
            else:
                recipe='linear-regressor-v1' if context['metadata']['demo_spec']['draft']['task_kind']=='regression' else 'linear-classifier-v1'
                value={'action':'research' if stage==0 else 'features' if stage==1 else 'stop',
                       'hypothesis':'Save the fit observation and continue.' if stage==0 else 'Measure a named product representation.',
                       'evidence_ids':evidence,'investigation':inv}
                if stage==1:value.update(recipe_id=recipe,hyperparameters={},feature_program=[{'op':'product','name':'joint_signal','columns':columns}])
                if stage==1 and os.environ.get('VERIFY_REPAIR_ENGINEER')=='1':value['code']=''
                if stage==0:value['next_step']='continue'
                message={'content':json.dumps(value)}
            return {'choices':[{'message':message,'finish_reason':'stop'}], 'usage':{'prompt_tokens':100,'completion_tokens':100}}
        cls.complete=complete
    return module
builtins.__import__=importing
sys.argv=['solution.py',sys.argv[1],'--llm-mode','mock']
runpy.run_path('solution.py',run_name='__main__')
'''


def verify(archive,directory,kind,exercise_engineer=False):
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from scripts.make_demo import make_demo
    from scripts.analyze_run import inspect_run
    directory.mkdir(parents=True,exist_ok=False)
    task=make_demo(directory/'task',kind,rows=480,test_rows=48)
    platform=directory/'platform';platform.mkdir()
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if '/' in name or not (name.endswith('.py') or name=='requirements.txt'):raise ValueError('Not a flat platform ZIP')
            (platform/name).write_bytes(z.read(name))
    env={k:v for k,v in os.environ.items() if not k.startswith(('OPENAI_','LLM_'))}
    env.update(PYTHONPATH='',LLM_MODE='mock',RESEARCH_MODE='adaptive',RESEARCH_PROTOCOL='compact',
               DATA_RESEARCH_MIN_SECONDS='0',TIME_BUDGET_SECONDS='90',MAX_WORKERS='1',GRAPH_PEERS='1',
               TRAIN_CPU_THREADS='1',TRAIN_MEMORY_MB='1024',CATALOG_WARMUP_EXPERIMENTS='0',
               LLM_MAX_CALLS='12',LLM_BOOTSTRAP_OUTPUT_TOKENS='1024',LLM_MAX_OUTPUT_TOKENS='1024',LLM_MAX_OUTPUT_TOKENS_LIMIT='2048',
               GRAPH_MIN_FREE_DISK_BYTES='0',WORK_DIR=str(directory/'runs'))
    if exercise_engineer:env.update(VERIFY_REPAIR_ENGINEER='1',LLM_REPAIR_ENGINEER_ENABLED='1')
    else:env.pop('VERIFY_REPAIR_ENGINEER',None)
    with (directory/'terminal.log').open('w') as stream:
        proc=subprocess.run([sys.executable,'-c',CHILD,str(task)],cwd=platform,env=env,
                            stdout=stream,stderr=subprocess.STDOUT,timeout=150)
    runs=sorted((directory/'runs').glob('run-*'))
    report=inspect_run(runs[-1])[0] if runs else {}
    events=[json.loads(l) for l in (runs[-1]/'events.jsonl').read_text().splitlines()] if runs else []
    counts={name:sum(e.get('event')==name for e in events) for name in [
        'research.python_analysis_completed','research.checkpoint_saved','research.checkpoint_completed',
        'research.action_proposed','experiment.evaluated','llm.repair_engineer_accepted','llm.repair_engineer_started']}
    analyses=[e for e in events if e.get('event')=='research.python_analysis_completed']
    # Background tuning may evaluate further configurations of the same program.
    # Verify the submitted representation ran, rather than an exact trial count.
    feature_evaluations=[e for e in events if e.get('event')=='experiment.evaluated'
        and any(step.get('name')=='joint_signal' and step.get('op')=='product'
                for step in e.get('hyperparameters',{}).get('feature_program',[]))]
    with (task.parent/'test.csv').open(newline='') as stream:test=list(csv.DictReader(stream))
    output=task.parent/'submission.csv'
    with output.open(newline='') as stream:
        reader=csv.DictReader(stream);columns=reader.fieldnames;submission=list(reader)
    output_valid=(columns==['id','label'] and [r['id'] for r in submission]==[r['id'] for r in test])
    if kind=='regression':output_valid=output_valid and all(math.isfinite(float(r['label'])) for r in submission)
    else:
        with (task.parent/'train.csv').open(newline='') as stream:labels={r['label'] for r in csv.DictReader(stream)}
        output_valid=output_valid and all(r['label'] in labels for r in submission)
    result={'kind':kind,'exit_code':proc.returncode,'state':report.get('state'),
            'events':counts,'output_valid':output_valid,'report':report,
            'scope':'scripted decisions; real EDA/fit/evaluation/publication'}
    (directory/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    if (proc.returncode or report.get('state')!='completed' or not output_valid
            or len(analyses)!=2 or not all(e.get('success') for e in analyses)
            or counts['research.checkpoint_saved']!=1 or not feature_evaluations
            or (exercise_engineer and (counts['llm.repair_engineer_accepted']!=1 or counts['llm.repair_engineer_started']!=1))):
        raise RuntimeError('Flat ZIP replay failed; see '+str(directory/'terminal.log'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--zip',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--exercise-engineer',action='store_true',help='Inject one interface error and require a repaired feature experiment in each task')
    a=p.parse_args();results=[]
    for kind in ['regression','classification']:
        result=verify(a.zip.resolve(),a.directory.resolve()/kind,kind,a.exercise_engineer);results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='report'}),flush=True)
    (a.directory/'verification.json').write_text(json.dumps({'zip_sha256':hashlib.sha256(a.zip.read_bytes()).hexdigest(),'results':results},ensure_ascii=False,indent=2))
