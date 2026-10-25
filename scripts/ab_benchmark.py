"""Sequential catalogue/Optuna versus LLM research benchmark, with outer audit.

Linux/Python 3.12. Default: two 30-minute runs, no leaderboard submissions.
Audit scores are calculated only after BOTH children have stopped.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from tools.files import resolve_inputs
    from agent.task import METRICS
    from agent.config import Config
    from dataclasses import fields
    from scripts.make_demo import make_demo
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)  # Never overwrite previous evidence.
    if args.demo:
        source_task = make_demo(root/'source-demo', args.kind, rows=360, test_rows=30)
        args.target, args.id_column = 'label', 'id'
    else:
        if not args.task or not args.target:
            raise ValueError('Real task requires --task and explicit --target')
        source_task = args.task.resolve()
    text, paths = resolve_inputs(source_task)
    # Preserve categorical tokens such as "NA" and identifier leading zeros.
    train = pd.read_csv(paths['train'], keep_default_na=False, na_values=[''],
                        dtype={args.id_column:str} if args.id_column else None)
    if args.target not in train or train[args.target].isna().any():
        raise ValueError('Target missing or contains null values')
    classification = args.kind == 'classification'
    metric = args.metric or ('f1_macro' if classification else 'rmse')
    if metric not in ({'f1_macro','accuracy'} if classification else {'rmse','mae','r2'}):
        raise ValueError('Unsupported metric for benchmark task kind')
    if classification:
        train[args.target] = train[args.target].astype(str)
        if train[args.target].value_counts().min() < 5:
            raise ValueError('Need >=5 observations per class for outer and inner stratification')
    elif not np.isfinite(pd.to_numeric(train[args.target], errors='raise')).all():
        raise ValueError('Regression target must be finite numeric values')
    ident = args.id_column or '__ab_row_id'
    if args.id_column:
        if ident not in train or train[ident].isna().any() or train[ident].duplicated().any():
            raise ValueError('--id-column must name a non-null unique identifier')
    else:
        if ident in train:
            raise ValueError('Reserved benchmark ID column already exists')
        train.insert(0, ident, [str(i) for i in range(len(train))])
    fit, audit = train_test_split(np.arange(len(train)), test_size=args.audit_fraction,
        random_state=args.seed, stratify=train[args.target] if classification else None)
    fit, audit = sorted(map(int,fit)), sorted(map(int,audit))
    if set(fit) & set(audit):
        raise AssertionError('Outer audit overlaps training')
    private = root/'private';private.mkdir(mode=0o700)
    truth = train.iloc[audit][[ident,args.target]]
    truth.to_csv(private/'audit-labels.csv',index=False)
    os.chmod(private/'audit-labels.csv', 0o600)
    task_kind = ('binary_classification' if train[args.target].nunique()==2 else 'multiclass_classification') if classification else 'regression'
    direction, prediction = METRICS[metric]
    draft = {'task_kind':task_kind, 'target_column':args.target,
        'metric':{'metric_id':metric,'direction':direction,'prediction_kind':prediction,'parameters':{}},
        'output':{'path':'submission.csv','columns':[ident,args.target], 'id_columns':[ident],
            'prediction_columns':[args.target], 'row_order':'test','encoding':'utf-8','separator':',','write_index':False}}
    validation = {'strategy':'stratified_holdout' if classification else 'random_holdout',
        'parameters':{'valid_fraction':.2},'seed':args.seed,'reason':'Frozen inner benchmark protocol; independent outer labels withheld.'}
    contract = {'draft':draft,'validation':validation}
    # Drop demo instructions/old synthetic contract, and normalize CSV references.
    text = re.sub(r'```agent-demo\s*.*?```', '', text, flags=re.S)
    text = re.sub(r'`[^`\n]*[/\\]([^`/\\\n]+\.csv)`', r'`\1`', text)
    text = ('# Local A/B benchmark override\n\nThis local task uses reduced train.csv and a substituted test.csv. '
        'Original row counts/split descriptions below are historical and do not describe this local split. '
        f'Predict `{args.target}` using metric `{metric}`. Exclude identifier `{ident}` from features. '
        f'Write submission.csv with columns `{ident},{args.target}` in test order. '
        'Task interpretation and inner validation are fixed equally for both modes.\n\n'+text)
    if args.mock:
        text += '\n```agent-demo\n'+json.dumps({'draft':draft})+'\n```\n'
    cases = []
    for mode in ['A','B']:
        case = root/mode;case.mkdir()
        train.iloc[fit].to_csv(case/'train.csv', index=False)
        train.iloc[audit].drop(columns=args.target).to_csv(case/'test.csv', index=False)
        sample = truth.copy();sample[args.target] = train[args.target].iloc[fit[0]] if classification else 0.
        sample.to_csv(case/'sample_submission.csv',index=False)
        (case/'task.md').write_text(text,encoding='utf-8')
        if paths.get('dictionary'):
            shutil.copyfile(paths['dictionary'],case/'data_dictionary.csv')
        write_json(case/'contract.json',contract)
        cases.append(case)
    hashes = {p.name:digest(p) for p in cases[0].iterdir() if p.is_file()}
    if hashes != {p.name:digest(p) for p in cases[1].iterdir() if p.is_file()}:
        raise AssertionError('A/B inputs are different')
    controlled = {f.name.upper() for f in fields(Config)}
    controlled |= set(re.findall(r'[\"\']([A-Z][A-Z0-9_]+)[\"\']', (ROOT/'agent/config.py').read_text()))
    report = {'schema_version':1,'source_train_sha256':digest(Path(paths['train'])),
        'target':args.target,'id_column':ident,'kind':args.kind,'metric':metric,'direction':direction,
        'seed':args.seed,'fit_rows':len(fit),'audit_rows':len(audit),'input_sha256':hashes,
        'seconds_per_mode':args.seconds,'checkpoints':args.checkpoints,'order':args.order,
        'mock':args.mock,'bootstrap':'explicit common contract; LLM interpretation excluded from A/B',
        'audit_split':'stratified random' if classification else 'random',
        'limitations':['Single split/seed; no confidence or causal feature claim.',
            'Outer holdout features are visible as test; labels never supplied to agent.',
            'Controlled ablation: catalogue knowledge/OOF remain available in both modes.',
            'This is a local raw-metric experiment, not the normalized leaderboard score.']}
    try:
        report['commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except subprocess.CalledProcessError:
        report['commit'] = None
    report['runtime_sha256'] = {str(p.relative_to(ROOT)):digest(p) for folder in ['agent','tools'] for p in sorted((ROOT/folder).rglob('*.py'))}
    report['packages'] = {}
    for name in ['numpy','pandas','scikit-learn','catboost','optuna','openai']:
        try:report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:report['packages'][name] = None
    write_json(root/'protocol.json',report)
    write_json(private/'split.json', {'fit_indices':fit,'audit_indices':audit})
    return root, report, controlled


def child_env(args, controlled):
    # Existing environment beats explicit .env, as in the production CLI.
    env = dict(os.environ)
    if args.env_file:
        from agent.runtime import load_env_file
        before = dict(os.environ)
        try:
            load_env_file(args.env_file)
            env = dict(os.environ)
        finally:
            os.environ.clear();os.environ.update(before)
    # Eliminate hidden local knobs, including old LLM call/output quotas.
    env = {k:v for k,v in env.items() if k not in controlled}
    defaults = json.loads((ROOT/'recipes/platform-env.json').read_text())
    env.update(defaults)
    env.update(AUTOML_PROFILE='platform', LLM_MODE='mock' if args.mock else 'live',
        TIME_BUDGET_SECONDS=str(args.seconds), SEED=str(args.seed), MAX_WORKERS='2',
        TRAIN_CPU_THREADS='3', TRAIN_MEMORY_MB='10240', BACKGROUND_CPU_THREADS='6',
        CATALOG_WARMUP_EXPERIMENTS='4', RESEARCH_MODE='adaptive', RESEARCH_PROTOCOL='compact',
        BACKGROUND_TUNING_ENABLED='1', FEATURE_SEED_BUDGET_FRACTION='0',
        DATA_RESEARCH_MIN_SECONDS='0', GRAPH_REFINEMENT_ENABLED='0',
        GRAPH_MIN_FREE_DISK_BYTES='0', PYTHONPATH='',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    return env


def execute(root, mode, args, env):
    case = root/mode
    env = {**env,'WORK_DIR':str(case/'runs')}
    command = [sys.executable,str(ROOT/'scripts/ab_worker.py'),'--case',str(case),'--mode',mode]
    started = time.monotonic()
    with (case/'terminal.log').open('w') as stream:
        process = subprocess.Popen(command,cwd=case,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code = process.wait(timeout=args.seconds+60)
        except subprocess.TimeoutExpired:
            # Signal the full group so neither learner nor research subprocess survives.
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            code = 124
        except BaseException:
            os.killpg(process.pid,signal.SIGKILL);process.wait();raise
    result = {'exit_code':code,'wall_seconds':time.monotonic()-started}
    write_json(case/'exit.json',result)
    return result


def score_csv(path, truth, protocol):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, accuracy_score
    ident, target = protocol['id_column'], protocol['target']
    pred = pd.read_csv(path,keep_default_na=False,dtype={ident:str})
    if list(pred.columns) != [ident,target] or len(pred) != len(truth) or pred[ident].tolist() != truth[ident].tolist():
        raise ValueError('Audit prediction schema/order differs from frozen test IDs')
    if protocol['kind'] == 'regression':
        values = pd.to_numeric(pred[target],errors='raise')
        if not np.isfinite(values).all():raise ValueError('Non-finite audit predictions')
        metrics = {'rmse':float(np.sqrt(mean_squared_error(truth[target],values))),
            'mae':float(mean_absolute_error(truth[target],values)), 'r2':float(r2_score(truth[target],values))}
    else:
        y, values = truth[target].astype(str), pred[target].astype(str)
        labels = sorted(y.unique())
        if not set(values) <= set(labels):raise ValueError('Unknown audit prediction labels')
        metrics = {'f1_macro':float(f1_score(y,values,labels=labels,average='macro',zero_division=0)),
                   'accuracy':float(accuracy_score(y,values))}
    return metrics


def activity_seconds(events):
    """Union wall intervals: API pending with no train/refit/OOF job active."""
    changes = []
    for e in events:
        timestamp = e.get('timestamp')
        if not timestamp:continue
        t = datetime.fromisoformat(timestamp.replace('Z','+00:00')).timestamp()
        name = e['event']
        if name == 'llm.call_started':changes.append((t,'api',str(e['call_no']),True))
        if name in {'llm.call_completed','llm.call_failed'} and 'call_no' in e:
            changes.append((t,'api',str(e['call_no']),False))
        if name == 'job.started':changes.append((t,'job',e['job_id'],True))
        if name == 'job.finished':changes.append((t,'job',e['job_id'],False))
        if name == 'run.finished':changes.append((t,'end','',False))
    active = {'api':set(),'job':set()};previous=None
    values = {'api_pending_wall_seconds':0.,'api_pending_without_worker_seconds':0.,'worker_active_wall_seconds':0.}
    for t,kind,ident,start in sorted(changes):
        dt = t-previous if previous is not None else 0
        if active['api']:
            values['api_pending_wall_seconds'] += dt
            if not active['job']:values['api_pending_without_worker_seconds'] += dt
        if active['job']:values['worker_active_wall_seconds'] += dt
        if kind in active:
            (active[kind].add if start else active[kind].discard)(ident)
        previous=t
    return values


def summarize(root):
    import pandas as pd
    protocol = json.loads((root/'protocol.json').read_text())
    for mode in ['A','B']:
        if not (root/mode/'exit.json').exists():
            raise ValueError('Both runs must finish before opening audit labels')
        for name, expected in protocol['input_sha256'].items():
            if digest(root/mode/name) != expected:raise ValueError('Frozen input changed: '+mode+'/'+name)
    truth = pd.read_csv(root/'private/audit-labels.csv',keep_default_na=False,dtype={protocol['id_column']:str})
    reports = {}
    for mode in ['A','B']:
        case = root/mode
        index = case/'publications/index.jsonl'
        publications = [json.loads(l) for l in index.read_text().splitlines()] if index.exists() else []
        checkpoints = []
        for mark in protocol['checkpoints']:
            ready = [r for r in publications if r['ready_seconds'] <= mark]
            row = {'seconds':mark,'available':bool(ready)}
            if ready:
                last = ready[-1]
                row.update(last, audit_metrics=score_csv(case/'publications'/last['csv'],truth,protocol))
            checkpoints.append(row)
        final_path = case/'submission.csv'
        final = score_csv(final_path,truth,protocol) if final_path.exists() else None
        events = [json.loads(l) for path in (case/'runs').glob('*/events.jsonl') for l in path.read_text().splitlines()]
        # A valid process exit alone is insufficient. Read final persisted state
        # after shutdown; event logs can lack their last records.
        states=[]
        for database in (case/'runs').glob('*/state.sqlite'):
            with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
                states.extend(row[0] for row in db.execute('SELECT state FROM runs'))
        state=states[0] if len(states)==1 else None
        evaluations = [e for e in events if e['event']=='experiment.evaluated']
        calls = [e for e in events if e['event']=='llm.call_completed']
        llm_wait = sum(e.get('duration_ms',0)/1000 for e in calls)
        # Sum across overlapping requests, not wall-clock idle time.
        reports[mode] = {'exit':json.loads((case/'exit.json').read_text()),
            'state':state, 'state_source':'read-only runs table after process exit',
            'event_log_has_finish':any(e['event']=='run.finished' for e in events),
            'checkpoints':checkpoints,'final_audit_metrics':final,
            'event_counts':{name:sum(e['event']==name for e in events) for name in sorted({e['event'] for e in events})},
            'evaluations':[ {k:e.get(k) for k in ['experiment_id','recipe_id','score','hyperparameters']} for e in evaluations],
            'llm_completed_request_seconds_sum':llm_wait,
            'activity':activity_seconds(events),
            'llm_last_token_budget':next((e['token_budget'] for e in reversed(events) if 'token_budget' in e),None),
            'fit_job_seconds_sum':sum(e.get('duration_seconds',0) for e in events if e['event']=='job.finished' and e.get('phase')=='train'),
            'error_events':[{k:e.get(k) for k in ['event','code','message']} for e in events if e.get('level')=='ERROR']}
        if mode == 'A' and reports[mode]['event_counts'].get('llm.call_started',0):
            raise ValueError('Control A unexpectedly called LLM; comparison invalid')
    comparisons = []
    metric = protocol['metric'];sign = 1 if protocol['direction']=='minimize' else -1
    for a,b in zip(reports['A']['checkpoints'],reports['B']['checkpoints']):
        item = {'seconds':a['seconds'],'A':a.get('audit_metrics',{}).get(metric),'B':b.get('audit_metrics',{}).get(metric)}
        item['improvement_B_over_A'] = sign*(item['A']-item['B']) if item['A'] is not None and item['B'] is not None else None
        comparisons.append(item)
    result = {'protocol':protocol,'modes':reports,'comparison':comparisons}
    write_json(root/'report.json',result)
    lines = ['# Local A/B benchmark','',f'Metric: {metric}. Positive gain means B is better.',
        '', '| Seconds | A audit | B audit | B gain |','|---:|---:|---:|---:|']
    for row in comparisons:
        fmt = lambda value:'unavailable' if value is None else f'{value:.6g}'
        lines.append(f"| {row['seconds']} | {fmt(row['A'])} | {fmt(row['B'])} | {fmt(row['improvement_B_over_A'])} |")
    for mode in ['A','B']:
        r = reports[mode]
        lines += ['',f"{mode}: exit={r['exit']['exit_code']}, state={r['state']}, final={r['final_audit_metrics']}",
            f"Errors logged: {len(r['error_events'])}. API attempts: {r['event_counts'].get('llm.call_started',0)}."]
        if not r['event_log_has_finish']:
            lines += ['Event log lacks run.finished: event counts and timings may be incomplete; state comes from SQLite.']
    lines += ['', 'Final output may differ from the last checkpoint because final confirmation/refit takes time.',
        'A missing CSV is a failed run, not a zero metric. Check exit/state/error_events before interpreting gains.',
        'Audit scores were computed only after both modes stopped. No normalized leaderboard score is inferred.',
        '30-minute configuration includes its own finalization reserve; this is not a prefix of a three-hour run.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    run = sub.add_parser('run')
    run.add_argument('--task',type=Path)
    run.add_argument('--target')
    run.add_argument('--id-column')
    run.add_argument('--kind',choices=['regression','classification'],required=True)
    run.add_argument('--metric',choices=['rmse','mae','r2','f1_macro','accuracy'])
    run.add_argument('--out',type=Path,required=True)
    run.add_argument('--env-file',type=Path)
    run.add_argument('--seconds',type=int,default=1800)
    run.add_argument('--checkpoints',default='300,900,1800')
    run.add_argument('--audit-fraction',type=float,default=.2)
    run.add_argument('--seed',type=int,default=42)
    run.add_argument('--order',choices=['A,B','B,A'],default='A,B')
    run.add_argument('--mock',action='store_true',help='Synthetic transport only; no quality inference')
    run.add_argument('--demo',action='store_true',help='Generate an unseen synthetic task for a smoke check')
    report = sub.add_parser('report');report.add_argument('directory',type=Path)
    probe = sub.add_parser('probe');probe.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(argv)
    if os.name != 'posix':parser.error('Use Linux/WSL2: process-group termination is required')
    if args.command == 'report':summarize(args.directory.resolve());return 0
    if args.command == 'probe':
        from scripts.make_demo import make_demo
        root = args.out.resolve();root.mkdir(parents=True,exist_ok=False)
        task = make_demo(root/'task','regression')
        p = subprocess.run([sys.executable,str(ROOT/'scripts/ab_worker.py'),'--probe-task',str(task),
            '--probe-output',str(root/'result')],cwd=ROOT,timeout=45)
        print('Probe result: '+str(root/'result/probe.json'),flush=True)
        return p.returncode
    if not 5 <= args.seconds <= 10800:parser.error('--seconds must be 5..10800')
    if not .05 <= args.audit_fraction <= .4:parser.error('--audit-fraction must be .05.. .4')
    args.checkpoints = sorted(set(int(x) for x in args.checkpoints.split(',')))
    if not args.checkpoints or args.checkpoints[0] <= 0 or args.checkpoints[-1] > args.seconds:
        parser.error('Checkpoints must be positive seconds within the run budget')
    if args.demo and not args.mock:parser.error('--demo requires --mock; live mode requires a real task')
    root, protocol, controlled = prepare(args)
    env = child_env(args,controlled)
    if not args.mock and not env.get('OPENAI_API_KEY'):
        parser.error('Set platform credentials or pass --env-file; no implicit .env loading')
    for mode in args.order.split(','):
        print(f'Starting {mode}; budget={args.seconds}s; log={root/mode/"terminal.log"}',flush=True)
        result = execute(root,mode,args,env)
        print(f'{mode} finished: {result}',flush=True)
    result = summarize(root)
    print('Report: '+str(root/'report.md'),flush=True)
    return 0 if all(r['state']=='completed' and r['exit']['exit_code']==0 for r in result['modes'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
