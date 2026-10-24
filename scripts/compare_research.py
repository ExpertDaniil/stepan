"""Run the 2x2 research comparison with a fixed task/seed and equal wall budgets.

Uses the configured live provider by default. Never embeds keys or submits to LB.
All model evaluations are performed by the existing independent evaluator.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib
import zipfile

ROOT=Path(__file__).resolve().parents[1]
MODES={'A':('legacy','0'),'B':('compact','0'),'C':('legacy','1'),'D':('compact','1')}


def summarize(directory, mode, elapsed, exit_code):
    runs=sorted(directory.glob('runs/run-*'))
    summary={}
    events=[]
    if runs:
        if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
        from scripts.analyze_run import inspect_run
        try:
            summary=inspect_run(runs[-1])[0]
        except (OSError, ValueError) as error:
            summary={'state':'inspection_failed','error':str(error)}
        path=runs[-1]/'events.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:events.append(json.loads(line))
                except ValueError:pass
    # A completed immutable summary outranks a partial streaming events file.
    if runs and (runs[-1]/'summary.md').is_file():
        text=(runs[-1]/'summary.md').read_text()
        fields=dict(line.split(': ',1) for line in text.splitlines() if ': ' in line and not line.startswith('|'))
        if fields.get('State'):
            if summary.get('state') in {'incomplete','failed','inspection_failed',None}:
                summary['telemetry_warning']='Streaming event log is incomplete; state/metric recovered from final summary.'
            summary['state']=fields['State']
            summary['published_metric']=fields.get('Published metric')
            summary['metric_id']=fields.get('Metric')
            summary['published_experiment_id']=fields.get('Published experiment')
    if runs:
        import sqlite3
        database=runs[-1]/'state.sqlite'
        if database.exists():
            with sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True) as db:
                summary['experiment_states']={state:count for state,count in db.execute('SELECT state,count(*) FROM experiments GROUP BY state')}
    counts={}
    for event in events:
        name=event.get('event',event.get('name','unknown'))
        counts[name]=counts.get(name,0)+1
    return {'mode':mode,'exit_code':exit_code,'elapsed_seconds':elapsed,
            'summary':summary,'event_counts':counts,
            'scope':'local_fixed_validation; no leaderboard claim'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--seconds',type=int,default=600)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--modes',default='ABCD')
    parser.add_argument('--env-file',type=Path)
    args=parser.parse_args()
    if not 60 <= args.seconds <= 10800 or args.repeats < 1 or not set(args.modes)<=set(MODES):parser.error('Invalid budget/repeats/modes')
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    from scripts.package_solution import build
    package=output/'comparison-solution.zip'
    metadata=build(package)
    frozen=output/'platform';frozen.mkdir()
    with zipfile.ZipFile(package) as archive:archive.extractall(frozen)
    results=[]
    # Rotate ordering between repeats to reduce provider/time-of-day ordering effects.
    modes=list(dict.fromkeys(args.modes))
    for repeat in range(args.repeats):
        order=modes[repeat%len(modes):]+modes[:repeat%len(modes)]
        for mode in order:
            directory=output/(str(repeat+1)+'-'+mode);directory.mkdir()
            protocol,python=MODES[mode]
            env=dict(os.environ, RESEARCH_MODE='adaptive',RESEARCH_PROTOCOL=protocol,
                     PYTHON_ANALYSIS_ENABLED=python,DATA_RESEARCH_MIN_SECONDS='0',
                     TIME_BUDGET_SECONDS=str(args.seconds),GRAPH_PEERS='1',MAX_WORKERS='1',
                     WORK_DIR=str(directory/'runs'),SEED='42')
            command=[sys.executable,str(frozen/'solution.py'),str(args.task.resolve())]
            if args.env_file:command+=['--env-file',str(args.env_file.resolve())]
            started=time.monotonic()
            with (directory/'terminal.log').open('w') as stream:
                proc=subprocess.Popen(command,cwd=directory,env=env,stdout=stream,stderr=subprocess.STDOUT)
                try:code=proc.wait(timeout=args.seconds+60)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    code=124
            result=summarize(directory,mode,time.monotonic()-started,code)
            results.append({'repeat':repeat+1,'zip_sha256':metadata['sha256'],**result})
            (output/'comparison.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
            print(json.dumps({'repeat':repeat+1,'mode':mode,'exit_code':code}),flush=True)
    return int(any(r['exit_code'] for r in results))


if __name__=='__main__':raise SystemExit(main())
