"""Benchmark-only child adapters. Never imported by solution.py or packaged.

Contract injection isolates research from bootstrap variability. The outer
benchmark evaluator is the only process that opens the audit labels.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(case, mode):
    from agent.runtime import main
    from agent.task_interpreter import TaskInterpreter
    from agent.submission import Submission
    from agent.researcher import Researcher
    from agent.errors import AgentError
    from agent.semantic_planning import unknown_model
    contract = json.loads((case / 'contract.json').read_text())
    publications = case / 'publications'
    publications.mkdir()
    started = time.monotonic()
    publish = Submission.publish

    def record(self, payload):
        result = publish(self, payload)
        if result['disposition'] == 'published':
            ready = time.monotonic()-started
            champion = result['champion']
            name = f"{champion['champion_version']:06d}.csv"
            shutil.copyfile(result['output_path'], publications / name)
            row = {'ready_seconds':ready, 'csv':name,
                   'experiment_id':champion['experiment_id'],
                   'selection_metric':champion['primary_metric'],
                   'selection_basis':'confirmation' if payload.get('confirmation_report_id') else 'selection'}
            with (publications / 'index.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
        return result

    def interpret(self, payload):
        return {'draft':{**deepcopy(contract['draft']), 'task_model':unknown_model()},
                'evidence':['Explicit local benchmark contract; no inferred target.'], 'unresolved_fields':[]}

    def no_research(self, payload):
        # Only model/representation proposal work is disabled. Identical warmup
        # and VersionSearch remain enabled, with no artificial call-limit errors.
        return {'proposal_id':'ab-control-'+uuid.uuid4().hex[:12],
                'researcher_id':payload['researcher_id'], 'plan':None,
                'decision':'stop', 'stop_reason':'A/B control: catalogue and background search only.'}

    with patch.object(TaskInterpreter, 'interpret', interpret), \
         patch.object(TaskInterpreter, 'choose_validation', lambda *_:deepcopy(contract['validation'])), \
         patch.object(Submission, 'publish', record):
        if mode == 'A':
            # Any accidental gateway call in A is a test failure, not a paid call.
            from agent.llm.gateway import Gateway
            with patch.object(Researcher, 'propose', no_research), \
                 patch.object(Gateway, 'complete', side_effect=AgentError('AB_UNEXPECTED_LLM', 'Control attempted an LLM call')):
                return main([str(case/'task.md')])
        return main([str(case/'task.md')])


def probe(task, output):
    """Exercise the real startup failure path without accessing the provider."""
    import os
    from agent.runtime import main
    from agent.llm.gateway import OpenAITransport
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(OPENAI_API_KEY='synthetic-timeout-probe',
        OPENAI_BASE_URL='https://timeout-probe.invalid/v1', OPENAI_MODEL='timeout-probe',
        LLM_MODE='live', TIME_BUDGET_SECONDS='30', LLM_TIMEOUT_SECONDS='1',
        LLM_OPERATION_TIMEOUT_SECONDS='4', WORK_DIR=str(output/'runs'))
    with patch.object(OpenAITransport, 'complete', side_effect=TimeoutError('Injected startup timeout; no network request')):
        code = main([str(task)])
    events = [json.loads(line) for p in (output/'runs').glob('*/events.jsonl') for line in p.read_text().splitlines()]
    result = {'agent_exit_code':code, 'csv_created':(task.parent/'submission.csv').is_file(),
        'timeout_injected':True, 'network_calls':0,
        'bootstrap_failed':any(e['event']=='run.failed' for e in events)}
    result['protected_against_startup_timeout'] = result['csv_created'] and code == 0
    (output/'probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return 0 if result['protected_against_startup_timeout'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path)
    parser.add_argument('--mode', choices=['A','B'])
    parser.add_argument('--probe-task', type=Path)
    parser.add_argument('--probe-output', type=Path)
    args = parser.parse_args()
    if args.probe_task:
        raise SystemExit(probe(args.probe_task.resolve(), args.probe_output.resolve()))
    raise SystemExit(run(args.case.resolve(), args.mode))
