"""Run a real synthetic fit with a scripted, offline researcher and extra evidence."""
from pathlib import Path
import argparse
import json
import os
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agent.llm.gateway import MockTransport
from agent.runtime import main
from scripts.make_demo import make_demo

SOURCE = '''from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
def make_pipeline(task_spec, params):
    numeric = Pipeline([("missing", SimpleImputer()), ("scale", StandardScaler())])
    return Pipeline([("features", ColumnTransformer([("numeric", numeric, ["x_numeric", "z_numeric"])])),
                     ("model", Ridge(alpha=params["alpha"]))])
'''


def run_demo(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError('Choose an empty directory for the demonstration')
    task = make_demo(directory / 'task', 'regression', rows=120, test_rows=30)
    original = MockTransport.complete
    proposed = set()

    def response(transport, request, timeout, context):
        schema = context['metadata']['schema_id']
        has_tool = request['messages'][-1]['role'] == 'tool'
        tool = None
        if schema == 'TaskDraft-v1' and not has_tool:
            tool = ('inspect_schema', {'table': 'train', 'offset': 0, 'limit': 8})
        elif schema == 'ResearchAction-v1':
            peer = context['metadata']['researcher_id']
            if peer not in proposed and not has_tool:
                tool = ('analyze_data', {'kind': 'summary', 'columns': ['x_numeric', 'z_numeric'],
                                        'group_by': None, 'include_target': False})
            else:
                stop = peer in proposed
                proposed.add(peer)
                evidence = [] if stop else [json.loads(request['messages'][-1]['content'])['artifact_id']]
                result = dict(action='stop' if stop else 'pipeline', recipe_id='python-pipeline-v1',
                    hyperparameters={} if stop else {'alpha': 1.0}, hypothesis='Scripted demonstration completed' if stop else 'Use the inspected numeric inputs with imputation and scaling',
                    expected_effect='Measure on the unchanged split', feature_steps=[],
                    code=None if stop else SOURCE, tuning=None, evidence_ids=evidence)
                return {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': json.dumps(result)}}]}
        if tool:
            name, args = tool
            return {'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 'demo-facts', 'type': 'function',
                                'function': {'name': name, 'arguments': json.dumps(args)}}]}}]}
        return original(transport, request, timeout, context)

    settings = {'LLM_MODE': 'mock', 'RESEARCH_MODE': 'adaptive', 'GRAPH_ENABLED': '1',
                'TIME_BUDGET_SECONDS': '90', 'MAX_WORKERS': '1', 'GRAPH_PEERS': '1', 'GRAPH_TRAIN_WORKERS': '1',
                'CATALOG_WARMUP_EXPERIMENTS': '0', 'GRAPH_MIN_FREE_DISK_BYTES': '0', 'LOG_LEVEL': 'ERROR'}
    with patch.dict(os.environ, settings), patch.object(MockTransport, 'complete', response):
        code = main([str(task), '--work-dir', str(directory / 'runs')])
    if code:
        raise RuntimeError('Demo failed; inspect the saved run')
    run = next((directory / 'runs').glob('run-*'))
    events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
    result = {'run_id': run.name, 'state': events[-1]['state'], 'metric': events[-1]['metric_id'],
              'score': events[-1]['score'], 'analysis_reports': len(list((run / 'analysis').glob('*.json'))),
              'publications': [{k: e[k] for k in ('experiment_id', 'score')} for e in events if e['event'] == 'submission.published'],
              'output': str(task.parent / 'submission.csv')}
    (directory / 'demo-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    print(json.dumps(run_demo(parser.parse_args().directory), ensure_ascii=False, indent=2))
