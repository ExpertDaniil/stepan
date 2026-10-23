"""Execute fit-only analysis -> input feature -> fit/evaluation -> checked CSV.

Only LLM replies are scripted. Models, workers, validation and publication are real.
This is an execution demonstration, not a benchmark of live LLM intelligence.
"""
import argparse
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.llm.gateway import MockTransport
from agent.runtime import main as run_agent
from scripts.analyze_run import inspect_run
from scripts.make_demo import make_demo


def demonstrate(directory, kind):
    import numpy as np
    import pandas as pd
    task = make_demo(directory/'task', kind, rows=480, test_rows=48)
    train = pd.read_csv(task.parent/'train.csv', dtype={'id': str})
    # Synthetic interaction is deliberately orthogonal to the additive baseline.
    latent = 5*train.x_numeric.fillna(0)*train.z_numeric
    train['label'] = (latent if kind == 'regression' else
                      np.where(latent < -4, 'A', np.where(latent > 4, 'C', 'B')))
    train.to_csv(task.parent/'train.csv', index=False)
    original = MockTransport.complete
    choices, feature_sent = 0, False

    def complete(transport, request, timeout, context):
        nonlocal choices, feature_sent
        if context['metadata']['schema_id'] != 'ResearchAction-v1':
            return original(transport, request, timeout, context)
        choices += 1
        if choices == 1 and request.get('tools'):
            message = {'role': 'assistant', 'tool_calls': [{'id': 'fit-evidence', 'type': 'function',
                'function': {'name': 'analyze_data', 'arguments': json.dumps({
                    'kind': 'association', 'columns': ['x_numeric', 'z_numeric'],
                    'group_by': None, 'include_target': True})}}]}
            return {'choices': [{'message': message, 'finish_reason': 'tool_calls'}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 100}}
        evidence = []
        for message in request['messages']:
            if message.get('role') == 'tool':
                reply = json.loads(message['content'])
                if reply.get('artifact_id'):
                    evidence.append(reply['artifact_id'])
        recipe = 'linear-regressor-v1' if kind == 'regression' else 'linear-classifier-v1'
        value = {'action': 'stop' if feature_sent else 'features', 'recipe_id': recipe,
            'hyperparameters': {} if feature_sent else {'alpha': .01} if kind == 'regression' else {'C': 100.0},
            'hypothesis': 'Synthetic demonstration completed.' if feature_sent else 'Test a product representation of the two observed numeric inputs.',
            'expected_effect': 'Measure the complete candidate on the frozen selection split; no claimed score.',
            'feature_steps': [], 'code': None, 'tuning': None, 'evidence_ids': evidence}
        if not feature_sent:
            value['feature_program'] = [{'op': 'product', 'name': 'joint_numeric_signal',
                                        'columns': ['x_numeric', 'z_numeric']}]
            feature_sent = True
        return {'choices': [{'message': {'role': 'assistant', 'content': json.dumps(value)},
                             'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 100}}

    settings = {'LLM_MODE': 'mock', 'RESEARCH_MODE': 'adaptive', 'TIME_BUDGET_SECONDS': '120',
        'GRAPH_ENABLED': '1', 'GRAPH_PEERS': '1', 'GRAPH_TRAIN_WORKERS': '1', 'MAX_WORKERS': '1',
        'TRAIN_CPU_THREADS': '1', 'CATALOG_WARMUP_EXPERIMENTS': '1', 'GRAPH_REFINEMENT_ENABLED': '0',
        'FEATURE_RESEARCH_ENABLED': '1', 'FEATURE_SEED_BUDGET_FRACTION': '0',
        'LLM_MAX_CALLS': '12', 'LLM_MAX_TOTAL_OUTPUT_TOKENS': '262144', 'LOG_LEVEL': 'ERROR',
        'MODEL_SNAPSHOT_MAX_BYTES': '0', 'GRAPH_MIN_FREE_DISK_BYTES': '0'}
    logs = io.StringIO()
    with patch.dict(os.environ, settings), patch.object(MockTransport, 'complete', complete), redirect_stderr(logs):
        status = run_agent([str(task), '--llm-mode', 'mock', '--work-dir', str(directory/'runs')])
    (directory/'execution.log').write_text(logs.getvalue(), encoding='utf-8')
    if status:
        raise RuntimeError(logs.getvalue()[-6000:])
    run = next((directory/'runs').glob('run-*'))
    report = inspect_run(run)[0]
    output = pd.read_csv(task.parent/'submission.csv', dtype={'id': str})
    if len(output) != 48 or output['label'].isna().any():
        raise AssertionError('Prediction output contract failed')
    if report['research_summary']['evaluated_owned_feature_programs'] != 1:
        raise AssertionError('The proposed input representation was not trained/evaluated')
    if report['research_summary']['completed_data_analyses'] < 1:
        raise AssertionError('No fit-only analysis was executed')
    selected = report['published_pipeline']['experiment_id']
    result = {'kind': kind, 'state': report['state'], 'selected_experiment_id': selected,
              'output_rows': len(output), 'research_summary': report['research_summary'],
              'job_occupancy': report['job_occupancy'],
              'interpretation': 'Scripted LLM choices; real training/evaluation/output. Not a live-LLM or competition score.'}
    (directory/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--kind', choices=['regression', 'classification', 'both'], default='both')
    args = parser.parse_args()
    for task_kind in ['regression', 'classification'] if args.kind == 'both' else [args.kind]:
        print(json.dumps(demonstrate(args.directory.resolve()/task_kind, task_kind), indent=2), flush=True)
