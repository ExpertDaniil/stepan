"""Fit-only Python exploration. Separate bounded process; no SQL/API credentials.

Python AST/audit guards are defence in depth, not a hostile-code OS sandbox.
"""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import tempfile
import time

from agent.contracts import canonical
from agent.errors import AgentError

PYTHON_ANALYSIS_SCHEMA = {'type': 'object', 'properties': {
    'question': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
    'code': {'type': 'string', 'minLength': 1, 'maxLength': 24000},
    'columns': {'type': 'array', 'minItems': 1, 'maxItems': 40,
                'items': {'type': 'string', 'minLength': 1}},
    'include_target': {'type': 'boolean'}, 'compare_test': {'type': 'boolean'}},
    'required': ['question', 'code', 'columns'], 'additionalProperties': False}


def validate_analysis(code):
    import ast
    from tools.code import validate_code
    tree = ast.parse(code)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'analyze']
    if len(functions) != 1 or len(functions[0].args.args) != 4:
        raise AgentError('INVALID_ANALYSIS_CODE', 'Export analyze(X_fit, y_fit, X_test, context)')
    # Reuse the pipeline's import/IO policy, without its estimator entrypoint.
    validate_code(code+'\n\ndef make_pipeline(task_spec, params):\n    return None\n')


def run_analysis(service, query, context=None):
    from agent.predictions import targets
    from agent.config import ROOT
    service.schemas._check(query, PYTHON_ANALYSIS_SCHEMA, '$')
    try:
        validate_analysis(query['code'])
    except (ValueError, SyntaxError) as exc:
        raise AgentError('INVALID_ANALYSIS_CODE', str(exc)[:600]) from exc
    columns = list(dict.fromkeys(query['columns']))
    excluded = set(targets(service.spec)) | set(service.spec['excluded_features'])
    if any(c not in service.train_columns or c in excluded for c in columns):
        raise AgentError('INVALID_ANALYSIS', 'columns must be available non-target, non-excluded inputs')
    if query.get('compare_test') and not set(columns) <= set(service.test_columns):
        raise AgentError('INVALID_ANALYSIS', 'Test comparison requires columns present in test')
    # The question is prose; identical computations across dialogues are cached.
    key = {k:v for k,v in query.items() if k != 'question'}
    digest = hashlib.sha256(canonical([service.validation_id, key]).encode()).hexdigest()
    with service.lock, service.budget.analysis_slot() if hasattr(service.budget, 'analysis_slot') else nullcontext():
        service.query_context = context
        if digest in service.cache:
            response = deepcopy(service.cache[digest])
            # Retain visibility for this work order without claiming a new computation.
            owner = (context or {}).get('metadata', {}).get('graph_binding', {}).get('work_order_id')
            service.records.append({'artifact_id': response['artifact_id'], 'query': {**key, 'kind': 'python'},
                                    'scope': 'fit_only', 'work_order_id': owner})
            response['reused'] = True
            return response
        started = time.monotonic()
        selected = columns + (targets(service.spec) if query.get('include_target') else [])
        frame, population, _ = service._sample('fit', selected, None, started)
        y = frame[targets(service.spec)].copy() if query.get('include_target') else None
        if y is not None and len(y.columns) == 1:y = y.iloc[:,0]
        test, test_population = None, None
        if query.get('compare_test'):
            test, test_population, _ = service._sample('test', columns, None, started)
        timeout = min(60, service.budget.phase_remaining()-service.budget.reserve_seconds()-1)
        if timeout < 1:raise AgentError('SEARCH_CLOSED', 'No analysis time before publication reserve')
        with tempfile.TemporaryDirectory(prefix='stepan-analysis-') as directory:
            directory = Path(directory)
            source = directory/'analysis.py';source.write_text(query['code'])
            data_path = directory/'input.pkl'
            with data_path.open('wb') as stream:
                pickle.dump((frame[columns].copy(), y, test, {'scope': 'fit_only',
                    'population_rows': population, 'analyzed_rows': len(frame),
                    'sampled': len(frame) < population, 'test_population_rows': test_population}), stream)
            output = directory/'result.json'
            env = {k:v for k,v in os.environ.items() if k in {'PATH','SYSTEMROOT','LANG','LC_ALL'}}
            env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            process = subprocess.Popen([sys.executable, '-m', 'tools.analysis_worker', str(source),
                str(data_path), str(output)], cwd=directory, env=env, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise AgentError('ANALYSIS_TIMEOUT', 'Python analysis reached its execution budget') from exc
            finally:
                try:os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
            if not output.exists():
                raise AgentError('ANALYSIS_EXECUTION_FAILED', 'Analysis process exited without a report: '+str(process.returncode))
            result = json.loads(output.read_text())
        result.update(scope='fit_only', validation_id=service.validation_id, query={**query,'kind':'python'},
                      fit_rows=service.fit_rows, matched_fit_rows=population, analyzed_rows=len(frame),
                      sampled=population > len(frame), duration_seconds=time.monotonic()-started)
        # Always retain failed executions as readable evidence; caller can repair code.
        art = service.artifacts.write_json('analysis/python-'+digest+'.json', result, 'research_analysis')
        code_art = service.artifacts.write_text('analysis/code-'+digest+'.py', query['code'], 'analysis_source', 'text/x-python')
        response = {'artifact_id': art['artifact_id'], 'code_artifact_id': code_art['artifact_id'],
                    'report': {k:v for k,v in result.items() if k != 'query'}}
        service.cache[digest] = deepcopy(response)
        service.records.append({'artifact_id': art['artifact_id'], 'query': {**key,'kind':'python'},
            'scope': 'fit_only', 'work_order_id': (context or {}).get('metadata', {}).get('graph_binding', {}).get('work_order_id')})
        service.log.event('research.python_analysis_completed', artifact_id=art['artifact_id'],
            code_artifact_id=code_art['artifact_id'], success=result['success'], rows=len(frame),
            duration_seconds=result['duration_seconds'])
        return response
