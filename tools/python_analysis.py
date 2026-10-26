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
    'include_target': {'type': 'boolean'}, 'compare_test': {'type': 'boolean'},
    'experiment_id': {'type': 'string', 'minLength': 1,
        'description': 'Optional best or visible experiment ID. context["oof"] contains aligned DataFrames predictions/errors, Series fold_ids, list row_ids; regression also residuals/squared_errors. y is a Series for one target only when include_target=true; X_test is None unless compare_test=true.'}},
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
    return tree


def execute_analysis(service, code, inputs, timeout):
    """Return a retained report even for import, process, memory or time failures."""
    from agent.config import ROOT
    from tools.analysis_worker import failure_report
    try:
        with tempfile.TemporaryDirectory(prefix='stepan-analysis-') as directory:
            directory = Path(directory)
            source = directory/'analysis.py';source.write_text(code)
            data_path = directory/'input.pkl'
            with data_path.open('wb') as stream:pickle.dump(inputs, stream)
            output = directory/'result.json'
            env = {k:v for k,v in os.environ.items() if k in {'PATH','SYSTEMROOT','LANG','LC_ALL'}}
            env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            process = subprocess.Popen([sys.executable, '-m', 'tools.analysis_worker', str(source),
                str(data_path), str(output)], cwd=directory, env=env, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True)
            with service.process_lock:service.active_process = process
            try:
                end = time.monotonic()+timeout
                while process.poll() is None:
                    if service.cancelled.is_set():
                        raise AgentError('RESEARCH_CANCELLED', 'Python analysis cancelled for publication')
                    if time.monotonic() >= end:
                        raise AgentError('ANALYSIS_TIMEOUT', 'Python analysis reached its execution budget')
                    try:process.wait(timeout=min(.1, max(.001, end-time.monotonic())))
                    except subprocess.TimeoutExpired:pass
            finally:
                try:os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
                with service.process_lock:service.active_process = None
            if not output.exists():
                code = 'ANALYSIS_MEMORY_LIMIT' if process.returncode == 137 else 'ANALYSIS_EXECUTION_FAILED'
                raise AgentError(code, 'Analysis process exited without a report: '+str(process.returncode))
            return json.loads(output.read_text())
    except AgentError as error:
        if error.code == 'RESEARCH_CANCELLED':raise
        return failure_report(error)
    except (OSError, ValueError) as error:
        wrapped = AgentError('ANALYSIS_EXECUTION_FAILED', str(error)[:600])
        return failure_report(wrapped)


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
    oof_source = None
    if query.get('experiment_id'):
        from agent.context_details import read_solution
        resolved = read_solution({'experiment_id': query['experiment_id'], 'view': 'result'}, context, service.artifacts, service.log)
        key['experiment_id'] = resolved['experiment_id']
        oof_source = deepcopy(getattr(service, 'oof_sources', {}).get(resolved['experiment_id']))
        if oof_source is None:
            raise AgentError('OOF_PENDING', 'Fit-only OOF is queued, unavailable or not yet completed for this exact version. Continue other analysis; old reports retain their experiment IDs.')
        key['oof_identity'] = oof_source['identity_sha256']
    digest = hashlib.sha256(canonical([service.validation_id, key]).encode()).hexdigest()
    with service.query_slot(context), service.budget.analysis_slot() if hasattr(service.budget, 'analysis_slot') else nullcontext():
        service.query_context = context
        if digest in service.cache:
            response = deepcopy(service.cache[digest])
            # Retain visibility for this work order without claiming a new computation.
            owner = (context or {}).get('metadata', {}).get('graph_binding', {}).get('work_order_id')
            service.records.append({'artifact_id': response['artifact_id'], 'query': {**key, 'kind': 'python'},
                                    'completed_elapsed_ms': int((time.monotonic()-getattr(service.budget, 'started', time.monotonic()))*1000),
                                    'scope': 'fit_only', 'work_order_id': owner})
            response['reused'] = True
            return response
        started = time.monotonic()
        service._check_time(started)
        selected = columns + (targets(service.spec) if query.get('include_target') else [])
        oof = None
        if oof_source:
            from agent.artifacts import file_sha256
            from agent.data_transport import transport
            from agent.predictions import validation_columns
            from agent.fit_diagnostics import error_values
            import pandas as pd
            import numpy as np
            identity = oof_source['identity']
            if (identity['validation_id'] != service.validation_id
                    or any(identity[k] != service.spec[k] for k in ('task_spec_id', 'dataset_id'))
                    or identity['train_sha256'] != service.artifacts.meta(service.spec['train_artifact_id'])['sha256']
                    or not set(identity['predicted_row_ids']) <= service.fit_indices):
                raise AgentError('MEMORY_SCOPE_DENIED', 'OOF identity is outside this fit partition')
            if file_sha256(service.artifacts.resolve(oof_source['identity_artifact_id'])) != oof_source['identity_sha256']:
                raise AgentError('MEMORY_SOURCE_CHANGED', 'OOF provenance changed')
            path = service.artifacts.resolve(identity['predictions_artifact_id'])
            if file_sha256(path) != identity['predictions_sha256']:
                raise AgentError('MEMORY_SOURCE_CHANGED', 'OOF predictions changed')
            names = validation_columns(service.spec)
            raw = pd.read_csv(path, dtype={n:str for n in names} if service.spec['metric']['prediction_kind'] == 'labels' else None, keep_default_na=False)
            rows = identity['predicted_row_ids']
            if raw['__row_index'].tolist() != rows or len(set(rows)) != len(rows):
                raise AgentError('MEMORY_SOURCE_CHANGED', 'OOF row identity changed')
            frame = transport(service.artifacts).read(service.spec['train_artifact_id'], service.spec,
                columns=list(dict.fromkeys([*selected, *targets(service.spec)])), rows=rows)
            population = service.fit_rows
            names_y = targets(service.spec)
            truth = frame[names_y if len(names_y)>1 else names_y[0]]
            values = raw[names].to_numpy()
            yy, pp, errors, squared = error_values(truth, values, service.spec)
            oof = {'experiment_id': identity['experiment_id'], 'row_ids': rows,
                'predictions': pd.DataFrame(values, index=frame.index, columns=names),
                'errors': pd.DataFrame(errors, index=frame.index),
                'fold_ids': pd.Series(raw['__fold'].to_numpy(), index=frame.index),
                'identity_artifact_id': oof_source['identity_artifact_id'], 'scope': 'fit_only_inner_oof'}
            if squared is not None:
                oof.update(squared_errors=pd.DataFrame(squared, index=frame.index), residuals=pd.DataFrame(pp-yy, index=frame.index))
        else:
            frame, population, _ = service._sample('fit', selected, None, started)
        y = frame[targets(service.spec)].copy() if query.get('include_target') else None
        if y is not None and len(y.columns) == 1:y = y.iloc[:,0]
        test, test_population = None, None
        if query.get('compare_test'):
            test, test_population, _ = service._sample('test', columns, None, started)
        timeout = min(60, service.budget.phase_remaining()-service.budget.reserve_seconds()-1)
        if (context or {}).get('_tool_deadline') is not None:
            timeout = min(timeout, context['_tool_deadline']-time.monotonic()-.1)
        if timeout < 1:raise AgentError('SEARCH_CLOSED', 'No analysis time before publication reserve')
        result = execute_analysis(service, query['code'], (frame[columns].copy(), y, test,
            {'scope': 'fit_only', 'oof': oof, 'target_columns': targets(service.spec),
             'population_rows': population, 'analyzed_rows': len(frame),
             'sampled': len(frame) < population, 'test_population_rows': test_population}), timeout)
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
            'code_artifact_id': code_art['artifact_id'],
            'completed_elapsed_ms': int((time.monotonic()-getattr(service.budget, 'started', time.monotonic()))*1000),
            'scope': 'fit_only', 'work_order_id': (context or {}).get('metadata', {}).get('graph_binding', {}).get('work_order_id')})
        service.log.event('research.python_analysis_completed', artifact_id=art['artifact_id'],
            code_artifact_id=code_art['artifact_id'], success=result['success'], rows=len(frame),
            duration_seconds=result['duration_seconds'],
            **{k:result[k] for k in ('error_code', 'error_type', 'error_message', 'failure_scope', 'guard_origin', 'repair_hint') if k in result})
        return response
