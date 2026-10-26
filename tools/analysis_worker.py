"""Bounded Python analysis worker. Invoked only by python_analysis service."""
import contextlib
import io
import json
import os
from pathlib import Path
import pickle
import sys
import threading
import traceback

from agent.errors import AgentError


def failure_report(error):
    """Keep a repairable failure separate from a measured ML outcome."""
    from tools.runtime_guard import GuardViolation
    origin = getattr(error, 'origin', None)
    if isinstance(error, GuardViolation):
        scope = 'execution_environment' if origin in {'owned_runtime', 'installed_library'} else 'analysis_code'
        code = 'GUARD_ENVIRONMENT_BLOCKED' if scope == 'execution_environment' else 'GUARD_POLICY_BLOCKED'
        hint = ('The analysis was not measured. The runtime must prepare the owned dependency.' if scope == 'execution_environment'
                else 'Use only the supplied X/y/T/context and allowed libraries; file, process and network access stay blocked.')
    elif isinstance(error, AgentError):
        code = error.code
        scope = 'resource_limit' if code in {'ANALYSIS_TIMEOUT', 'ANALYSIS_MEMORY_LIMIT'} else 'execution_environment'
        hint = 'Use a smaller fit sample or a cheaper computation.' if scope == 'resource_limit' else 'Inspect the retained report before retrying.'
    else:
        code, scope = 'ANALYSIS_CODE_FAILED', 'analysis_code'
        hint = ('Check X.columns and the supplied context["oof"] field names and shapes; repair the saved code.'
                if isinstance(error, (KeyError, IndexError, AttributeError)) else
                'Check input dtypes, array shapes and the failing line; repair the saved code and return compact aggregates.')
    return {'success': False, 'error': traceback.format_exc(limit=5)[-3000:], 'error_code': code,
            'error_type': type(error).__name__, 'error_message': str(error)[:600],
            'failure_scope': scope, 'guard_origin': origin, 'repair_hint': hint}


def clean_result(value, depth=0):
    import numpy as np
    import pandas as pd
    if depth > 5:return '[nested result omitted]'
    if value is pd.NA or value is pd.NaT:return None
    if isinstance(value,pd.DataFrame):return clean_result(value.head(20).reset_index().to_dict('records'),depth+1)
    if isinstance(value,pd.Series):return clean_result(value.head(40).to_dict(),depth+1)
    if isinstance(value,np.ndarray):return clean_result(value[:40].tolist(),depth+1)
    if isinstance(value,np.generic):return clean_result(value.item(),depth)
    if isinstance(value,dict):return {str(k)[:120]:clean_result(v,depth+1) for k,v in list(value.items())[:40]}
    if isinstance(value,(list,tuple)):return [clean_result(v,depth+1) for v in value[:40]]
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,str):return value[:2000]
    if value is None or isinstance(value,(int,float,bool)):return value
    return str(value)[:200]


class BoundedText(io.StringIO):
    def write(self, text):
        left = max(0, 4000-self.tell())
        super().write(text[:left])
        return len(text)


def main():
    source, data, output = map(Path, sys.argv[1:])
    code = source.read_text()
    from tools.python_analysis import validate_analysis
    from tools.runtime_guard import PipelineGuard
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from joblib import parallel_backend
    with data.open('rb') as stream:X, y, test, metadata = pickle.load(stream)
    # The watchdog lives in trusted code and uses peak RSS, independent of PID namespace.
    done = threading.Event()
    def watch():
        from agent.worker import self_peak_mb
        while not done.wait(.1):
            if self_peak_mb() > 1024:
                os._exit(137)
    watchdog = threading.Thread(target=watch, daemon=True)
    watchdog.start()
    guard = PipelineGuard(source, trusted_thread=watchdog)
    log = BoundedText()
    result = {}
    try:
        validate_analysis(code)
        # Use the same owned-backend/native preparation as generated pipelines.
        from tools.runtime_guard import prepare_pipeline_imports
        prepare_pipeline_imports(source, analysis=True)
        with threadpool_limits(limits=1), parallel_backend('threading',n_jobs=1), contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            guard.active = True
            namespace = {'__name__': 'generated_analysis'}
            exec(compile(code,str(source),'exec'),namespace)
            result = {'success': True, 'result': clean_result(namespace['analyze'](X,y,test,metadata))}
            encoded = json.dumps(result,ensure_ascii=False,allow_nan=False)
            if len(encoded.encode()) > 12000:
                raise ValueError('ANALYSIS_OUTPUT_TOO_LARGE: return a smaller report')
    except BaseException as error:
        result = failure_report(error)
    finally:
        guard.active = False
        done.set()
    result['stdout'] = log.getvalue()
    output.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__ == '__main__':main()
