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


class BoundedText(io.StringIO):
    def write(self, text):
        left = max(0, 4000-self.tell())
        super().write(text[:left])
        return len(text)


def main():
    source, data, output = map(Path, sys.argv[1:])
    code = source.read_text()
    from tools.python_analysis import validate_analysis
    validate_analysis(code)
    from tools.runtime_guard import PipelineGuard
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from joblib import parallel_backend
    # Initialize declared libraries before closing IO, as in the pipeline worker.
    import ast, importlib
    from tools.runtime_guard import preload_native
    declared = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                importlib.import_module(alias.name)
                declared.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            importlib.import_module(node.module)
            declared.add(node.module)
    preload_native(declared)
    with data.open('rb') as stream:X, y, test, metadata = pickle.load(stream)
    # The watchdog lives in trusted code and uses peak RSS, independent of PID namespace.
    done = threading.Event()
    def watch():
        from agent.worker import self_peak_mb
        while not done.wait(.1):
            if self_peak_mb() > 1024:
                os._exit(137)
    threading.Thread(target=watch, daemon=True).start()
    guard = PipelineGuard(source)
    log = BoundedText()
    result = {}
    def clean(value, depth=0):
        if depth > 5:return '[nested result omitted]'
        if isinstance(value,pd.DataFrame):return clean(value.head(20).reset_index().to_dict('records'),depth+1)
        if isinstance(value,pd.Series):return clean(value.head(40).to_dict(),depth+1)
        if isinstance(value,np.ndarray):return clean(value[:40].tolist(),depth+1)
        if isinstance(value,np.generic):return clean(value.item(),depth+1)
        if isinstance(value,dict):return {str(k)[:120]:clean(v,depth+1) for k,v in list(value.items())[:40]}
        if isinstance(value,(list,tuple)):return [clean(v,depth+1) for v in value[:40]]
        if isinstance(value,float) and not np.isfinite(value):return None
        if isinstance(value,str):return value[:2000]
        if value is None or isinstance(value,(int,float,bool)):return value
        return str(value)[:200]
    try:
        with threadpool_limits(limits=1), parallel_backend('threading',n_jobs=1), contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            guard.active = True
            namespace = {'__name__': 'generated_analysis'}
            exec(compile(code,str(source),'exec'),namespace)
            result = {'success': True, 'result': clean(namespace['analyze'](X,y,test,metadata))}
            encoded = json.dumps(result,ensure_ascii=False,allow_nan=False)
            if len(encoded.encode()) > 12000:
                result = {'success':False,'error':'ANALYSIS_OUTPUT_TOO_LARGE: return a smaller report'}
    except BaseException:
        result = {'success':False,'error':traceback.format_exc(limit=5)[-3000:]}
    finally:
        guard.active = False
        done.set()
    result['stdout'] = log.getvalue()
    output.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__ == '__main__':main()
