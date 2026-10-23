"""One process per job. Receives a JSON job file, never an API key."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
import threading
import io
from contextlib import ExitStack


class SnapshotLimit(Exception):
    pass


class LimitedSnapshotFile(io.FileIO):
    def __init__(self, path, maximum):
        super().__init__(path, "wb")
        self.maximum = maximum

    def write(self, data):
        if self.tell()+len(data) > self.maximum:
            raise SnapshotLimit("Compressed model snapshot exceeds its byte allowance")
        return super().write(data)


def save_snapshot(model, path, maximum):
    """Optional diagnostics: predictions and reconstruction do not depend on pickle."""
    import joblib
    path = Path(path)
    temporary = path.with_suffix(".joblib.tmp")
    report = {"status": "disabled", "size_bytes": 0, "limit_bytes": maximum}
    if maximum <= 0:
        return report
    try:
        with LimitedSnapshotFile(temporary, maximum) as stream:
            joblib.dump(model, stream, compress=3)
        os.replace(temporary, path)
        report.update(status="saved", size_bytes=path.stat().st_size)
    except SnapshotLimit:
        report["status"] = "size_limit"
    except Exception as exc:
        report.update(status="unavailable", error_type=type(exc).__name__)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def self_peak_mb():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024
    except ImportError:
        return 0.0


def watch_job(job, done):
    """Worker-side peak-RSS guard also works with restricted parent procfs views."""
    started = time.monotonic()
    while not done.is_set():
        memory = self_peak_mb()
        code = "JOB_MEMORY_LIMIT" if memory > job["memory_limit_mb"] else "JOB_TIMEOUT" if time.monotonic() - started >= job["timeout_seconds"] else None
        if code:
            result = {"success": False, "error_type": code, "error": "Worker resource limit reached", "peak_memory_mb": memory}
            target = Path(job["result_path"])
            temp = target.with_suffix(".watchdog.tmp")
            temp.write_text(json.dumps(result), encoding="utf-8")
            os.replace(temp, target)
            print(code, file=sys.stderr, flush=True)
            os._exit(137 if code == "JOB_MEMORY_LIMIT" else 124)
        done.wait(0.1)


def execute(job):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from agent.ml import enforce_threads, validate_estimator
    from tools.data import read_table
    from agent.predictions import read_training, targets
    if job.get('code_sha256'):
        from agent.artifacts import file_sha256
        if file_sha256(job['code_path']) != job['code_sha256']:
            raise ValueError('Pipeline code changed between job admission and worker execution')
    task = job["task_spec"]
    train = read_training(job["train_path"], task)
    target_columns = targets(task)
    features = [c for c in train if c not in task["excluded_features"] and c not in target_columns]
    target = train[target_columns] if len(target_columns) > 1 else train[target_columns[0]]
    if task['task_kind'] == 'regression':
        target = target.apply(pd.to_numeric, errors='raise') if len(target_columns) > 1 else pd.to_numeric(target, errors='raise')
    else:
        target = target.astype(str)
    train_rows = len(train)
    if job['phase'] == 'train':
        fit_indices = json.loads(Path(job['train_indices_path']).read_text(encoding='utf-8'))
        valid_indices = json.loads(Path(job['valid_indices_path']).read_text(encoding='utf-8'))
        fit_x, fit_y = train.iloc[fit_indices][features].copy(), target.iloc[fit_indices].copy()
        predict_x = train.iloc[valid_indices][features].copy()
    else:
        test = read_table(job['test_path'])
        fit_x, fit_y, predict_x = train[features].copy(), target.copy(), test[features].copy()
        del test
    # Generated code receives no frame/Series backing storage containing validation labels.
    del train, target
    params = dict(job["hyperparameters"])
    params.update(_cpu_threads=job["cpu_threads"], _seed=job["seed"])
    module_name = "generated_" + job["job_id"].replace("-", "_")
    module_spec = importlib.util.spec_from_file_location(module_name, job["code_path"])
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    guard = None
    if job.get('guard_generated'):
        from tools.runtime_guard import PipelineGuard, prepare_pipeline_imports
        prepare_pipeline_imports(job['code_path'], job['hyperparameters'].get('base_recipe_id'))
        guard = PipelineGuard(job['code_path'])
    limits = ExitStack()
    limits.enter_context(threadpool_limits(limits=job['cpu_threads']))
    if job.get('guard_generated'):
        from joblib import parallel_backend
        limits.enter_context(parallel_backend('threading', n_jobs=job['cpu_threads']))
    if guard:
        guard.active = True
    try:
        return execute_pipeline(job, module, module_spec, params, task, train_rows, fit_x, fit_y, predict_x, guard, limits)
    finally:
        if guard:
            guard.active = False
        limits.close()


def execute_pipeline(job, module, module_spec, params, task, train_rows, fit_x, fit_y, predict_x, guard, limits):
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from agent.ml import enforce_threads, validate_estimator
    from agent.predictions import predict, targets, validation_columns
    module_spec.loader.exec_module(module)
    print("AUTOML_WORKER "+json.dumps({"stage": "constructing", "phase": job["phase"], "rows": train_rows, "features": len(fit_x.columns)}),flush=True)
    model = validate_estimator(enforce_threads(module.make_pipeline(task, params), job["cpu_threads"]))
    if len(targets(task)) > 1:
        from agent.ml import SequentialTargets
        model = SequentialTargets(model)
    if not callable(getattr(model, "fit", None)) or not callable(getattr(model, "predict", None)):
        raise TypeError("make_pipeline must return an object implementing fit and predict")
    started = time.monotonic()
    print("AUTOML_WORKER "+json.dumps({"stage": "fit_started", "phase": job["phase"], "cpu_threads": job["cpu_threads"]}),flush=True)
    with limits:
        if job["phase"] == "train":
            valid_indices = predict_x.index.tolist()
            model.fit(fit_x, fit_y)
            fit_seconds = time.monotonic() - started
            prediction = predict(model, predict_x, task, 'train')
            frame = pd.DataFrame(prediction, columns=validation_columns(task))
            frame.insert(0, '__row_index', valid_indices)
        else:
            model.fit(fit_x, fit_y)
            fit_seconds = time.monotonic() - started
            prediction = predict(model, predict_x, task, 'refit')
            if len(prediction) != len(predict_x):
                raise ValueError("Model changed prediction row count")
            if guard:
                guard.active = False
            ids = pd.read_csv(job["test_path"], dtype=str, keep_default_na=False, encoding="utf-8")
            frame = ids[task["output"]["id_columns"]].copy()
            for index, column in enumerate(task['output']['prediction_columns']):
                frame[column] = prediction[:, index]
            frame = frame[task["output"]["columns"]]
    if guard:
        guard.active = False
    from agent.feature_engineering import FeatureProgram
    from sklearn.pipeline import Pipeline
    fitted = model.models_ if hasattr(model, 'models_') else [model]
    for target_index, fitted_model in enumerate(fitted):
        if isinstance(fitted_model, Pipeline):
            transformer = fitted_model.named_steps.get('features')
            if isinstance(transformer, FeatureProgram):
                print('AUTOML_WORKER '+json.dumps({'stage': 'features_fitted', 'target_index': target_index,
                    'report': transformer.report_}, allow_nan=False), flush=True)
    output = Path(job["prediction_path"])
    print("AUTOML_WORKER "+json.dumps({"stage": "prediction_completed", "phase": job["phase"], "fit_seconds": fit_seconds, "prediction_rows": len(frame)}),flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8")
    model_path = Path(job["model_path"])
    snapshot = save_snapshot(model, model_path, job.get("model_snapshot_max_bytes", 16777216))
    return {"success": True, "fit_seconds": fit_seconds, "total_seconds": time.monotonic() - started,
            "rows": len(frame), "prediction_path": str(output),
            "model_path": str(model_path) if model_path.is_file() else None, "model_snapshot": snapshot}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_file", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job_file.read_text(encoding="utf-8"))
    result_path = Path(job["result_path"])
    done = threading.Event()
    watchdog = threading.Thread(target=watch_job, args=(job, done), daemon=True, name='worker-watchdog')
    watchdog.start()
    try:
        result = execute(job)
        result["peak_memory_mb"] = self_peak_mb()
        exit_code = 0
    except BaseException as exc:
        traceback.print_exc()
        result = {"success": False, "error": str(exc), "error_type": type(exc).__name__}
        exit_code = 1
    finally:
        done.set()
        watchdog.join(timeout=1)
    temp = result_path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, result_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
