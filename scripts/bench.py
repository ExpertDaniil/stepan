"""Local hidden-test bench: split a task's train, run solution.py, score its CSV.

Example:
  python scripts/bench.py --task data/regression/task.md --split temporal:Year_Factor --budget 1800
  python scripts/bench.py --task data/classification/task.md --split stratified --budget 1800 --mock

The organizers' baseline value is unknown locally, so the case formula is applied
against a fixed reference (sklearn HistGradientBoosting with default settings) on
the same hidden split. Compare variants by the raw metric and the proxy score.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def split_rows(frame, target, how, fraction, seed):
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    if how.startswith('temporal:'):
        column = how.split(':', 1)[1]
        last = frame[column].max()
        return frame.index[frame[column] < last], frame.index[frame[column] == last]
    if how.startswith('group:'):
        column = how.split(':', 1)[1]
        a, b = next(GroupShuffleSplit(1, test_size=fraction, random_state=seed).split(frame, groups=frame[column].astype(str)))
        return frame.index[a], frame.index[b]
    stratify = frame[target].astype(str) if how == 'stratified' else None
    a, b = train_test_split(np.arange(len(frame)), test_size=fraction, random_state=seed, stratify=stratify)
    return frame.index[a], frame.index[b]


def score(kind, metric, truth, prediction):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
    if kind == 'regression':
        truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
        return float(mean_absolute_error(truth, prediction)) if metric == 'mae' else float(np.sqrt(np.mean((truth-prediction)**2)))
    truth, prediction = np.asarray(truth).astype(str), np.asarray(prediction).astype(str)
    return float(accuracy_score(truth, prediction)) if metric == 'accuracy' else float(f1_score(truth, prediction, average='macro'))


def points(kind, value, reference):
    if kind == 'regression':
        return 100*max(0.0, 1-value/reference) if reference > 0 else 0.0
    return 100*max(0.0, (value-reference)/(1-reference)) if reference < 1 else 0.0


def reference_prediction(kind, train, test, target, excluded):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    features = [c for c in train.columns if c not in excluded and c != target]
    def encode(frame):
        out = frame[features].copy()
        for column in features:
            if out[column].dtype == object:
                codes = {v: i for i, v in enumerate(sorted(train[column].dropna().astype(str).unique()))}
                out[column] = out[column].astype(str).map(codes).astype(float)
        return out
    model = (HistGradientBoostingRegressor if kind == 'regression' else HistGradientBoostingClassifier)(random_state=0)
    model.fit(encode(train), train[target] if kind == 'regression' else train[target].astype(str))
    return model.predict(encode(test))


def main():
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, type=Path)
    parser.add_argument('--target', default=None, help='Defaults to the single train column absent from test')
    parser.add_argument('--id', default='id')
    parser.add_argument('--split', default='random', help='random | stratified | temporal:COL | group:COL')
    parser.add_argument('--fraction', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--metric', default=None, help='rmse | mae | f1_macro | accuracy; inferred from task kind')
    parser.add_argument('--budget', type=int, default=1800)
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--api-down', action='store_true', help='Point the provider at an unreachable address')
    parser.add_argument('--env-file', type=Path, default=None)
    parser.add_argument('--results', type=Path, default=ROOT/'bench_results.csv')
    parser.add_argument('--keep', action='store_true')
    args = parser.parse_args()
    source = args.task.resolve().parent
    train = pd.read_csv(source/'train.csv', keep_default_na=False, na_values=[''])
    test = pd.read_csv(source/'test.csv', nrows=5, keep_default_na=False, na_values=[''])
    target = args.target or next(c for c in train.columns if c not in test.columns)
    kind = 'regression' if pd.api.types.is_numeric_dtype(train[target]) and train[target].nunique() > 20 else 'classification'
    metric = args.metric or ('rmse' if kind == 'regression' else 'f1_macro')
    fit_rows, hidden_rows = split_rows(train, target, args.split, args.fraction, args.seed)
    work = Path(tempfile.mkdtemp(prefix='stepan-bench-'))
    task_dir = work/'task'
    task_dir.mkdir()
    for path in source.iterdir():
        if path.is_file() and path.name not in {'train.csv', 'test.csv', 'sample_submission.csv', 'submission.csv'}:
            shutil.copy2(path, task_dir/path.name)
    fit, hidden = train.loc[fit_rows], train.loc[hidden_rows]
    fit.to_csv(task_dir/'train.csv', index=False)
    hidden.drop(columns=[target]).to_csv(task_dir/'test.csv', index=False)
    sample = hidden[[args.id, target]].copy()
    sample[target] = fit[target].iloc[0]
    sample.to_csv(task_dir/'sample_submission.csv', index=False)
    env = dict(os.environ, TIME_BUDGET_SECONDS=str(args.budget), WORK_DIR=str(work/'runs'))
    if args.api_down:
        env.update(OPENAI_BASE_URL='http://127.0.0.1:9/v1', OPENAI_API_KEY='invalid-key-for-bench')
    command = [sys.executable, str(ROOT/'solution.py'), str(task_dir/'task.md')]
    if args.mock:
        command += ['--llm-mode', 'mock']
    if args.env_file:
        command += ['--env-file', str(args.env_file.resolve())]
    started = time.monotonic()
    try:
        done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=args.budget+180)
        code, stderr = done.returncode, done.stderr
    except subprocess.TimeoutExpired as error:
        code, stderr = 'timeout', (error.stderr or b'').decode('utf-8', 'replace') if isinstance(error.stderr, bytes) else str(error.stderr)
    elapsed = time.monotonic()-started
    output = task_dir/'submission.csv'
    result = {'task': str(args.task), 'split': args.split, 'budget': args.budget, 'mock': args.mock, 'api_down': args.api_down,
              'exit_code': code, 'seconds': round(elapsed, 1), 'metric': metric, 'value': None, 'reference': None,
              'proxy_points': None, 'valid_csv': False, 'git': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
              cwd=ROOT, capture_output=True, text=True).stdout.strip()}
    reference = score(kind, metric, hidden[target], reference_prediction(kind, fit, hidden, target, {args.id}))
    result['reference'] = round(reference, 6)
    if output.is_file():
        submission = pd.read_csv(output, keep_default_na=False, na_values=[''])
        valid = (list(submission.columns) == [args.id, target] and len(submission) == len(hidden)
                 and submission[args.id].astype(str).tolist() == hidden[args.id].astype(str).tolist()
                 and submission[target].notna().all())
        result['valid_csv'] = bool(valid)
        if valid:
            value = score(kind, metric, hidden[target], submission[target])
            result.update(value=round(value, 6), proxy_points=round(points(kind, value, reference), 3))
    result['stderr_tail'] = (stderr or '')[-400:].replace('\n', ' | ')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    new = not args.results.exists()
    with args.results.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        if new:
            writer.writeheader()
        writer.writerow(result)
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    else:
        print('kept', work)


if __name__ == '__main__':
    main()
