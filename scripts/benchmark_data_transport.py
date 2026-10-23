"""Paired synthetic CSV/cache input preparation in fresh concurrent workers.

Measures transport only, including integrity checks and private split copies;
not training speed, live model reasoning, or competition quality. No API calls.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def child(job_path):
    import numpy as np
    import pandas as pd
    from agent.data_transport import worker_inputs
    from agent.worker import self_peak_mb
    job = json.loads(Path(job_path).read_text())
    csv_reads = 0
    original = pd.read_csv

    def counted(*args, **kwargs):
        nonlocal csv_reads
        csv_reads += 1
        return original(*args, **kwargs)

    pd.read_csv = counted
    started = time.perf_counter()
    _, fit_x, fit_y, valid_x = worker_inputs(job)
    elapsed = time.perf_counter()-started
    # Linux ru_maxrss can include a larger pre-exec parent high-water mark.
    # VmHWM belongs to this executable's current address space.
    status = Path('/proc/self/status')
    peak = (int(next(line.split()[1] for line in status.read_text().splitlines()
                     if line.startswith('VmHWM:')))/1024 if status.is_file() else self_peak_mb())
    digest = hashlib.sha256()
    for frame in (fit_x, fit_y, valid_x):
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return {'seconds': elapsed, 'peak_rss_mb': peak, 'csv_reads': csv_reads, 'data_sha256': digest.hexdigest()}


def benchmark(rows=100000, numeric=40, repeats=3, workers=2):
    import numpy as np
    import pandas as pd
    from agent.artifacts import Artifacts
    from agent.data_transport import transport
    with tempfile.TemporaryDirectory(prefix='stepan-transport-benchmark-') as temp:
        root = Path(temp)
        rng = np.random.default_rng(20261023)
        frame = pd.DataFrame({f'number_{i}': rng.normal(size=rows) for i in range(numeric)})
        for index in range(6):
            frame[f'category_{index}'] = rng.choice(['NA', 'NULL', 'обычный', 'rare', ''], size=rows)
        frame['id'] = np.arange(rows)
        frame['target'] = 3*frame.number_0+frame.number_1
        frame.loc[::29, 'number_0'] = np.nan
        path = root/'train.csv'
        frame.to_csv(path, index=False)
        del frame
        log = SimpleNamespace(event=lambda *a, **k: None)
        artifacts = Artifacts(root/'run', 'benchmark', log)
        source = artifacts.register(path, 'input', external=True)['artifact_id']
        spec = {'task_kind': 'regression', 'target_column': 'target', 'excluded_features': ['id']}
        fit, valid = root/'fit.json', root/'valid.json'
        fit.write_text(json.dumps(list(range(int(rows*.8)))))
        valid.write_text(json.dumps(list(range(int(rows*.8), rows))))
        build_start = time.perf_counter()
        reference = transport(artifacts).prepare(source, spec)
        build_seconds = time.perf_counter()-build_start
        if not reference:
            raise RuntimeError('Benchmark did not create a typed cache')
        job = {'phase': 'train', 'task_spec': spec, 'train_path': str(path),
               'train_indices_path': str(fit), 'valid_indices_path': str(valid)}
        paths = {}
        for mode in ('csv', 'cache'):
            paths[mode] = root/(mode+'.job.json')
            paths[mode].write_text(json.dumps({**job, **({'input_tables': {'train': reference}} if mode == 'cache' else {})}))
        records = {'csv': [], 'cache': []}
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

        def measure(mode):
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', str(paths[mode])],
                                    capture_output=True, text=True, check=True, env=env)
            return json.loads(result.stdout)

        for repeat in range(repeats):
            # Alternate order; both methods use the OS page cache, no root-only
            # cache dropping. Every batch runs fresh competing worker processes.
            for mode in (('csv', 'cache') if repeat % 2 == 0 else ('cache', 'csv')):
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    records[mode].extend(pool.map(lambda _: measure(mode), range(workers)))
        hashes = {r['data_sha256'] for entries in records.values() for r in entries}
        if len(hashes) != 1:
            raise AssertionError('CSV and cache produced different model inputs')
        report = {'rows': rows, 'numeric_columns': numeric, 'string_columns': 6, 'workers_per_batch': workers,
                  'repeats': repeats, 'csv_bytes': path.stat().st_size,
                  'cache_bytes': transport(artifacts).stored_bytes, 'cache_build_seconds': build_seconds,
                  'identical_model_inputs': True, 'records': records,
                  'interpretation': 'Synthetic transport benchmark with warm OS cache; excludes imports/training. Includes hashes, decoding and detached fit/validation arrays.'}
        for mode, entries in records.items():
            report[mode] = {'median_seconds': statistics.median(r['seconds'] for r in entries),
                            'median_peak_rss_mb': statistics.median(r['peak_rss_mb'] for r in entries),
                            'total_csv_reads': sum(r['csv_reads'] for r in entries)}
        report['median_speedup'] = report['csv']['median_seconds']/report['cache']['median_seconds']
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--rows', type=int, default=100000)
    parser.add_argument('--numeric', type=int, default=40)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.rows < 10 or args.numeric < 2 or args.repeats < 1:
        parser.error('Need rows>=10, numeric>=2 and repeats>=1')
    result = child(args.child) if args.child else benchmark(args.rows, args.numeric, args.repeats, args.workers)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text+'\n', encoding='utf-8')
    print(text)
