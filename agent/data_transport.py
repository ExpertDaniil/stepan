"""Run-local typed input cache. JSON carries references, never whole tables.

Only the trusted dispatcher builds caches. Readers map immutable NumPy columns
and materialize private, projected rows before any candidate code runs. No
pickle, learned preprocessing, cross-task reuse, or resident full-table cache.
"""
from contextlib import contextmanager, ExitStack
from pathlib import Path
import json
import os
import shutil
import threading
import time
import uuid

from agent.artifacts import file_sha256
from agent.errors import AgentError

VERSION = 'numpy-columns-v1'
MAX_CACHE_BYTES = 512 * 1024 * 1024
MAX_CACHE_COLUMNS = 128  # Leave file descriptors for logs/models/library workers.


def source_stamp(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ino]


@contextmanager
def mapped_column(root, entry):
    """Validate bytes before mapping; close descriptors even on decode failure."""
    import numpy as np
    path = root / entry['file']
    if path.is_symlink() or path.parent != root or file_sha256(path) != entry['sha256']:
        raise AgentError('DATA_CACHE_INVALID', 'Input cache column bytes changed')
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        if array.ndim != 1 or array.dtype.hasobject:
            raise AgentError('DATA_CACHE_INVALID', 'Invalid typed input column')
        yield array
    finally:
        array._mmap.close()


def cache_manifest(reference):
    path = Path(reference['path'])
    if path.is_symlink() or file_sha256(path) != reference['sha256']:
        raise AgentError('DATA_CACHE_INVALID', 'Input cache manifest bytes changed')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest['version'] != VERSION:
        raise AgentError('DATA_CACHE_INVALID', 'Unsupported input cache format')
    if source_stamp(manifest['source_path']) != manifest['source_stamp']:
        raise AgentError('INPUT_CHANGED', 'Input CSV changed after cache creation')
    return path, manifest


def project(manifest, arrays, selected, rows):
    import numpy as np
    import pandas as pd
    count = manifest['rows']
    index = None if rows is None else np.asarray(rows)
    if index is not None:
        if index.ndim != 1 or (index.size and (index.dtype.kind not in 'iu' or index.min() < 0 or index.max() >= count)):
            raise AgentError('INVALID_SPLIT', 'Input cache row indices are out of range')
        index = index.astype(np.intp, copy=False)
    data = {}
    for name in selected:
        column, array = manifest['columns'][name], arrays[name]
        if len(array) != count:
            raise AgentError('DATA_CACHE_INVALID', 'Input cache row count changed')
        # Full reads copy once. Advanced indexing already copies selected rows;
        # remove the memmap subclass without a second data-sized allocation.
        values = np.array(array if index is None else array[index], copy=index is None, subok=False)
        if column['kind'] == 'strings':
            dictionary = column['dictionary']
            if values.size and (values.min() < -1 or values.max() >= len(dictionary)):
                raise AgentError('DATA_CACHE_INVALID', 'Invalid string dictionary index')
            decoded = np.empty(len(values), dtype=object)
            present = values >= 0
            decoded[~present] = np.nan
            if present.any():
                decoded[present] = np.asarray(dictionary, dtype=object)[values[present]]
            values = decoded
        data[name] = values
    return pd.DataFrame(data, index=pd.RangeIndex(count) if index is None else index, copy=False)


@contextmanager
def cached_projection(reference, columns=None):
    path, manifest = cache_manifest(reference)
    selected = list(manifest['columns']) if columns is None else list(columns)
    if len(set(selected)) != len(selected) or any(c not in manifest['columns'] for c in selected):
        raise AgentError('INVALID_TABLE', 'Unknown or duplicate projected input column')
    with ExitStack() as stack:
        arrays = {name: stack.enter_context(mapped_column(path.parent, manifest['columns'][name])) for name in selected}
        yield manifest, lambda rows=None: project(manifest, arrays, selected, rows)


def read_cached(reference, columns=None, rows=None):
    """Return detached pandas data; .base cannot reveal withheld target rows."""
    with cached_projection(reference, columns) as (_, read):
        return read(rows)


@contextmanager
def cached_chunks(reference, columns, chunksize=4096):
    with cached_projection(reference, columns) as (manifest, read):
        yield (read(range(start, min(start+chunksize, manifest['rows'])))
               for start in range(0, manifest['rows'], chunksize))


class DataTransport:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.references = {}
        self.lock = threading.RLock()
        self.stored_bytes = 0

    def existing(self, artifact_id, spec=None):
        key = (artifact_id, (spec['target_column'],) if spec and spec.get('task_kind') not in {None, 'regression'} else ())
        with self.lock:
            return self.references.get(key)

    def prepare(self, artifact_id, spec=None):
        from tools.data import read_table
        import pandas as pd
        import numpy as np
        source = self.artifacts.resolve(artifact_id)
        # Classification labels must preserve literal text, including leading 0s.
        dtype = ({spec['target_column']: str} if spec and spec['task_kind'] != 'regression' else None)
        key = (artifact_id, tuple(dtype or ()))
        with self.lock:
            if key in self.references:
                return self.references[key]
            started = time.monotonic()
            stamp = source_stamp(source)
            expected = self.artifacts.meta(artifact_id)['sha256']
            if file_sha256(source) != expected:
                raise AgentError('INPUT_CHANGED', 'Input CSV differs from its registered version')
            frame = read_table(source, dtype=dtype)
            root = self.artifacts.safe_path('data-cache')
            root.mkdir(parents=True, exist_ok=True)
            temporary = root / ('.building-' + uuid.uuid4().hex)
            temporary.mkdir(mode=0o700)
            try:
                estimate = int(frame.memory_usage(index=False, deep=True).sum())
                if len(frame.columns) > MAX_CACHE_COLUMNS:
                    raise ValueError('cache_column_allowance')
                if estimate > MAX_CACHE_BYTES-self.stored_bytes or shutil.disk_usage(root).free < 2*estimate+64*1024*1024:
                    raise ValueError('cache_byte_allowance')
                metadata = {}
                for number, name in enumerate(frame):
                    if time.monotonic()-started > 15:
                        raise ValueError('cache_build_time_allowance')
                    values = frame[name].to_numpy()
                    entry = {'kind': 'numeric', 'file': str(number)+'.npy'}
                    if values.dtype.kind not in 'biuf':
                        # Unusual mixed Python scalars retain the exact CSV path.
                        if pd.api.types.infer_dtype(values, skipna=True) not in {'string', 'boolean', 'empty'}:
                            raise ValueError('unsupported_mixed_dtype')
                        codes, unique = pd.factorize(values, sort=False)
                        values = codes.astype(np.int32 if len(unique) < 2**31 else np.int64)
                        entry.update(kind='strings', dictionary=unique.tolist())
                    target = temporary / entry['file']
                    np.save(target, values, allow_pickle=False)
                    entry['sha256'] = file_sha256(target)
                    metadata[name] = entry
                if source_stamp(source) != stamp or file_sha256(source) != expected:
                    raise AgentError('INPUT_CHANGED', 'Input CSV changed during cache creation')
                manifest = {'version': VERSION, 'source_path': str(source), 'source_sha256': expected,
                            'source_stamp': stamp, 'rows': len(frame), 'columns': metadata}
                (temporary/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
                size = sum(p.stat().st_size for p in temporary.iterdir())
                if size > MAX_CACHE_BYTES-self.stored_bytes:
                    raise ValueError('cache_byte_allowance')
                destination = root / uuid.uuid4().hex
                os.replace(temporary, destination)
                manifest_path = destination/'manifest.json'
                reference = {'path': str(manifest_path), 'sha256': file_sha256(manifest_path)}
                self.stored_bytes += size
                self.references[key] = reference
                self.artifacts.register(manifest_path, 'input_cache_manifest')
                self.artifacts.log.event('data.cache_prepared', source_artifact_id=artifact_id,
                    rows=len(frame), columns=len(frame.columns), size_bytes=size,
                    build_seconds=time.monotonic()-started, format=VERSION)
                return reference
            except (OSError, ValueError) as exc:
                # Optional acceleration must not prevent the first valid output.
                self.references[key] = None
                self.artifacts.log.event('data.cache_bypassed', source_artifact_id=artifact_id,
                                         reason=str(exc)[:200])
                return None
            finally:
                shutil.rmtree(temporary, ignore_errors=True)

    def read(self, artifact_id, spec=None, columns=None, rows=None):
        from tools.data import read_table
        reference = self.prepare(artifact_id, spec)
        if reference:
            return read_cached(reference, columns, rows)
        dtype = ({spec['target_column']: str} if spec and spec['task_kind'] != 'regression' else None)
        source = self.artifacts.resolve(artifact_id)
        usecols = list(read_table(source, nrows=0).columns[:1]) if columns == [] else columns
        frame = read_table(source, dtype=dtype, usecols=usecols)
        if columns is not None:
            frame = frame[columns]
        return frame.iloc[rows].copy() if rows is not None else frame


def worker_inputs(job):
    """Read only the required X/y rows; maintain CSV compatibility for old jobs."""
    import pandas as pd
    from agent.predictions import read_training, targets
    from tools.data import read_table
    task = job['task_spec']
    names = targets(task)
    reference = job.get('input_tables', {}).get('train')
    fit_indices = json.loads(Path(job['train_indices_path']).read_text()) if job['phase'] == 'train' else None
    valid_indices = json.loads(Path(job['valid_indices_path']).read_text()) if job['phase'] == 'train' else None
    if reference:
        # Schema and count are validated by read_cached before use.
        empty = read_cached(reference, columns=[])
        manifest = json.loads(Path(reference['path']).read_text(encoding='utf-8'))
        features = [c for c in manifest['columns'] if c not in {*names, *task['excluded_features']}]
        train_rows = len(empty)
        fit_x = read_cached(reference, features, fit_indices)
        target = read_cached(reference, names, fit_indices)
        predict_x = read_cached(reference, features, valid_indices) if job['phase'] == 'train' else None
    else:
        train = read_training(job['train_path'], task)
        train_rows = len(train)
        features = [c for c in train if c not in {*names, *task['excluded_features']}]
        selected = train if fit_indices is None else train.iloc[fit_indices]
        fit_x, target = selected[features].copy(), selected[names].copy()
        predict_x = train.iloc[valid_indices][features].copy() if job['phase'] == 'train' else None
    if job['phase'] == 'refit':
        test_reference = job.get('input_tables', {}).get('test')
        predict_x = (read_cached(test_reference, features) if test_reference else
                     read_table(job['test_path'], usecols=features)[features].copy())
    target = target.apply(pd.to_numeric, errors='raise') if task['task_kind'] == 'regression' else target.astype(str)
    return train_rows, fit_x, target if len(names) > 1 else target[names[0]], predict_x


def transport(artifacts):
    # Shared only inside this run's trusted services; no module-global cache.
    with artifacts.lock:
        if not hasattr(artifacts, '_data_transport'):
            artifacts._data_transport = DataTransport(artifacts)
        return artifacts._data_transport
