"""Transport must preserve values, split isolation and actual worker outputs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent.artifacts import Artifacts
from agent.data_transport import DataTransport, read_cached, cached_chunks, worker_inputs, transport
from agent.errors import AgentError
from tools.data import read_table


class DataTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.events = []
        self.log = SimpleNamespace(event=lambda *a, **k: self.events.append((a, k)))
        self.artifacts = Artifacts(self.root/'run', 'transport-test', self.log)
        self.tables = transport(self.artifacts)

    def tearDown(self):
        self.temp.cleanup()

    def source(self, text, name='train.csv'):
        path = self.root/name
        path.write_text(text, encoding='utf-8')
        return self.artifacts.register(path, 'input', external=True)['artifact_id']

    def test_values_dtypes_order_and_missing_literals_match_csv(self):
        ident = self.source('i,f,cat,bool,blank\n9007199254740993,1.125,NA,True,\n2,,NULL,False,\n3,inf,Привет,True,\n4,-inf,,False,\n')
        expected = read_table(self.artifacts.resolve(ident))
        actual = self.tables.read(ident)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        projected = self.tables.read(ident, columns=['cat', 'i'], rows=[3, 0, 0])
        pd.testing.assert_frame_equal(projected, expected.iloc[[3, 0, 0]][['cat', 'i']], check_exact=True)
        self.assertEqual(len(self.tables.read(ident, columns=[])), 4)
        self.assertEqual(self.tables.read(ident, columns=['i'], rows=[]).shape, (0, 1))

    def test_classification_labels_preserve_text_and_have_separate_parse_key(self):
        ident = self.source('x,target\n1,001\n2,002\n3,001\n')
        spec = {'task_kind': 'multiclass_classification', 'target_column': 'target'}
        self.assertEqual(self.tables.read(ident)['target'].tolist(), [1, 2, 1])
        self.assertEqual(self.tables.read(ident, spec)['target'].tolist(), ['001', '002', '001'])

    def test_parallel_readers_reuse_one_build_and_cannot_modify_shared_inputs(self):
        ident = self.source('x,cat\n1,a\n2,b\n3,a\n')
        with patch('tools.data.read_table', wraps=read_table) as parse:
            with ThreadPoolExecutor(max_workers=2) as pool:
                frames = list(pool.map(lambda _: self.tables.read(ident), range(2)))
            self.assertEqual(parse.call_count, 1)
        frames[0].loc[0, 'x'] = 999
        self.assertEqual(frames[1].loc[0, 'x'], 1)
        self.assertEqual(self.tables.read(ident).loc[0, 'x'], 1)

    def test_fit_targets_have_no_backing_store_with_holdout_answers(self):
        ident = self.source('id,x,target,other\n0,1,10,100\n1,2,20,200\n2,3,987654321,987654322\n')
        spec = {'task_kind': 'regression', 'target_column': 'target', 'target_columns': ['target', 'other'], 'excluded_features': ['id']}
        fit, valid = self.root/'fit.json', self.root/'valid.json'
        fit.write_text('[1,0]'); valid.write_text('[2]')
        job = {'task_spec': spec, 'phase': 'train', 'train_indices_path': str(fit), 'valid_indices_path': str(valid),
               'train_path': str(self.artifacts.resolve(ident)), 'input_tables': {'train': self.tables.prepare(ident, spec)}}
        cached = worker_inputs(job)
        legacy = worker_inputs({k: v for k, v in job.items() if k != 'input_tables'})
        self.assertEqual(cached[0], legacy[0])
        for actual, expected in zip(cached[1:], legacy[1:]):
            pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        for frame in cached[1:]:
            for column in frame:
                array = frame[column].to_numpy()
                while isinstance(array, np.ndarray):
                    self.assertNotIsInstance(array, np.memmap)
                    self.assertFalse(np.isin(array, [987654321, 987654322]).any())
                    array = array.base

    def test_cache_tampering_fails_before_unpickling_or_model_execution(self):
        ident = self.source('x,target\n1,2\n3,4\n')
        reference = self.tables.prepare(ident)
        path = Path(reference['path'])
        manifest = json.loads(path.read_text())
        column = path.parent/manifest['columns']['x']['file']
        data = bytearray(column.read_bytes()); data[-1] ^= 1; column.write_bytes(data)
        with self.assertRaises(AgentError) as caught:
            read_cached(reference, ['x'])
        self.assertEqual(caught.exception.code, 'DATA_CACHE_INVALID')
        path.write_text('{}')
        with self.assertRaises(AgentError) as caught:
            read_cached(reference)
        self.assertEqual(caught.exception.code, 'DATA_CACHE_INVALID')

    def test_input_mutation_and_invalid_projections_are_rejected(self):
        ident = self.source('x\n1\n2\n')
        reference = self.tables.prepare(ident)
        for columns, rows in [(['other'], None), (['x', 'x'], None), (None, [-1]), (None, [1.0]), (None, [2])]:
            with self.subTest(columns=columns, rows=rows), self.assertRaises(AgentError):
                read_cached(reference, columns, rows)
        self.artifacts.resolve(ident).write_text('x\n999\n2\n')
        with self.assertRaises(AgentError) as caught:
            read_cached(reference)
        self.assertEqual(caught.exception.code, 'INPUT_CHANGED')

    def test_original_registered_hash_is_checked_before_build(self):
        ident = self.source('x\n1\n2\n')
        self.artifacts.resolve(ident).write_text('x\n3\n4\n')
        with self.assertRaises(AgentError) as caught:
            self.tables.prepare(ident)
        self.assertEqual(caught.exception.code, 'INPUT_CHANGED')

    def test_bounded_cache_and_disk_failures_keep_csv_compatibility(self):
        ident = self.source('x,cat\n1,a\n2,b\n')
        expected = read_table(self.artifacts.resolve(ident))
        with patch('agent.data_transport.MAX_CACHE_BYTES', 1):
            self.assertIsNone(self.tables.prepare(ident))
        pd.testing.assert_frame_equal(self.tables.read(ident), expected)
        self.assertFalse(list((self.artifacts.root/'data-cache').glob('.building-*')))
        other = DataTransport(self.artifacts)
        with patch('numpy.save', side_effect=OSError('simulated disk full')):
            self.assertIsNone(other.prepare(ident))
        self.assertFalse(list((self.artifacts.root/'data-cache').glob('.building-*')))
        with patch('agent.data_transport.MAX_CACHE_COLUMNS', 1):
            self.assertIsNone(DataTransport(self.artifacts).prepare(ident))

    def test_boolean_objects_with_missing_cells_preserve_dtype(self):
        ident = self.source('flag,x\nTrue,1\n,2\nFalse,3\n')
        self.assertIsNotNone(self.tables.prepare(ident))
        pd.testing.assert_frame_equal(self.tables.read(ident), read_table(self.artifacts.resolve(ident)))

    def test_unusual_mixed_scalars_use_lossless_csv_fallback(self):
        ident = self.source('mixed\n1\na\n')
        # pandas low_memory can infer mixed Python types across parser blocks.
        with patch('tools.data.read_table', return_value=pd.DataFrame({'mixed': [1, 'a']})):
            self.assertIsNone(self.tables.prepare(ident))
        pd.testing.assert_frame_equal(self.tables.read(ident), read_table(self.artifacts.resolve(ident)))

    def test_cache_chunks_match_full_parse_without_additional_csv_reads(self):
        ident = self.source('x,cat\n1,a\n2,NA\n3,\n4,b\n5,a\n')
        reference = self.tables.prepare(ident)
        with patch('pandas.read_csv', side_effect=AssertionError('unexpected CSV reparse')):
            with cached_chunks(reference, ['cat', 'x'], chunksize=2) as reader:
                chunks = list(reader)
        self.assertEqual([len(c) for c in chunks], [2, 2, 1])
        pd.testing.assert_frame_equal(pd.concat(chunks), read_table(self.artifacts.resolve(ident))[['cat', 'x']])

    def test_analysis_uses_cache_without_revealing_withheld_target(self):
        from tools.analysis import AnalysisService
        ident = self.source('x,target\n1,10\n2,20\n3,999999\n')
        test = self.source('x\n4\n', 'test.csv')
        spec = {'task_kind': 'regression', 'target_column': 'target', 'excluded_features': [],
                'train_artifact_id': ident, 'test_artifact_id': test}
        self.tables.prepare(ident, spec); self.tables.prepare(test)
        fit = self.artifacts.write_json('fit.json', [0, 1])
        validation = {'validation_id': 'fixed', 'train_rows_artifact_id': fit['artifact_id']}
        budget = SimpleNamespace(snapshot=lambda: {'search_allowed': True}, phase_remaining=lambda: 60)
        service = AnalysisService(spec, validation, self.artifacts, budget, self.log)
        with patch('tools.analysis.read_table', side_effect=AssertionError('unexpected CSV reparse')):
            result = service.analyze({'kind': 'summary', 'columns': ['x'], 'group_by': None, 'include_target': True})
        self.assertNotIn('999999', str(result))
        self.assertEqual(result['report']['matched_fit_rows'], 2)

    def test_real_train_and_refit_write_identical_csv_with_or_without_cache(self):
        from agent.worker import execute
        from tests.helpers import Harness
        harness = Harness()
        try:
            spec = harness.spec
            tables = transport(harness.artifacts)
            fit, valid = harness.root/'fit.json', harness.root/'valid.json'
            fit.write_text(json.dumps(list(range(72)))); valid.write_text(json.dumps(list(range(72, 90))))
            job = {'task_spec': spec, 'train_path': str(harness.artifacts.resolve(spec['train_artifact_id'])),
                   'test_path': str(harness.artifacts.resolve(spec['test_artifact_id'])),
                   'train_indices_path': str(fit), 'valid_indices_path': str(valid),
                   'code_path': str(harness.artifacts.resolve(harness.code['artifact_id'])),
                   'hyperparameters': {'C': 1.0}, 'cpu_threads': 1, 'seed': 42, 'model_snapshot_max_bytes': 0}
            for phase in ('train', 'refit'):
                outputs = []
                for mode in ('csv', 'cache'):
                    output = harness.root/(phase+'-'+mode+'.csv')
                    cache = ({'input_tables': {'train': tables.prepare(spec['train_artifact_id'], spec),
                                              'test': tables.prepare(spec['test_artifact_id'])}} if mode == 'cache' else {})
                    result = execute({**job, **cache, 'job_id': phase+'-'+mode, 'phase': phase,
                                      'prediction_path': str(output), 'model_path': str(output)+'.joblib'})
                    self.assertTrue(result['success'])
                    outputs.append(output.read_bytes())
                self.assertEqual(outputs[0], outputs[1])
        finally:
            harness.close()


if __name__ == '__main__':
    unittest.main()
