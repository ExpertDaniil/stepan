"""Research obtains fresh bounded evidence without sharing holdout labels or SQL."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from agent.artifacts import Artifacts
from agent.errors import AgentError
from tools.analysis import ANALYSIS_SCHEMA, AnalysisService
from tools.data import read_table


def query(kind='summary', **changes):
    value = dict(kind=kind, columns=['x'], group_by=None, include_target=False)
    value.update(changes)
    return value


class AnalysisAccessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.events = []
        self.log = SimpleNamespace(event=lambda *args, **kwargs: self.events.append((args, kwargs)))
        self.artifacts = Artifacts(self.root/'run', 'synthetic-analysis', self.log)
        self.budget = SimpleNamespace(snapshot=lambda: {'search_allowed': True}, phase_remaining=lambda: 60)

    def tearDown(self):
        self.temp.cleanup()

    def service(self, train=None, test=None, indices=None):
        train = train if train is not None else pd.DataFrame({'x': range(30000),
            'group': ['ordinary']*29998+['rare', 'holdout-only'], 'target': [7]*29999+[999999]})
        test = test if test is not None else pd.DataFrame({'x': [1, 2], 'group': ['rare', 'ordinary'], 'target': [555555]*2})
        spec = {'target_column': 'target', 'excluded_features': [], 'dictionary_artifact_id': None}
        for role, frame in [('train', train), ('test', test)]:
            path = self.root/(role+'.csv'); frame.to_csv(path, index=False)
            spec[role+'_artifact_id'] = self.artifacts.register(path, 'synthetic', external=True)['artifact_id']
        partition = self.artifacts.write_json('fit.json', indices if indices is not None else list(range(len(train)-1)))
        validation = {'validation_id': 'fixed-synthetic', 'train_rows_artifact_id': partition['artifact_id']}
        return AnalysisService(spec, validation, self.artifacts, self.budget, self.log)

    def test_filter_finds_rare_group_before_sampling_and_excludes_holdout(self):
        service = self.service()
        result = service.analyze(query(columns=['group'], include_target=True,
            filter={'column': 'group', 'value': 'rare'}))['report']
        self.assertEqual(result['matched_fit_rows'], 1)
        self.assertEqual(result['analyzed_rows'], 1)
        self.assertFalse(result['sampled'])
        self.assertEqual(result['columns'][1]['quantiles']['1.0'], 7)
        empty = service.analyze(query(include_target=True, filter={'column': 'group', 'value': 'holdout-only'}))['report']
        self.assertEqual(empty['matched_fit_rows'], 0)
        self.assertIsNone(empty['columns'][0]['missing_fraction'])
        self.assertTrue(all(c['unique'] == 0 for c in empty['columns']))

    def test_uniform_sample_is_bounded_and_population_is_explicit(self):
        result = self.service().analyze(query())['report']
        self.assertEqual(result['fit_rows'], 29999)
        self.assertEqual(result['matched_fit_rows'], 29999)
        self.assertEqual(result['analyzed_rows'], 20000)
        self.assertTrue(result['sampled'])

    def test_excluded_training_identifier_remains_available_for_fit_diagnosis(self):
        service = self.service()
        service.spec['excluded_features'] = ['group']
        report = service.analyze(query('group', columns=['group', 'x'], group_by='group',
            include_target=True, filter={'column': 'group', 'value': 'rare'}))['report']
        self.assertEqual(report['diagnostic_only_columns'], ['group'])
        self.assertEqual(report['matched_fit_rows'], 1)
        self.assertEqual(report['groups'][0]['target']['quantiles']['1.0'], 7)
        self.assertEqual(service.spec['excluded_features'], ['group'])
        held_out = service.analyze(query(columns=['group'],
            filter={'column': 'group', 'value': 'holdout-only'}))['report']
        self.assertEqual(held_out['matched_fit_rows'], 0)

    def test_schema_pages_discover_columns_beyond_initial_preview(self):
        train = pd.DataFrame({**{'f'+str(i): [i]*3 for i in range(140)}, 'target': [1, 2, 3]})
        service = self.service(train, train.drop(columns='target'))
        names = []
        offset = 0
        while offset is not None:
            report = service.analyze(query('schema', columns=[], offset=offset, limit=12))['report']
            self.assertEqual(report['scope'], 'schema_only')
            names.extend(c['column'] for c in report['columns'])
            offset = report['next_offset']
        self.assertEqual(names, list(train))
        self.assertEqual(report['columns'][-1]['role'], 'target')
        self.assertNotIn('quantiles', str(report))

    def test_cached_reports_and_history_are_not_mutable_by_other_peers(self):
        service = self.service()
        first = service.analyze(query())
        original_id = first['artifact_id']
        first['report']['columns'].clear()
        recent = service.recent(); recent[0]['query']['columns'].append('injected')
        saved = service.analyze(query('report', columns=[], report_id=original_id))
        self.assertEqual(saved, service.analyze(query()))
        self.assertTrue(saved['report']['columns'])
        self.assertEqual(service.recent()[0]['query']['columns'], ['x'])
        self.assertEqual(len(service.recent()), 1)

    def test_concurrent_requests_share_one_immutable_report_without_sql(self):
        service = self.service()
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda _: service.analyze(query()), range(2)))
        self.assertEqual(replies[0], replies[1])
        self.assertIsNot(replies[0], replies[1])
        self.assertEqual(len(service.recent()), 1)

    def test_comparison_filters_both_populations_and_never_test_target(self):
        report = self.service().analyze(query('compare', filter={'column': 'group', 'value': 'rare'}))['report']
        self.assertEqual(report['matched_fit_rows'], 1)
        self.assertEqual(report['matched_test_rows'], 1)
        self.assertEqual(report['test_rows'], 2)
        self.assertNotIn('555555', str(report))
        self.assertEqual([c['column'] for c in report['test']], ['x'])

    def test_unavailable_comparison_is_domain_error_not_key_error(self):
        service = self.service(test=pd.DataFrame({'other': [1]}))
        with self.assertRaises(AgentError) as caught:
            service.analyze(query('compare'))
        self.assertEqual(caught.exception.code, 'INVALID_ANALYSIS')

    def test_malformed_and_target_filter_requests_are_rejected(self):
        service = self.service()
        invalid = [None, [], {}, query(columns=['']), query(filter={'column': 'target', 'value': 7}),
            query(filter={'column': 'x', 'value': [1]}), query(filter={'column': 'x', 'value': float('nan')}),
            query(filter={'column': 'x', 'value': 'a'*201}), query(offset=1),
            query('schema', columns=[], offset=True), query('schema', columns=[], limit=13)]
        for item in invalid:
            with self.subTest(query=item), self.assertRaises(AgentError) as caught:
                service.analyze(item)
            self.assertEqual(caught.exception.code, 'INVALID_ANALYSIS')

    def test_stream_reads_only_requested_columns_and_stops_at_time_boundary(self):
        service = self.service()
        ticks = iter([True, True, False])
        self.budget.snapshot = lambda: {'search_allowed': next(ticks, False)}
        with patch('tools.analysis.read_table', wraps=read_table) as reader, self.assertRaises(AgentError) as caught:
            service.analyze(query())
        self.assertEqual(caught.exception.code, 'SEARCH_CLOSED')
        self.assertEqual(reader.call_args.kwargs['usecols'], ['x'])
        self.assertEqual(reader.call_args.kwargs['chunksize'], 4096)
        self.assertEqual(service.recent(), [])

    def test_public_schema_matches_saved_contract(self):
        import json
        schema = json.loads((Path(__file__).parents[1]/'contracts/contracts.schema.json').read_text(encoding='utf-8'))
        self.assertEqual(schema['$defs']['AnalysisQuery'], ANALYSIS_SCHEMA)

    def test_boolean_summary_and_missing_filter_are_supported(self):
        train = pd.DataFrame({'x': [True, False, True, False], 'group': ['a', None, None, 'a'], 'target': [3, 4, 5, 99]})
        service = self.service(train, train.drop(columns='target'))
        report = service.analyze(query(filter={'column': 'group', 'value': None}))['report']
        self.assertEqual(report['matched_fit_rows'], 2)
        self.assertEqual(report['columns'][0]['unique'], 2)
        self.assertEqual(sorted(report['columns'][0]['top_values']), [['False', 1], ['True', 1]])


if __name__ == '__main__':
    unittest.main()
