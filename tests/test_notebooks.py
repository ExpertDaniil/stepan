import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from agent.config import ROOT
from agent.context import output_schemas
from agent.errors import AgentError
from agent.llm.gateway import Gateway
from agent.researcher import Researcher
from tests.graph_helpers import GraphHarness
from tools.notebooks import NotebookLibrary, NOTEBOOK_SCHEMA


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness(research_mode='adaptive')
        g = self.h.g
        self.library = NotebookLibrary(g.spec, {'train_columns': ['Patient', 'Weeks']}, g.artifacts, schemas=g.schemas)

    def tearDown(self):
        self.h.close()

    def test_original_bytes_licenses_provenance_and_annotation_cells(self):
        self.assertEqual(self.h.g.schemas.schema['$defs']['NotebookQuery'], NOTEBOOK_SCHEMA)
        root = self.library.root
        for book in self.library.notebooks.values():
            raw = (root/book['path']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), book['sha256'])
            self.assertTrue(book['original'])
            self.assertFalse(book['executed_by_this_project'])
            cells = json.loads(raw)['cells']
            self.assertEqual(len(cells), book['cell_count'])
            for approach in self.library.approaches.values():
                if approach['notebook_id'] == book['id']:
                    for i in approach['cell_indices']+approach['context_cell_indices']:
                        self.assertLess(i, len(cells))
                        self.assertGreaterEqual(i, 0)
            provenance = json.loads((root/book['provenance_path']).read_text(encoding='utf-8'))
            self.assertEqual(provenance['license'], book['license'])
            for source in provenance['files']:
                actual = (root/book['path']).parent/source['path']
                self.assertEqual(hashlib.sha256(actual.read_bytes()).hexdigest(), source['sha256'])

    def test_search_matches_concrete_features_without_claiming_semantic_proof(self):
        result = self.library.consult({'kind': 'search', 'keywords': [], 'limit': 2})
        self.assertEqual(result['total_matches'], 4)
        self.assertEqual(len(result['approaches']), 2)
        card = result['approaches'][0]
        self.assertEqual(card['matching_facts']['column_name_hints']['group_id'], ['Patient'])
        self.assertFalse(card['matching_facts']['semantic_roles_confirmed'])
        self.assertTrue(card['risks'])
        self.library.task_kind = 'classification'
        mismatches = self.library.consult({'kind': 'search', 'offset': 2, 'limit': 2})
        self.assertEqual(len(mismatches['approaches']), 2)
        self.assertTrue(all(not c['matching_facts']['task_kind_matches'] for c in mismatches['approaches']))
        self.assertIsNone(mismatches['next_offset'])

    def test_read_pages_only_source_and_keeps_context_and_risks(self):
        query = {'kind': 'read', 'approach_id': 'osic-linear-baseline', 'cell_index': 2, 'max_chars': 31}
        first = self.library.consult(query)
        self.assertEqual(first['next_offset'], 31)
        second = self.library.consult({**query, 'offset': 31})
        book = self.library.notebooks[first['notebook_id']]
        cell = json.loads((self.library.root/book['path']).read_bytes())['cells'][2]
        self.assertEqual(first['source']+second['source'], ''.join(cell['source'])[:62])
        self.assertFalse(first['outputs_included'])
        self.assertFalse(first['execution_performed'])
        self.assertNotIn('outputs', first)
        self.assertEqual(first['source_identity']['commit'], 'cd6dfa56c012e3c0eb81ed15534f436b3ba16d19')
        self.assertIn('simple-logreg.ipynb', first['source_identity']['source_url'])
        self.assertEqual(first['context_cell_indices'], [0, 1])
        saved = self.h.g.artifacts.read_json(first['artifact_id'])
        self.assertEqual(saved['sha256'], book['sha256'])
        self.assertIn('not_instructions', saved['material_status'])

    def test_unannotated_cells_changed_source_and_unsafe_paths_are_rejected(self):
        with self.assertRaises(AgentError):
            self.library.consult({'kind': 'read', 'approach_id': 'osic-linear-baseline', 'cell_index': 8})
        with self.assertRaises(AgentError):
            self.library.consult({'kind': 'search', 'approach_id': 'unused'})
        with self.assertRaises(ValueError):
            self.library.consult({'kind': 'read', 'approach_id': 'osic-linear-baseline', 'cell_index': 2, 'max_chars': 6000})
        with tempfile.TemporaryDirectory() as temp:
            shutil.copytree(self.library.root, Path(temp)/'sources')
            local = NotebookLibrary(self.h.g.spec, {}, self.h.g.artifacts, root=Path(temp)/'sources')
            source = local.root/local.notebooks['osic-linear-original']['path']
            source.write_bytes(source.read_bytes()+b' ')
            with self.assertRaises(AgentError) as error:
                local.consult({'kind': 'read', 'approach_id': 'osic-linear-baseline', 'cell_index': 2})
            self.assertEqual(error.exception.code, 'NOTEBOOK_SOURCE_CHANGED')
            local.notebooks['osic-linear-original']['path'] = '../outside.ipynb'
            with self.assertRaises(AgentError):
                local.consult({'kind': 'read', 'approach_id': 'osic-linear-baseline', 'cell_index': 2})

    def test_researcher_reads_local_cells_through_gateway_and_rejects_them_as_measured_evidence(self):
        g = self.h.g
        for cite_reference in (False, True):
            with self.subTest(cite_reference=cite_reference):
                class Transport:
                    def complete(transport, request, timeout, context):
                        if request['messages'][-1]['role'] != 'tool':
                            call = {'id': 'local-read', 'type': 'function', 'function': {'name': 'consult_notebook',
                                'arguments': json.dumps({'kind': 'read', 'approach_id': 'osic-linear-baseline', 'cell_index': 5})}}
                            message = {'role': 'assistant', 'content': None, 'tool_calls': [call]}
                        else:
                            material = json.loads(request['messages'][-1]['content'])
                            self.assertIn('LinearRegression', material['source'])
                            self.assertFalse(material['execution_performed'])
                            message = {'role': 'assistant', 'content': json.dumps({
                                'action': 'catalog', 'recipe_id': 'linear-regressor-v1', 'hyperparameters': {},
                                'hypothesis': 'Сравнить собственную линейную модель на действующей проверке',
                                'expected_effect': 'Получить независимое измерение', 'feature_steps': [], 'code': None,
                                'tuning': None, 'evidence_ids': [material['artifact_id']] if cite_reference else []})}
                        return {'choices': [{'message': message, 'finish_reason': 'stop'}],
                                'usage': {'prompt_tokens': 100, 'completion_tokens': 100}}
                    def close(transport):
                        pass
                gateway = Gateway(g.config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), Transport())
                gateway.tools.notebooks = self.library
                # The configured production environment additionally attaches analyze_data.
                from tools.analysis import AnalysisService
                gateway.tools.analysis = AnalysisService(g.spec, g.validation, g.artifacts, g.budget, g.log)
                researcher = Researcher(g.bus, gateway, g.artifacts, g.spec, g.catalog, g.config, g.log)
                researcher.validation_id = g.validation['validation_id']
                ticket = self.h.issue(peer='researcher-2' if cite_reference else 'researcher-1')
                try:
                    if cite_reference:
                        with self.assertRaises(AgentError) as error:
                            researcher.propose(ticket['researcher_request']['payload'])
                        self.assertEqual(error.exception.code, 'INVALID_RESEARCH_ACTION')
                    else:
                        proposal = researcher.propose(ticket['researcher_request']['payload'])
                        self.assertEqual(proposal['plan']['recipe_id'], 'linear-regressor-v1')
                finally:
                    gateway.close()
