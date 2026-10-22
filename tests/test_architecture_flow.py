import tempfile
import unittest
from pathlib import Path
from agent.errors import AgentError
from agent.research_tools import ResearchTools, SchemaInspection
from scripts.demo_research import run_demo
from tests.graph_helpers import GraphHarness


class ArchitectureFlowTests(unittest.TestCase):
    def test_complete_schema_is_pageable_without_rows_or_target_values(self):
        profile = {'train_columns': ['c' + str(i) for i in range(300)], 'train_dtypes': {}, 'train_missing_fraction': {}}
        inspector = SchemaInspection(profile)
        page = inspector.inspect({'table': 'train', 'offset': 280, 'limit': 64})
        self.assertEqual(page['columns'][-1]['name'], 'c299')
        self.assertIsNone(page['next_offset'])
        self.assertNotIn('values', page)

    def test_protocol_view_is_a_copy_and_capabilities_are_explicit(self):
        h = GraphHarness()
        try:
            tools = ResearchTools(h.g.bus, h.g.artifacts)
            with self.assertRaises(AgentError):
                tools.definitions(['analyze_data'])
            with self.assertRaises(AgentError):
                tools.definitions(['unknown'])
            view = h.controller.evaluator.describe_protocol(h.g.validation['validation_id'])
            self.assertNotIn('valid_indices', view)
            view['proposal']['reason'] = 'changed'
            self.assertNotEqual(h.controller.evaluator.describe_protocol(h.g.validation['validation_id'])['proposal']['reason'], 'changed')
        finally:
            h.close()

    def test_extra_facts_to_real_fit_and_atomic_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_demo(Path(directory) / 'demo')
            self.assertEqual(result['state'], 'completed')
            self.assertEqual(result['analysis_reports'], 1)
            self.assertGreaterEqual(len(result['publications']), 2)
            self.assertTrue(Path(result['output']).is_file())
