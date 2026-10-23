from copy import deepcopy
import json
import unittest

from agent.contracts import Schemas
from agent.context import output_schemas
from agent.llm.gateway import Gateway
from agent.llm.protocol import prepare_tool_calls
from agent.tool_arguments import normalize_tool_arguments
from tests.graph_helpers import GraphHarness
from tests.test_adaptive_research import action
from tools.analysis import AnalysisService
from tools.knowledge import KNOWLEDGE_SCHEMA


class ToolSemanticsTests(unittest.TestCase):
    def test_id_read_ignores_ranking_words_but_preserves_unread_pages(self):
        left = normalize_tool_arguments('search_knowledge', {'ids': ['method-a'], 'query': 'first wording', 'tags': ['missing']})
        right = normalize_tool_arguments('search_knowledge', {'ids': ['method-a'], 'query': 'a different question', 'max_results': 1})
        self.assertEqual(left, right)
        self.assertNotEqual(left, normalize_tool_arguments('search_knowledge', {'ids': ['method-a'], 'offset': 1000}))
        schemas = Schemas()
        self.assertEqual(schemas.schema['$defs']['KnowledgeQuery'], KNOWLEDGE_SCHEMA)
        schemas.validate({'ids': ['method-a']}, 'KnowledgeQuery')

    def test_semantic_duplicates_in_one_batch_keep_exact_tool_reply_linkage(self):
        def call(ident, query):
            return {'id': ident, 'function': {'name': 'search_knowledge', 'arguments': json.dumps({'ids': ['method-a'], 'query': query})}}
        schemas = Schemas()
        prepared = prepare_tool_calls([call('a', 'first'), call('b', 'second')],
            {'search_knowledge': KNOWLEDGE_SCHEMA}, schemas._check, 'researcher-1')
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]['args']['query'], '')
        self.assertIsNone(prepared[0]['error'])

    def test_report_reads_fill_defaults_but_do_not_hide_incompatible_requests(self):
        args = normalize_tool_arguments('analyze_data', {'kind': 'report', 'report_id': 'a'})
        self.assertEqual(args['columns'], [])
        bad = normalize_tool_arguments('analyze_data', {**args, 'columns': ['new-analysis-column']})
        self.assertEqual(bad['columns'], ['new-analysis-column'])

    def test_reading_a_just_returned_report_reuses_it_and_closes_tool_phase(self):
        h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800,
                         llm_max_calls=None, llm_max_total_output_tokens=None)
        try:
            requests, executions = [], []
            class Analysis:
                def analyze(self, args):
                    executions.append(deepcopy(args))
                    return {'artifact_id': 'analysis-unit', 'report': {'scope': 'fit_only'}}
            class Transport:
                def complete(self, request, timeout, context):
                    requests.append(deepcopy(request))
                    if len(requests) < 3:
                        args = ({'kind': 'summary', 'columns': ['x_numeric'], 'group_by': None, 'include_target': False}
                                if len(requests) == 1 else {'kind': 'report', 'report_id': 'analysis-unit'})
                        message = {'tool_calls': [{'id': 'read', 'function': {'name': 'analyze_data', 'arguments': json.dumps(args)}}]}
                    else:
                        message = {'content': json.dumps(action('catalog'))}
                    return {'choices': [{'message': message, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 100}}
            g = h.g
            gateway = Gateway(g.config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), Transport())
            gateway.analysis = Analysis()
            gateway.complete({'purpose': 'research_proposal', 'context_artifact_id': h.issue()['work_order']['context_id'],
                'response_schema_id': 'ResearchAction-v1', 'tool_schema_ids': ['analyze_data'],
                'max_output_tokens': 1024, 'deadline_seconds': 300, 'attempt_limit': 8})
            self.assertEqual(len(executions), 1)
            self.assertNotIn('tools', requests[-1])
            replies = [json.loads(m['content']) for m in requests[-1]['messages'] if m.get('role') == 'tool']
            self.assertTrue(replies[-1]['reused'])
        finally:
            h.close()
