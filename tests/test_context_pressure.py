"""Synthetic regressions for context failures; written, not run in this session."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

from agent.context_budget import compact_tool_reply, request_bytes
from agent.contracts import Schemas
from agent.errors import AgentError
from agent.llm.gateway import Gateway
from agent.semantic_planning import validate_investigation


class ContextPressureTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.gateway = Gateway.__new__(Gateway)
        self.gateway.log = SimpleNamespace(event=lambda *a, **k: self.events.append((a, k)))

    def request(self, replies):
        user = {'branch_id': 'branch-a', 'base_experiment_id': 'exp-base',
            'base_code': 'def make_pipeline(task_spec, params):\n    return trusted_factory(params)\n',
            'task_contract': {'target': 'label', 'metric': 'f1_macro', 'direction': 'maximize'},
            'base_result': {'experiment_id': 'exp-base', 'recipe_id': 'trees', 'value': .6},
            'facts': [], 'profile_summary': {
                'task_text': 'Условие и ограничения. '*1700,
                'data': {'train_columns': ['amount', 'sector', 'label'],
                         'test_columns': ['amount', 'sector'],
                         'core_profile': {'columns': [['amount', 'float', 0, 0], ['sector', 'str', 0, 0]]}},
                'semantic_task': {'model_id': 'model-a', 'target': {'meaning': 'row class'}}}}
        calls = [{'id': 'call-'+str(i), 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps({'question': 'Inspect a synthetic distribution'})}}
            for i, (name, _) in enumerate(replies)]
        return {'model': 'synthetic', 'max_tokens': 32768,
            'tools': [{'type': 'function', 'function': {'name': 'analyze_data', 'parameters': {'type': 'object'}}}],
            'messages': [{'role': 'system', 'content': 'Required output contract. '*700},
                {'role': 'user', 'content': json.dumps(user, ensure_ascii=False)},
                {'role': 'assistant', 'content': None, 'tool_calls': calls},
                *[{'role': 'tool', 'tool_call_id': 'call-'+str(i), 'content': json.dumps(value, ensure_ascii=False)}
                  for i, (_, value) in enumerate(replies)]]}

    @staticmethod
    def knowledge():
        return {'artifact_id': 'knowledge-a', 'scope': 'external_experience', 'results': [
            {'id': 'method-a', 'kind': 'method', 'title': 'Rare groups',
             'mechanism': 'A candidate method, not current-task evidence.',
             'cautions': ['Do not fit on holdout labels.'], 'code_example': 'valid_example\n'*2200}]}

    def test_latest_knowledge_reply_can_fit_without_losing_the_task_or_base(self):
        request = self.request([('search_knowledge', self.knowledge())])
        before = json.loads(request['messages'][1]['content'])
        self.assertGreater(request_bytes(request), 98304)
        calls = deepcopy(request['messages'][2]['tool_calls'])
        self.gateway._compact_research_request(request, 98304)
        self.assertLessEqual(request_bytes(request), 98304)
        after = json.loads(request['messages'][1]['content'])
        for field in ('task_contract', 'base_result', 'base_code', 'branch_id'):
            self.assertEqual(after[field], before[field])
        self.assertEqual(after['profile_summary']['data'], before['profile_summary']['data'])
        self.assertEqual(after['profile_summary']['semantic_task'], before['profile_summary']['semantic_task'])
        self.assertEqual(request['messages'][2]['tool_calls'], calls)
        result = json.loads(request['messages'][-1]['content'])
        self.assertEqual(result['results'][0]['id'], 'method-a')
        self.assertEqual(result['scope'], 'external_experience')
        self.assertNotIn('code_example', result['results'][0])

    def test_parallel_analysis_batch_retains_scope_support_and_observations(self):
        reports = [('analyze_data', {'artifact_id': 'report-'+str(i), 'report': {
            'scope': 'fit_only', 'validation_id': 'validation-a', 'fit_rows': 90000,
            'analyzed_rows': 20000, 'sampled': True, 'query': {'kind': 'group', 'columns': ['sector']},
            'groups': [{'group': 'sector-'+str(j), 'rows': 100+j,
                        'class_counts': {'minority': 2, 'majority': 98}, 'definition': 'synthetic '*60}
                       for j in range(12)]}}) for i in range(4)]
        request = self.request([*reports, ('search_knowledge', self.knowledge())])
        self.assertGreater(request_bytes(request), 98304)
        self.gateway._compact_research_request(request, 98304)
        self.assertLessEqual(request_bytes(request), 98304)
        for message in request['messages'][3:7]:
            reply = json.loads(message['content'])
            self.assertEqual(reply['report']['scope'], 'fit_only')
            self.assertEqual(reply['report']['analyzed_rows'], 20000)
            self.assertTrue(reply['report']['groups'])
            self.assertEqual(reply['report']['groups'][0]['class_counts']['minority'], 2)
        calls = [c['id'] for m in request['messages'] for c in m.get('tool_calls', [])]
        self.assertEqual(calls, [m['tool_call_id'] for m in request['messages'] if m['role'] == 'tool'])

    def test_text_page_compaction_keeps_utf8_offsets_and_does_not_claim_a_complete_record(self):
        value = {'reference_id': 'frozen-a', 'section': 'semantics/0', 'scope': {'branch': 'a'},
            'offset': 73, 'total_bytes': 18073, 'next_offset': None,
            'text': 'Определение. '*800, 'evidence_ids': ['record-a'], 'idea_ids': ['idea-a']}
        original = deepcopy(value)
        projected = compact_tool_reply(value, 1600)
        self.assertLessEqual(request_bytes(projected), 1600)
        self.assertEqual(projected['next_offset'], 73+len(projected['text'].encode('utf-8')))
        self.assertEqual(projected['total_bytes'], 18073)
        self.assertEqual(projected['scope'], value['scope'])
        self.assertEqual(projected['evidence_ids'], [])
        self.assertEqual(projected['idea_ids'], [])
        self.assertTrue(original['text'].startswith(projected['text']))
        self.assertEqual(value, original)

    def test_full_knowledge_code_is_removed_whole_and_remains_in_the_original(self):
        value = self.knowledge()
        projected = compact_tool_reply(value, 1800)
        self.assertLessEqual(request_bytes(projected), 1800)
        record = projected['results'][0]
        self.assertNotIn('code_example', record)
        self.assertIn('code_example', value['results'][0])
        self.assertIn('method-a', record['detail_hint'])
        self.assertEqual(record['cautions'], value['results'][0]['cautions'])

    def test_analysis_compaction_retains_exact_counts_and_marks_omitted_groups(self):
        value = {'artifact_id': 'report-a', 'report': {'scope': 'fit_only', 'fit_rows': 1000,
            'query': {'kind': 'group', 'columns': ['sector'], 'investigation': {'mechanism': 'prose '*2000}},
            'groups': [{'group': str(i), 'rows': 50, 'violation_count': i} for i in range(12)]}}
        original = deepcopy(value)
        projected = compact_tool_reply(value, 600)
        self.assertLess(len(projected['report']['groups']), len(value['report']['groups']))
        self.assertEqual(projected['report']['groups'][0], value['report']['groups'][0])
        self.assertEqual(projected['report']['fit_rows'], 1000)
        self.assertEqual(projected['report']['display_omitted_groups'], 12-len(projected['report']['groups']))
        self.assertEqual(value, original)

    def test_irreducible_contract_is_preserved_instead_of_silently_cut(self):
        request = self.request([])
        original = deepcopy(request['messages'][:2])
        self.gateway._compact_research_request(request, 1024)
        self.assertGreater(request_bytes(request), 1024)
        self.assertEqual(request['messages'][0], original[0])
        self.assertEqual(json.loads(request['messages'][1]['content']), json.loads(original[1]['content']))


class DiagnosticCitationTests(unittest.TestCase):
    def test_analysis_accepts_a_fit_diagnostic_already_in_the_issued_context(self):
        investigation = {'question': 'Which groups account for the error?', 'basis_ids': ['fit-report-a'],
            'columns': ['amount'], 'interpretation': 'statistical', 'mechanism': 'Different group composition.',
            'assumptions': [], 'expected_observation': 'Different class proportions.',
            'alternative_explanation': 'Sampling variation.', 'decision_rule': 'Choose a measured follow-up.'}
        context = {'metadata': {'semantic_contract': 'semantic-planning-1'}, 'messages': [{}, {'content': json.dumps({
            'profile_summary': {'scientific_protocol': {'fit_diagnostics': [
                {'artifact_id': 'fit-report-a', 'experiment_id': 'exp-a'}]}}})}]}
        result = validate_investigation(investigation, context, set(), ['amount'], Schemas())
        self.assertEqual(result['basis_ids'], ['fit-report-a'])

    def test_other_branch_or_future_diagnostic_is_still_rejected(self):
        investigation = {'question': 'Inspect the error.', 'basis_ids': ['not-issued'], 'columns': ['amount'],
            'interpretation': 'exploratory', 'mechanism': 'Unknown.', 'assumptions': [],
            'expected_observation': 'A distribution.', 'alternative_explanation': 'No pattern.',
            'decision_rule': 'Decide after measuring.'}
        context = {'metadata': {}, 'messages': [{}, {'content': json.dumps({'profile_summary': {}})}]}
        with self.assertRaises(AgentError) as caught:
            validate_investigation(investigation, context, set(), ['amount'], Schemas())
        self.assertEqual(caught.exception.code, 'INVALID_INVESTIGATION')
