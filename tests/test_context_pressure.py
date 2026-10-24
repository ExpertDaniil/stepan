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


class ResearchContextSchemaTests(unittest.TestCase):
    def test_runtime_metadata_passes_for_both_tasks_and_unknown_fields_still_fail(self):
        from tests.graph_helpers import GraphHarness
        from agent.semantic_planning import build_task_model, unknown_model
        for kind, background in [(kind, background) for kind in ('regression', 'classification') for background in (False, True)]:
            with self.subTest(kind=kind, background=background):
                h = GraphHarness(kind, research_mode='adaptive', background_tuning_enabled=background)
                try:
                    g = h.g
                    model = build_task_model(unknown_model(), '', h.h.profile['summary'], g.schemas, g.spec)
                    art = g.artifacts.write_json('task/context-schema-model.json', model, 'task_semantic_model')
                    g.spec.update(semantic_model_id=model['model_id'], semantic_model_artifact_id=art['artifact_id'])
                    ticket = h.issue()
                    context = g.artifacts.read_json(ticket['work_order']['context_id'])
                    self.assertTrue({'graph_request_max_bytes', 'detail_artifact_id', 'detail_scope',
                        'pipeline_execution' if background else 'semantic_contract'} <= set(context['metadata']))
                    g.schemas.validate(context, 'GResearchContext')
                    context['metadata']['unknown_runtime_flag'] = True
                    with self.assertRaises(AgentError):
                        g.schemas.validate(context, 'GResearchContext')
                finally:
                    h.close()


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


class ContextAssemblyPressureTests(unittest.TestCase):
    def check_history_pressure(self, kind):
        from agent.context_budget import assembly_limit
        from tests.graph_helpers import GraphHarness
        h = GraphHarness(kind, research_mode='adaptive', time_budget_seconds=10800)
        try:
            g = h.g
            # Inject only graph state and scores; no estimator fit in this fixture.
            h.result('researcher-1', {}, .6 if kind != 'regression' else 3)
            head = next(record for record in g.heads(h.root) if record['researcher_id'] == 'researcher-1')
            original_code = g.artifacts.resolve(g.store.get(head['experiment_id'])['plan']['code_artifact_id']).read_text()
            suffix = '\nAdditional task rules: '+('устойчивость '*500)
            g.context.task_text += suffix
            records = []
            for i in range(6):
                question = {'kind': 'summary', 'columns': ['x_numeric'], 'group_by': None, 'include_target': False,
                    'investigation': {'question': 'я'*400, 'basis_ids': [head['experiment_id']],
                        'columns': ['x_numeric'], 'interpretation': 'exploratory', 'mechanism': 'я'*500,
                        'assumptions': ['я'*200 for _ in range(4)], 'expected_observation': 'я'*400,
                        'alternative_explanation': 'я'*400, 'decision_rule': 'я'*500}}
                g.schemas.validate(question, 'AnalysisQuery')
                self.assertLessEqual(request_bytes(question), 8192)
                art = g.artifacts.write_json('analysis/synthetic-'+str(i)+'.json',
                    {'scope': 'fit_only', 'query': question, 'columns': [{'column': 'x_numeric', 'mean': i}]}, 'research_analysis')
                records.append({'artifact_id': art['artifact_id'], 'query': question,
                                'scope': 'fit_only', 'work_order_id': 'fixture-order'})
            g.context.analysis = SimpleNamespace(recent=lambda **kwargs: deepcopy(records))
            ticket = h.issue()
            context = g.artifacts.read_json(ticket['work_order']['context_id'])
            user = json.loads(context['messages'][1]['content'])
            limit = assembly_limit(g.config, context['metadata'])
            self.assertLessEqual(request_bytes(context['messages']), limit)
            self.assertTrue(user['profile_summary']['task_text'].endswith(suffix))
            self.assertEqual(user['base_code'], original_code)
            self.assertEqual(user['base_result']['hyperparameters'], g.store.get(head['experiment_id'])['plan']['model']['hyperparameters'])
            self.assertTrue(any('investigation_details' in r for r in user['profile_summary']['analysis_history']))
            source = g.artifacts.read_json(context['metadata']['detail_artifact_id'])
            self.assertEqual(source['sections']['analyses'][0]['query'], records[0]['query'])
            self.assertIn('investigation', records[0]['query'])
            g.schemas.validate(user, 'GContextUser')
        finally:
            h.close()

    def test_classification_reissue_fits_before_gateway_with_six_long_analysis_queries(self):
        self.check_history_pressure('classification')

    def test_regression_reissue_fits_before_gateway_with_six_long_analysis_queries(self):
        self.check_history_pressure('regression')

    def test_irreducible_assembly_saves_rejected_input_and_measured_byte_contributors(self):
        from tests.graph_helpers import GraphHarness
        h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800)
        try:
            h.g.context.task_text = 'A mandatory condition. '*20000
            with self.assertRaises(AgentError) as caught:
                h.issue()
            self.assertEqual(caught.exception.code, 'CONTEXT_TOO_LARGE')
            details = caught.exception.details
            self.assertGreater(details['context_bytes'], details['limit_bytes'])
            rejected = h.g.artifacts.read_json(details['rejected_context_artifact_id'])
            user = json.loads(rejected['messages'][1]['content'])
            self.assertEqual(user['profile_summary']['task_text'], h.g.context.task_text)
            self.assertGreater(details['profile_field_bytes']['task_text'], details['limit_bytes'])
        finally:
            h.close()


class DeferredPeerAccountingTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.controller = SimpleNamespace(research_stalls={},
            log=SimpleNamespace(event=lambda name, **data: self.events.append(data)))
        self.work = {'branch_id': 'branch-a', 'researcher_id': 'peer-a'}

    def test_one_context_failure_is_one_explicit_deferral(self):
        from agent.graph.research_scheduler import record_stall
        self.assertTrue(record_stall(self.controller, self.work, 'failure:CONTEXT_TOO_LARGE', 'generation-a',
                                     defer_reason='Context could not fit'))
        record = self.controller.research_stalls[('branch-a', 'peer-a')]
        self.assertEqual(record['count'], 1)
        self.assertTrue(record['deferred'])
        self.assertEqual(self.events[0]['repetitions'], 1)
        self.assertFalse(self.events[0]['heuristic'])

    def test_real_repetition_needs_two_actual_outcomes_and_new_measurement_resets_it(self):
        from agent.graph.research_scheduler import record_stall
        self.assertFalse(record_stall(self.controller, self.work, 'duplicate', 'generation-a'))
        self.assertTrue(record_stall(self.controller, self.work, 'duplicate', 'generation-a'))
        self.assertEqual(self.events[0]['repetitions'], 2)
        self.assertFalse(record_stall(self.controller, self.work, 'duplicate', 'generation-b'))
        self.assertEqual(self.controller.research_stalls[('branch-a', 'peer-a')]['count'], 1)

    def test_one_deferred_peer_does_not_block_its_neighbor(self):
        from agent.graph.research_scheduler import record_stall
        record_stall(self.controller, self.work, 'failure:CONTEXT_TOO_LARGE', 'generation-a', defer_reason='Context limit')
        neighbor = {**self.work, 'researcher_id': 'peer-b'}
        self.assertFalse(record_stall(self.controller, neighbor, 'duplicate', 'generation-a'))
        self.assertFalse(self.controller.research_stalls[('branch-a', 'peer-b')]['deferred'])

    def test_actual_refill_failure_is_counted_once_while_other_dialogue_finishes(self):
        from threading import Event
        from unittest.mock import patch
        from agent.graph.controller import research_batch
        from agent.research_feedback import FeedbackMailbox
        from tests.graph_helpers import GraphHarness
        h = GraphHarness('regression', research_mode='adaptive', time_budget_seconds=10800, background_tuning_enabled=False)
        release = Event()
        try:
            c, g = h.controller, h.g
            c.graph, c.degraded, c.research_stalls = g, False, {}
            c.call = lambda cid, payload, fn, key=None: fn(payload)
            c.llm_available = lambda pending=0: True
            c.set_fallback = lambda reason=None: None
            c.prepare_adaptive_proposal = lambda plan, *args: plan
            c.record_research_feedback = lambda *args, **kwargs: None
            c.local_search = SimpleNamespace(has_series=lambda *args: False, optuna=None)
            gateway = SimpleNamespace(feedback=FeedbackMailbox(g.artifacts), cancelled=Event())
            tickets = [h.issue(peer) for peer in ('researcher-1', 'researcher-2')]
            proposal = h.proposal(tickets[0])['proposal']
            def propose(payload):
                if payload['researcher_id'] == 'researcher-1':
                    return proposal
                if not release.wait(3):
                    raise AssertionError('Neighbor never reached the refill path')
                return {'decision': 'stop', 'plan': None, 'stop_reason': 'Synthetic completed dialogue'}
            c.researcher = SimpleNamespace(propose=propose, gateway=gateway)
            def train(plans, work):
                for plan, ticket in zip(plans, work):
                    payload = h.proposal(ticket)
                    payload['proposal']['plan'] = plan
                    payload['manifest'] = g.constructor.build(plan)
                    h.commit(g.invoke('G04', payload), 3)  # Injected score; no model fit.
                return len(plans)
            c.run_training = train
            def failed_issue(*args, **kwargs):
                release.set()
                raise AgentError('CONTEXT_TOO_LARGE', 'Synthetic early context pressure')
            with patch('agent.graph.controller.issue', side_effect=failed_issue):
                accepted, _ = research_batch(c, [], [], tickets)
            self.assertEqual(accepted, 1)
            self.assertTrue(c.degraded)
            self.assertEqual(c.research_stalls[(h.root, 'researcher-1')]['count'], 1)
            self.assertEqual(c.research_stalls[(h.root, 'researcher-2')]['signature'], 'stop')
        finally:
            release.set()
            h.close()
