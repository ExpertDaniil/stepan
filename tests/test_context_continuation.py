"""Context-pressure regressions use observable synthetic reports, never live APIs."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.context_budget import compact_research_request, request_bytes
from agent.context_details import read_details
from agent.context_transcript import batches, compact_history
from agent.context import output_schemas
from agent.errors import AgentError
from agent.llm.gateway import Gateway
from agent.researcher import Researcher
from tests.graph_helpers import GraphHarness


def turn(number, code_size=0, parallel=1):
    calls, replies = [], []
    for n in range(parallel):
        ident = 'call-'+str(number)+'-'+str(n)
        args = {'question': 'Проверить пропуски '+str(number), 'columns': ['x_numeric'],
            'code': 'def analyze(X_fit,y_fit,X_test,context):\n    return {}\n'+('# x\n'*(code_size//4))+'# variant '+str(n)}
        calls.append({'id': ident, 'type': 'function', 'function': {
            'name': 'run_analysis', 'arguments': json.dumps(args, ensure_ascii=False)}})
        replies.append({'role': 'tool', 'tool_call_id': ident, 'content': json.dumps({
            'artifact_id': 'report-'+ident, 'report': {'scope': 'fit_only', 'success': True,
                'result': {'missing_count': number, 'rmse': 42.125}}})})
    return [{'role': 'assistant', 'content': 'Compare missingness without changing the target.', 'tool_calls': calls}, *replies]


def request(*turns):
    user = {'branch_id': 'branch-a', 'task_contract': {'target': 'y', 'metric': 'rmse'},
        'base_code': 'def make_pipeline(task_spec,params):\n    return build(params)\n',
        'profile_summary': {'task_text': 'Keep the fit-only boundary.', 'data': {
            'train_columns': ['x_numeric', 'y'], 'test_columns': ['x_numeric']}}}
    return {'model': 'synthetic', 'max_tokens': 32768, 'messages': [
        {'role': 'system', 'content': 'Return a complete executable solution.'},
        {'role': 'user', 'content': json.dumps(user, ensure_ascii=False)},
        *[m for group in turns for m in group]]}


class TranscriptPackingTests(unittest.TestCase):
    def setUp(self):
        self.archives = []

    def archive(self, messages):
        self.archives.append(deepcopy(messages))
        return {'reference_id': 'art-'+str(len(self.archives)).zfill(16), 'section': 'updates'}

    def test_old_python_arguments_are_archived_at_the_original_96_kib_limit(self):
        value = request(turn(1, 22000, 4), turn(2))
        original = deepcopy(value)
        result = compact_research_request(value, 98304, archive=self.archive)
        self.assertLessEqual(request_bytes(value), 98304)
        self.assertTrue(result['archived_tool_call_ids'])
        self.assertEqual(json.loads(value['messages'][1]['content']), json.loads(original['messages'][1]['content']))
        self.assertEqual(value['messages'][-2], original['messages'][-2])
        self.assertEqual(json.loads(value['messages'][-1]['content']), json.loads(original['messages'][-1]['content']))
        self.assertEqual(self.archives, [original['messages']])
        summary = next(json.loads(m['content'])['research_transcript'] for m in value['messages']
            if m['role'] == 'user' and 'research_transcript' in json.loads(m['content']))
        self.assertIn('42.125', summary['records'][-1]['observation_preview'])
        self.assertEqual(summary['records'][-1]['scope'], 'fit_only')

    def test_seventy_two_extra_bytes_do_not_roll_over_the_dialogue(self):
        value = request(turn(1, 22000, 4), turn(2))
        limit = request_bytes(value)-72
        newest = deepcopy(value['messages'][-2:])
        removed = compact_history(value, limit, self.archive, deepcopy(value['messages']))
        self.assertTrue(removed)
        self.assertLessEqual(request_bytes(value), limit)
        self.assertEqual(value['messages'][-2:], newest)

    def test_five_observed_overflow_sizes_fit_on_synthetic_tool_transcripts(self):
        for target, count in [(98376, 15), (99970, 12), (102549, 9), (99779, 7), (101190, 17)]:
            with self.subTest(request_bytes=target, tool_pairs=count):
                value = request(*(turn(n) for n in range(count)))
                missing = target-request_bytes(value)
                for message in value['messages']:
                    if not message.get('tool_calls'):
                        continue
                    call = message['tool_calls'][0]
                    args = json.loads(call['function']['arguments'])
                    extra = min(20000, missing)
                    args['code'] += 'x'*extra
                    call['function']['arguments'] = json.dumps(args, ensure_ascii=False)
                    missing -= extra
                    if not missing:
                        break
                self.assertEqual(request_bytes(value), target)
                compact_research_request(value, 98304, archive=self.archive)
                self.assertLessEqual(request_bytes(value), 98304)

    def test_one_oversized_parallel_batch_keeps_all_exact_calls_in_archive(self):
        value = request(turn(1, 22000, 8))
        original = deepcopy(value['messages'])
        result = compact_research_request(value, 98304, archive=self.archive)
        self.assertEqual(len(result['archived_tool_call_ids']), 8)
        self.assertEqual(self.archives[0], original)
        self.assertLessEqual(request_bytes(value), 98304)
        self.assertFalse(batches(value['messages']))

    def test_incomplete_batches_and_protected_contract_are_not_removed(self):
        value = request(turn(1, 22000, 4))
        value['messages'].pop()
        original = deepcopy(value)
        self.assertEqual(compact_history(value, 1000, self.archive, value['messages']), [])
        self.assertEqual(value, original)
        self.assertFalse(self.archives)

    def test_irreducible_task_is_not_silently_truncated(self):
        value = request(turn(1))
        user = json.loads(value['messages'][1]['content'])
        user['profile_summary']['task_text'] = 'Required condition. '*20000
        value['messages'][1]['content'] = json.dumps(user)
        compact_research_request(value, 98304, archive=self.archive)
        self.assertGreater(request_bytes(value), 98304)
        self.assertEqual(json.loads(value['messages'][1]['content'])['profile_summary']['task_text'], user['profile_summary']['task_text'])
        self.assertFalse(self.archives)

    def test_repeated_compaction_has_one_bounded_summary_and_preserves_archive_chain(self):
        value = request(turn(1, 22000, 4), turn(2))
        compact_research_request(value, 98304, archive=self.archive)
        first_reference = 'art-'+str(1).zfill(16)
        value['messages'].extend(turn(3, 22000, 4))
        compact_research_request(value, 98304, archive=self.archive)
        summaries = [m for m in value['messages'] if m['role'] == 'user' and 'research_transcript' in json.loads(m['content'])]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(len(self.archives), 2)
        self.assertIn(first_reference, json.dumps(self.archives[1]))
        self.assertLessEqual(request_bytes(value), 98304)
        unchanged = deepcopy(value)
        compact_research_request(value, 98304, archive=self.archive)
        self.assertEqual(value, unchanged)
        self.assertEqual(len(self.archives), 2)

    def test_utf8_and_parallel_tool_linkage_survive(self):
        value = request(turn(1, 22000, 4), turn(2, parallel=3))
        compact_research_request(value, 98304, archive=self.archive)
        json.dumps(value, ensure_ascii=False).encode('utf-8').decode('utf-8')
        calls = [c['id'] for m in value['messages'] for c in m.get('tool_calls', [])]
        replies = [m['tool_call_id'] for m in value['messages'] if m['role'] == 'tool']
        self.assertEqual(calls, replies)

    def test_varied_transcripts_fit_without_orphaning_calls_or_changing_the_task(self):
        rng = random.Random(42)
        for case in range(20):
            with self.subTest(case=case):
                self.archives.clear()
                value = request(*(turn(n, rng.randrange(1000, 24000), rng.randrange(1, 5))
                    for n in range(rng.randrange(1, 7))))
                task = json.loads(value['messages'][1]['content'])
                limit = rng.choice([12000, 98304, 196608])
                compact_research_request(value, limit, archive=self.archive)
                self.assertLessEqual(request_bytes(value), limit)
                self.assertEqual(json.loads(value['messages'][1]['content']), task)
                calls = [c['id'] for m in value['messages'] for c in m.get('tool_calls', [])]
                replies = [m['tool_call_id'] for m in value['messages'] if m['role'] == 'tool']
                self.assertEqual(calls, replies)


class GatewayContinuationTests(unittest.TestCase):
    def setUp(self):
        self.h = GraphHarness('regression', research_mode='adaptive', graph_hard_decisions=False,
            time_budget_seconds=10800, llm_max_calls=None, llm_max_total_output_tokens=None)
        g = self.h.g
        self.gateway = Gateway(g.config, g.budget, g.artifacts, g.bus, g.log, output_schemas(g.bus), SimpleNamespace())
        self.gateway.budget.reasoning_estimate = lambda: 1

    def tearDown(self):
        self.h.close()

    def test_archives_are_readable_only_in_the_issued_scope(self):
        context = self.h.g.artifacts.read_json(self.h.issue()['work_order']['context_id'])
        messages = request(turn(1))['messages']
        ref = self.gateway._archive_transcript(messages, context)
        reply = read_details(ref, context, self.h.g.artifacts, self.h.g.log)
        self.assertIn('call-1-0', reply['text'])
        other = self.h.g.artifacts.read_json(self.h.issue('researcher-2')['work_order']['context_id'])
        with self.assertRaises(AgentError) as caught:
            read_details(ref, other, self.h.g.artifacts, self.h.g.log)
        self.assertEqual(caught.exception.code, 'MEMORY_SCOPE_DENIED')
        other['metadata']['authorized_feedback_ids'] = [ref['reference_id']]
        with self.assertRaises(AgentError) as caught:
            read_details(ref, other, self.h.g.artifacts, self.h.g.log)
        self.assertEqual(caught.exception.code, 'MEMORY_SCOPE_DENIED')

    def test_archive_results_are_directly_readable_and_modified_bytes_are_rejected(self):
        g = self.h.g
        context = g.artifacts.read_json(self.h.issue()['work_order']['context_id'])
        ref = self.gateway._archive_transcript(request(turn(1, 22000))['messages'], context)
        query = {**ref, 'section': 'tool/call-1-0/result'}
        result = read_details(query, context, g.artifacts, g.log)
        self.assertEqual(json.loads(result['text'])['report']['result']['rmse'], 42.125)
        self.assertNotIn('def analyze', result['text'])
        g.artifacts.resolve(ref['reference_id']).write_text('{}')
        with self.assertRaises(AgentError):
            read_details(query, context, g.artifacts, g.log)

    def test_visible_decision_exposes_archive_in_a_fresh_context(self):
        ticket = self.h.issue()
        context = self.h.g.artifacts.read_json(ticket['work_order']['context_id'])
        ref = self.gateway._archive_transcript(request(turn(1))['messages'], context)
        from tests.test_adaptive_research import action
        value = action('research')
        art = self.h.g.artifacts.write_json('research/synthetic-output.json', value, 'llm_output')
        researcher = Researcher(self.h.g.bus, None, self.h.g.artifacts, self.h.g.spec,
            self.h.g.catalog, self.h.g.config, self.h.g.log)
        researcher.save_decision(value, None, ticket['researcher_request']['payload'], context,
            {'output_artifact_id': art['artifact_id'], 'tool_calls': [{'name': 'transcript_archive', **ref}]})
        fresh = self.h.g.artifacts.read_json(self.h.issue('researcher-2')['work_order']['context_id'])
        snapshot = self.h.g.artifacts.read_json(fresh['metadata']['detail_artifact_id'])
        self.assertTrue(snapshot['sections']['transcripts_index'])
        query = {'reference_id': fresh['metadata']['detail_artifact_id'], 'section': 'transcript/'+ref['reference_id']}
        index = json.loads(read_details(query, fresh, self.h.g.artifacts, self.h.g.log)['text'])
        self.assertEqual(index[0]['call_id'], 'call-1-0')
        query['section'] = index[0]['result_section']
        self.assertEqual(json.loads(read_details(query, fresh, self.h.g.artifacts, self.h.g.log)['text'])['report']['result']['rmse'], 42.125)

    def test_abandoned_dialogue_archive_survives_without_a_final_research_decision(self):
        g = self.h.g
        ticket = self.h.issue()
        context = g.artifacts.read_json(ticket['work_order']['context_id'])
        ref = self.gateway._archive_transcript(request(turn(1))['messages'], context)
        g.abandon(ticket['work_order']['work_order_id'], 'Synthetic provider failure after analysis')
        fresh = g.artifacts.read_json(self.h.issue('researcher-2')['work_order']['context_id'])
        snapshot = g.artifacts.read_json(fresh['metadata']['detail_artifact_id'])
        self.assertIn('transcript/'+ref['reference_id'], snapshot['documents'])

    def test_transcript_visibility_respects_the_exact_fork_time_cut(self):
        from agent.context_transcript import visible_transcripts
        g = self.h.g
        ticket = self.h.issue()
        context = g.artifacts.read_json(ticket['work_order']['context_id'])
        ref = self.gateway._archive_transcript(request(turn(1))['messages'], context)
        archived = g.artifacts.read_json(ref['reference_id'])
        owner = ticket['work_order']['work_order_id']
        memory = {'decisions': [], 'visible_work_order_ids': [owner],
            'visible_analysis_cutoffs': {owner: archived['completed_elapsed_ms']-1}}
        self.assertEqual(list(visible_transcripts(memory, g.artifacts, context['metadata']['detail_scope'])), [])
        memory['visible_analysis_cutoffs'][owner] += 1
        self.assertEqual(len(list(visible_transcripts(memory, g.artifacts, context['metadata']['detail_scope']))), 1)
        memory['visible_work_order_ids'] = ['sibling-order']
        self.assertEqual(list(visible_transcripts(memory, g.artifacts, context['metadata']['detail_scope'])), [])

    def test_long_code_batch_reaches_final_model_decision_without_controller_checkpoint(self):
        g = self.h.g
        requests = []
        class Transport:
            def complete(inner, value, timeout, context):
                requests.append(deepcopy(value))
                if len(requests) == 1:
                    message = turn(1, 22000, 4)[0]
                elif len(requests) == 2:
                    self.assertNotIn('call-1-0', json.dumps([m for m in value['messages'] if m.get('tool_calls')]))
                    self.assertIn('research_transcript', json.dumps(value))
                    message = {'tool_calls': [{'id': 'small-check', 'type': 'function', 'function': {
                        'name': 'analyze_data', 'arguments': json.dumps({'kind': 'summary', 'columns': ['x_numeric'],
                            'group_by': None, 'include_target': False})}}]}
                else:
                    self.assertNotIn('call-1-0', json.dumps([m for m in value['messages'] if m.get('tool_calls')]))
                    message = {'content': json.dumps({'action': 'catalog', 'recipe_id': 'linear-regressor-v1',
                        'hyperparameters': {'alpha': 1}, 'hypothesis': 'Use the completed fit-only observations.'})}
                return {'choices': [{'message': message, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 10}}
        self.gateway.transport = Transport()
        from agent.research_feedback import FeedbackMailbox
        self.gateway.feedback = FeedbackMailbox(g.artifacts)
        self.gateway.feedback.publish(self.h.root, {'branch_id': self.h.root,
            'task_spec_id': g.spec['task_spec_id'], 'dataset_id': g.spec['dataset_id'],
            'validation_id': g.validation['validation_id'], 'results': [], 'series': []})
        self.gateway.analysis = SimpleNamespace(analyze=lambda *a, **k: {'artifact_id': 'small-report', 'report': {'scope': 'fit_only'}})
        def report(service, args, context):
            return {'artifact_id': 'synthetic-report', 'report': {'scope': 'fit_only', 'success': True, 'result': {'count': 3}}}
        with patch('tools.python_analysis.run_analysis', side_effect=report):
            answer = self.gateway.complete({'purpose': 'research_proposal', 'context_artifact_id': self.h.issue()['work_order']['context_id'],
                'response_schema_id': 'ResearchAction-v1', 'tool_schema_ids': ['run_analysis', 'analyze_data'],
                'max_output_tokens': 1024, 'deadline_seconds': 300, 'attempt_limit': 6})
        self.assertEqual(answer['parsed_payload']['action'], 'catalog')
        self.assertEqual(len(requests), 3)
        self.assertEqual(g.artifacts.meta(answer['output_artifact_id'])['kind'], 'llm_output')
        self.assertTrue(any(c['name'] == 'transcript_archive' for c in answer['tool_calls']))
        self.assertTrue(all(request_bytes(r) <= 98304 for r in requests))
        for value in requests:
            updates = [m for m in value['messages'] if m['role'] == 'user' and 'research_updates' in json.loads(m['content'])]
            self.assertEqual(len(updates), 1)
        self.assertFalse(g.budget.llm_dialogues)

    def test_first_502_on_ninth_turn_does_not_wait_until_dialogue_deadline(self):
        g = self.h.g
        ticket = self.h.issue()
        context = g.artifacts.read_json(ticket['work_order']['context_id'])
        calls = []
        class Transport:
            def complete(inner, value, timeout, ctx):
                calls.append(deepcopy(value))
                n = len(calls)
                if n <= 8:
                    message = {'tool_calls': [{'id': 'page-'+str(n), 'type': 'function', 'function': {
                        'name': 'read_memory', 'arguments': json.dumps({'reference_id': context['metadata']['detail_artifact_id'],
                            'section': 'task_description', 'offset': n-1, 'limit_bytes': 256})}}]}
                elif n == 9:
                    error = RuntimeError('synthetic upstream error');error.status_code = 502
                    raise error
                else:
                    message = {'content': json.dumps({'action': 'catalog', 'recipe_id': 'linear-regressor-v1',
                        'hyperparameters': {'alpha': 1}, 'hypothesis': 'Finish from the completed observations.'})}
                return {'choices': [{'message': message, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 10}}
        self.gateway.transport = Transport()
        with patch.object(self.gateway.cancelled, 'wait', return_value=False) as pause:
            result = self.gateway.complete({'purpose': 'research_proposal', 'context_artifact_id': ticket['work_order']['context_id'],
                'response_schema_id': 'ResearchAction-v1', 'tool_schema_ids': ['read_memory'],
                'max_output_tokens': 1024, 'deadline_seconds': 300, 'attempt_limit': 10})
        self.assertEqual(result['parsed_payload']['action'], 'catalog')
        self.assertEqual(len(calls), 10)
        pause.assert_called_once_with(1)
        self.assertNotIn('tools', calls[-1])
        self.assertFalse(g.budget.llm_dialogues)

    def test_two_peers_compact_independently_without_sharing_archive_permissions(self):
        g = self.h.g
        tickets = [self.h.issue(peer) for peer in ('researcher-1', 'researcher-2')]
        counts, references = {}, {}
        class Transport:
            def complete(inner, value, timeout, context):
                peer = context['metadata']['researcher_id']
                counts[peer] = counts.get(peer, 0)+1
                if counts[peer] == 1:
                    message = turn(1, 22000, 4)[0]
                else:
                    summary = next(json.loads(m['content'])['research_transcript'] for m in value['messages']
                        if m['role'] == 'user' and 'research_transcript' in json.loads(m['content']))
                    references[peer] = summary['archive']['reference_id']
                    self.assertIn(references[peer], context['metadata']['authorized_feedback_ids'])
                    message = {'content': json.dumps({'action': 'catalog', 'recipe_id': 'linear-regressor-v1',
                        'hyperparameters': {'alpha': 1}, 'hypothesis': 'Complete this peer research.'})}
                return {'choices': [{'message': message, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 10}}
        self.gateway.transport = Transport()
        def complete(ticket):
            return self.gateway.complete({'purpose': 'research_proposal', 'context_artifact_id': ticket['work_order']['context_id'],
                'response_schema_id': 'ResearchAction-v1', 'tool_schema_ids': ['run_analysis'],
                'max_output_tokens': 1024, 'deadline_seconds': 300, 'attempt_limit': 6})
        with patch('tools.python_analysis.run_analysis', return_value={'artifact_id': 'report',
                'report': {'scope': 'fit_only', 'success': True, 'result': {'count': 1}}}):
            with ThreadPoolExecutor(max_workers=2) as pool:
                answers = list(pool.map(complete, tickets))
        self.assertTrue(all(a['parsed_payload']['action'] == 'catalog' for a in answers))
        self.assertEqual(len(set(references.values())), 2)
        for ticket in tickets:
            peer = ticket['work_order']['researcher_id']
            archive = g.artifacts.read_json(references[peer])
            self.assertEqual(archive['scope']['work_order_id'], ticket['work_order']['work_order_id'])

    def test_irreducible_limit_saves_explicit_checkpoint_without_claiming_model_decision(self):
        g = self.h.g
        class Transport:
            def complete(inner, value, timeout, context):
                context['metadata']['graph_request_max_bytes'] = 1024
                return {'choices': [{'message': {'tool_calls': [{'id': 'inspect', 'type': 'function', 'function': {
                    'name': 'analyze_data', 'arguments': json.dumps({'kind': 'summary', 'columns': ['x_numeric'],
                        'group_by': None, 'include_target': False})}}]}, 'finish_reason': 'stop'}],
                    'usage': {'completion_tokens': 10}}
        self.gateway.transport = Transport()
        self.gateway.analysis = SimpleNamespace(analyze=lambda *a, **k: {'artifact_id': 'saved-report', 'report': {'scope': 'fit_only'}})
        result = self.gateway.complete({'purpose': 'research_proposal', 'context_artifact_id': self.h.issue()['work_order']['context_id'],
            'response_schema_id': 'ResearchAction-v1', 'tool_schema_ids': ['analyze_data'],
            'max_output_tokens': 1024, 'deadline_seconds': 300, 'attempt_limit': 6})
        self.assertEqual(g.artifacts.meta(result['output_artifact_id'])['kind'], 'controller_research_checkpoint')
        events = [json.loads(l) for l in (g.artifacts.root/'events.jsonl').read_text().splitlines()]
        summary = next(r for r in events if r['event'] == 'research.dialogue_summary')
        self.assertFalse(summary['decision_received'])
        self.assertTrue(summary['checkpoint_saved'])
        self.assertIn('transcripts_index', result['parsed_payload']['investigation']['decision_rule'])


if __name__ == '__main__':
    unittest.main()
